"""CUDA scheduling contracts exercised on CPU with bounded, explicit CUDA stubs.

These tests simulate CUDA failures; real GPU measurements are covered by the
separate hardware validation, never inferred from the mocked memory numbers.
"""

from contextlib import nullcontext
from types import SimpleNamespace
import sys
import weakref

import pandas as pd
import pytest
import torch

from deepkoala import adaptive, cli
from deepkoala.batching import GIB, ProteinRecord, plan_memory


MIB = 2 ** 20


@pytest.fixture
def cuda_stub(monkeypatch):
    state = {
        "total": 8 * GIB,
        "free": 6 * GIB,
        "allocated": 150 * MIB,
        "reserved": 200 * MIB,
        "peak_allocated": 150 * MIB,
        "peak_reserved": 200 * MIB,
        "empty_calls": 0,
        "reset_calls": 0,
        "synchronize_calls": 0,
    }

    def empty_cache():
        state["empty_calls"] += 1

    def reset_peak(device):
        state["reset_calls"] += 1
        state["peak_allocated"] = state["allocated"]
        state["peak_reserved"] = state["reserved"]

    def synchronize(device):
        state["synchronize_calls"] += 1

    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (state["free"], state["total"]))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: state["allocated"])
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: state["reserved"])
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: state["peak_allocated"])
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: state["peak_reserved"])
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_peak)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    return state


def _records(lengths, *, start=0):
    return [ProteinRecord(index, "duplicate", "A" * length)
            for index, length in enumerate(lengths, start)]


def _result(records):
    # Distinct ordinal-dependent values reveal incorrect reordering of duplicate IDs.
    return [[record.index] for record in records], [[record.index / 100] for record in records]


def _bare_predictor(*, max_batch_size=4, residue_limit=100, fallback="error"):
    predictor = adaptive.AdaptivePredictor.__new__(adaptive.AdaptivePredictor)
    predictor.model = object()
    predictor.device = torch.device("cuda")
    predictor.topk = 1
    predictor.max_batch_size = max_batch_size
    predictor.residue_limit = residue_limit
    predictor.oom_fallback = fallback
    predictor.cpu_model_factory = None
    predictor.cpu_model = None
    predictor.oom_retries = predictor.cpu_fallbacks = 0
    predictor.gpu_unavailable_reason = None
    predictor.max_successful_padded = predictor.max_successful_batch = 0
    predictor.max_peak_allocated = predictor.max_peak_reserved = 0
    predictor.allocator_budget = GIB
    predictor.bytes_per_residue = 12 * 1024
    predictor.plan = plan_memory(8 * GIB, 6 * GIB)
    return predictor


def test_constructor_calibrates_real_batch_shapes_and_respects_explicit_lower_ceiling(monkeypatch, cuda_stub):
    calls = []
    measured_bytes_per_residue = 24 * 1024

    def measured_prediction(model, records, device, topk):
        assert device.type == "cuda"
        calls.append((len(records), records[0].length))
        cuda_stub["peak_allocated"] = cuda_stub["allocated"] + len(records) * records[0].length * measured_bytes_per_residue
        cuda_stub["peak_reserved"] = cuda_stub["peak_allocated"] + 10 * MIB
        return _result(records)

    monkeypatch.setattr(adaptive, "predict_records", measured_prediction)
    predictor = adaptive.AdaptivePredictor(
        SimpleNamespace(cls=SimpleNamespace(out_features=2)), "cuda", 1,
        memory_preset="4", max_batch_size=7, max_padded_residues=123,
    )

    assert calls == [(4, 256), (4, 1024), (16, 256)]
    assert predictor.plan.preset.name == "4"
    assert predictor.plan.budget_bytes == int(3.2 * GIB)
    assert predictor.allocator_budget == predictor.plan.budget_bytes - adaptive.CUDA_OVERHEAD_BYTES
    assert predictor.bytes_per_residue == pytest.approx(1.35 * measured_bytes_per_residue)
    assert predictor.residue_limit == 123
    assert predictor.max_batch_size == 7
    assert predictor.max_peak_allocated == cuda_stub["allocated"] + 4096 * measured_bytes_per_residue
    assert predictor.max_peak_reserved == predictor.max_peak_allocated + 10 * MIB
    assert cuda_stub["reset_calls"] == 3
    assert cuda_stub["synchronize_calls"] == 6
    assert cuda_stub["empty_calls"] >= 2


def test_constructor_calibration_oom_keeps_conservative_budget_and_records_peak(monkeypatch, cuda_stub, caplog):
    def oom(model, records, device, topk):
        cuda_stub["peak_allocated"] = 300 * MIB
        cuda_stub["peak_reserved"] = 350 * MIB
        raise torch.cuda.OutOfMemoryError("probe failed")

    monkeypatch.setattr(adaptive, "predict_records", oom)
    predictor = adaptive.AdaptivePredictor(SimpleNamespace(cls=SimpleNamespace(out_features=2)), "cuda", 1)
    assert predictor.bytes_per_residue >= 12 * 1024
    assert predictor.residue_limit > 0
    assert predictor.max_peak_allocated == 300 * MIB
    assert predictor.max_peak_reserved == 350 * MIB
    assert "calibration OOM" in caplog.text
    assert cuda_stub["reset_calls"] == 1


def test_constructor_skips_calibration_when_headroom_is_too_small(monkeypatch, cuda_stub, caplog):
    cuda_stub.update(total=2 * GIB, free=10 * MIB, reserved=150 * MIB)

    def should_not_probe(*args, **kwargs):
        pytest.fail("Calibration must not run without headroom")

    monkeypatch.setattr(adaptive, "predict_records", should_not_probe)
    predictor = adaptive.AdaptivePredictor(SimpleNamespace(cls=SimpleNamespace(out_features=2)), "cuda", 1,
                                           oom_fallback="cpu", cpu_model_factory=object)
    assert predictor.residue_limit == 1
    assert "insufficient GPU headroom" in predictor.gpu_unavailable_reason
    assert cuda_stub["reset_calls"] == 0
    assert "too small for calibration" in caplog.text


def test_insufficient_model_headroom_reports_error_if_cpu_fallback_is_disabled(cuda_stub):
    cuda_stub.update(total=2 * GIB, free=10 * MIB, reserved=150 * MIB)
    with pytest.raises(RuntimeError, match="insufficient GPU headroom.*--oom-fallback cpu"):
        adaptive.AdaptivePredictor(SimpleNamespace(cls=SimpleNamespace(out_features=2)), "cuda", 1)


def test_insufficient_headroom_routes_even_short_sequences_to_cpu_lazily(monkeypatch, cuda_stub):
    cuda_stub.update(total=2 * GIB, free=10 * MIB, reserved=150 * MIB)
    created = []
    calls = []
    gpu_model = SimpleNamespace(cls=SimpleNamespace(out_features=2))

    def factory():
        model = object()
        created.append(model)
        return model

    def cpu_only(model, records, device, topk):
        assert device.type == "cpu"
        assert model is created[0]
        calls.append([record.index for record in records])
        return _result(records)

    monkeypatch.setattr(adaptive, "predict_records", cpu_only)
    predictor = adaptive.AdaptivePredictor(gpu_model, "cuda", 1,
                                           oom_fallback="cpu", cpu_model_factory=factory)
    assert created == []
    result = predictor.predict_window(_records([1, 2]))
    assert calls == [[0], [1]]
    assert len(created) == 1
    assert [record.index for record, _, _ in result] == [0, 1]
    assert predictor.model is gpu_model
    assert predictor.cpu_fallbacks == 2
    assert predictor.oom_retries == 0
    assert cuda_stub["reset_calls"] == 0


def test_limited_classifier_headroom_reduces_batch_cap_without_exceeding_user_limit(cuda_stub):
    baseline = int(1.6 * GIB) - adaptive.CUDA_OVERHEAD_BYTES - MIB
    cuda_stub.update(total=2 * GIB, free=2 * GIB - baseline, reserved=baseline,
                     allocated=baseline, peak_allocated=baseline, peak_reserved=baseline)
    predictor = adaptive.AdaptivePredictor(SimpleNamespace(cls=SimpleNamespace(out_features=10_000)), "cuda", 1,
                                           max_batch_size=512)
    assert predictor.max_batch_size == 4
    assert predictor.residue_limit > 1
    assert predictor.gpu_unavailable_reason is None
    assert cuda_stub["reset_calls"] == 0


def test_try_predict_drops_failed_frame_and_retained_tensors_before_return(monkeypatch):
    references = []

    class TemporaryTensor:
        pass

    def oom(*args):
        temporary = TemporaryTensor()
        references.append(weakref.ref(temporary))
        raise torch.cuda.OutOfMemoryError("simulated CUDA OOM")

    monkeypatch.setattr(adaptive, "predict_records", oom)
    result, error = adaptive.try_predict(object(), _records([3]), torch.device("cuda"), 1)
    assert result is None
    assert error == "simulated CUDA OOM"
    assert references[0]() is None


@pytest.mark.parametrize("error", [RuntimeError("CUDA illegal memory access"), ValueError("bad input"), MemoryError("host RAM")])
def test_only_cuda_oom_is_caught(monkeypatch, cuda_stub, error):
    def fail(*args):
        raise error

    monkeypatch.setattr(adaptive, "predict_records", fail)
    predictor = _bare_predictor(fallback="cpu")
    with pytest.raises(type(error), match=str(error)):
        predictor.predict_window(_records([3, 2]))
    assert predictor.oom_retries == 0
    assert predictor.cpu_fallbacks == 0
    assert cuda_stub["empty_calls"] == 0


def test_oom_splits_once_releases_traceback_and_changes_future_batches(monkeypatch, cuda_stub, caplog):
    calls = []
    references = []

    class TemporaryTensor:
        pass

    def predict(model, records, device, topk):
        assert all(reference() is None for reference in references)
        calls.append([record.index for record in records])
        if len(records) > 2:
            temporary = TemporaryTensor()
            references.append(weakref.ref(temporary))
            cuda_stub["peak_allocated"] = 700 * MIB
            raise torch.cuda.OutOfMemoryError("simulated allocation limit")
        return _result(records)

    monkeypatch.setattr(adaptive, "predict_records", predict)
    predictor = _bare_predictor(max_batch_size=4, residue_limit=100)
    result = predictor.predict_window(_records([2] * 7))

    assert [len(call) for call in calls] == [4, 2, 2, 2, 1]
    assert predictor.residue_limit == 4
    assert predictor.oom_retries == 1
    assert [record.index for record, _, _ in result] == list(range(7))
    assert [indices[0] for _, indices, _ in result] == list(range(7))
    assert predictor.max_successful_batch == 2
    assert predictor.max_peak_allocated == 700 * MIB
    assert "splitting 4 sequences" in caplog.text

    predictor.predict_window(_records([2] * 3, start=7))
    assert [len(call) for call in calls[-2:]] == [2, 1]
    assert predictor.oom_retries == 1
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("lengths,batch_cap,residue_limit,expected_batches", [
    ([8, 2, 3, 2, 7, 1, 4], 2, 100, [[5, 1], [3, 2], [6, 4], [0]]),
    ([8, 2, 3, 2, 7, 1, 4], 3, 12, [[5, 1, 3], [2, 6], [4], [0]]),
    ([2] * 5, 2, 4, [[0, 1], [2, 3], [4]]),
    ([], 2, 4, []),
], ids=["batch-cap", "both-budgets", "exact-boundary-and-final-batch", "empty-window"])
def test_sorting_preserves_duplicate_ids_and_respects_both_budgets(
    monkeypatch, cuda_stub, lengths, batch_cap, residue_limit, expected_batches,
):
    calls = []

    def predict(model, records, device, topk):
        assert len(records) <= batch_cap
        assert len(records) * max(record.length for record in records) <= residue_limit
        calls.append([record.index for record in records])
        return _result(records)

    monkeypatch.setattr(adaptive, "predict_records", predict)
    predictor = _bare_predictor(max_batch_size=batch_cap, residue_limit=residue_limit)
    source = _records(lengths)
    predictions = predictor.predict_window(source)
    ordinals = list(range(len(lengths)))
    assert calls == expected_batches
    assert [record.index for record, _, _ in predictions] == ordinals
    assert [indices[0] for _, indices, _ in predictions] == ordinals
    assert [values[0] for _, _, values in predictions] == pytest.approx([i / 100 for i in ordinals])
    assert [record.index for record in source] == ordinals


def test_successful_batch_over_target_reduces_subsequent_batches(monkeypatch, cuda_stub, caplog):
    calls = []

    def predict(model, records, device, topk):
        calls.append(len(records))
        cuda_stub["peak_allocated"] = 1200 if len(calls) == 1 else 500
        return _result(records)

    monkeypatch.setattr(adaptive, "predict_records", predict)
    predictor = _bare_predictor(max_batch_size=4, residue_limit=100)
    predictor.allocator_budget = 1000
    result = predictor.predict_window(_records([2] * 7))
    assert calls == [4, 2, 1]
    assert predictor.residue_limit == 5
    assert predictor.oom_retries == 0
    assert len(result) == 7
    assert "exceeded the scheduling target" in caplog.text


@pytest.mark.parametrize("kind", ["budget", "cuda_oom"])
def test_failed_singleton_reports_name_index_length_without_truncation(monkeypatch, cuda_stub, kind):
    def oom(*args):
        raise torch.cuda.OutOfMemoryError("simulated singleton OOM")

    monkeypatch.setattr(adaptive, "predict_records", oom)
    predictor = _bare_predictor(residue_limit=10 if kind == "budget" else 100)
    record = ProteinRecord(19, "too-long", "A" * 11)
    with pytest.raises(RuntimeError, match="'too-long'.*index 19, 11 aa.*not truncated or skipped"):
        predictor.predict_window([record])
    assert predictor.cpu_fallbacks == 0
    assert predictor.oom_retries == (kind == "cuda_oom")
    assert record.sequence == "A" * 11


def test_cpu_fallback_is_lazy_reused_and_keeps_gpu_model_for_later_records(monkeypatch, cuda_stub):
    predictor = _bare_predictor(residue_limit=20, fallback="cpu")
    gpu_model = predictor.model
    cpu_model = object()
    factories = []
    calls = []

    def factory():
        factories.append(True)
        return cpu_model

    def predict(model, records, device, topk):
        calls.append((model, device.type, [record.index for record in records]))
        if device.type == "cuda" and any(record.length == 6 for record in records):
            raise torch.cuda.OutOfMemoryError("singleton six")
        return _result(records)

    predictor.cpu_model_factory = factory
    monkeypatch.setattr(adaptive, "predict_records", predict)
    assert predictor.cpu_model is None
    predictor.predict_window(_records([2]))
    assert factories == []
    # First fallback follows a real CUDA OOM; second exceeds the scheduling budget.
    first = predictor.predict_window(_records([6], start=1))
    second = predictor.predict_window(_records([30], start=2))
    predictor.predict_window(_records([2], start=3))

    assert factories == [True]
    assert predictor.model is gpu_model
    assert predictor.cpu_model is cpu_model
    assert predictor.cpu_fallbacks == 2
    assert predictor.oom_retries == 1
    assert [row[0].index for row in first + second] == [1, 2]
    assert calls[-1] == (gpu_model, "cuda", [3])
    assert [(device, indices) for model, device, indices in calls if model is cpu_model] == [("cpu", [1]), ("cpu", [2])]


def test_cpu_fallback_failure_propagates_without_counting_a_completed_record(monkeypatch, cuda_stub):
    predictor = _bare_predictor(residue_limit=1, fallback="cpu")
    predictor.cpu_model_factory = object

    def fail(*args):
        raise MemoryError("CPU fallback cannot allocate")

    monkeypatch.setattr(adaptive, "predict_records", fail)
    with pytest.raises(MemoryError, match="CPU fallback"):
        predictor.predict_window(_records([2]))
    assert predictor.cpu_fallbacks == 0


def _mock_adaptive_inference_on_cpu(monkeypatch, infer, *, failure_index=None):
    """Use the real reader/scheduler/CSV pipeline and toy model, avoiding CUDA."""
    predictors = []
    original_predict_records = adaptive.predict_records
    monkeypatch.setattr(infer, "resolve_device", lambda device: torch.device("cuda"))
    monkeypatch.setattr(infer.GRUClassifier, "to", lambda self, *args, **kwargs: self)

    def predictor_factory(model, device, topk, **kwargs):
        predictor = _bare_predictor(max_batch_size=kwargs["max_batch_size"])
        predictor.model, predictor.topk = model, topk
        predictor.oom_fallback = kwargs["oom_fallback"]
        predictor.cpu_model_factory = kwargs["cpu_model_factory"]
        predictors.append(predictor)
        return predictor

    def cpu_prediction(model, records, device, topk):
        if failure_index is not None and any(record.index >= failure_index for record in records):
            raise RuntimeError("late inference failure")
        if len(records) > 2:
            raise torch.cuda.OutOfMemoryError("synthetic batch limit")
        return original_predict_records(model, records, torch.device("cpu"), topk)

    monkeypatch.setattr(infer, "AdaptivePredictor", predictor_factory)
    monkeypatch.setattr(adaptive, "predict_records", cpu_prediction)
    return predictors


@pytest.mark.integration
def test_inference_after_sort_and_oom_writes_every_row_once_in_original_order(monkeypatch, cuda_stub, runtime_modules, tmp_path):
    infer, _ = runtime_modules
    predictors = _mock_adaptive_inference_on_cpu(monkeypatch, infer)
    source = tmp_path / "duplicates.fasta"
    names = ["duplicate", "other", "duplicate", "fourth", "fifth", "sixth", "duplicate"]
    source.write_text("".join(f">{name}\tprotein description\n{'A' * length}\n" for name, length in zip(names, [4, 1, 3, 1, 2, 6, 2])))
    output = tmp_path / "predictions.csv"
    files_before = set(tmp_path.iterdir())
    stats = infer.inference(str(source), str(output), date="202501", device="cuda", batch_size=4,
                            sort_window=5, detail=True, topk=2, num_workers=0)
    frame = pd.read_csv(output, keep_default_na=False)
    assert frame["name"].tolist() == [name for name in names for _ in range(2)]
    assert frame["predict_label"].tolist() == ["K00001", "K00002"] * 7
    assert frame["probability"].tolist() == pytest.approx([0.75, 0.25] * 7)
    assert stats["total"] == stats["annotated"] == 7
    assert stats["batching"]["oom_retries"] == predictors[0].oom_retries == 1
    assert set(tmp_path.iterdir()) == files_before | {output}


@pytest.mark.integration
@pytest.mark.parametrize("existing_output", [False, True])
def test_late_adaptive_failure_keeps_existing_output_and_removes_temporary_file(monkeypatch, cuda_stub, runtime_modules, tmp_path, existing_output):
    infer, _ = runtime_modules
    _mock_adaptive_inference_on_cpu(monkeypatch, infer, failure_index=2)
    source = tmp_path / "input.fasta"
    source.write_text(">first\nAAA\n>second\nA\n>third\nAA\n")
    output = tmp_path / "result.csv"
    previous = b"previous completed output\n"
    if existing_output:
        output.write_bytes(previous)
    files_before = set(tmp_path.iterdir())
    with pytest.raises(RuntimeError, match="late inference failure"):
        infer.inference(str(source), str(output), date="202501", device="cuda",
                        batch_size=2, sort_window=2, num_workers=0)
    assert output.exists() == existing_output
    if existing_output:
        assert output.read_bytes() == previous
    assert set(tmp_path.iterdir()) == files_before


@pytest.mark.parametrize("kwargs,match", [
    ({"memory_preset": "bogus"}, "Unknown memory_preset"),
    ({"memory_preset": "2"}, "require a CUDA device"),
    ({"oom_fallback": "skip"}, "oom_fallback"),
    ({"batch_size": 0}, "batch_size"),
    ({"batch_size": True}, "batch_size"),
    ({"sort_window": 0}, "sort_window"),
    ({"sort_window": 1.5}, "sort_window"),
    ({"max_padded_residues": -1}, "max_padded_residues"),
    ({"max_padded_residues": True}, "max_padded_residues"),
])
def test_invalid_inference_options_fail_before_model_loading(monkeypatch, kwargs, match):
    from deepkoala import infer

    def do_not_load(*args):
        pytest.fail("Invalid options must be rejected before model loading")

    monkeypatch.setattr(infer, "find_latest_date", do_not_load)
    with pytest.raises(ValueError, match=match):
        infer.inference("unused.fasta", "unused.csv", device="cpu", **kwargs)


def _run_cli_capture(monkeypatch, options, *, cuda_available=True):
    received = {}

    def inference(**kwargs):
        received.update(kwargs)
        return {"total": 0, "annotated": 0}

    monkeypatch.setattr(cli, "inference", inference)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(sys, "argv", ["deepkoala", "-i", "in.fasta", "-o", "out.csv", *options])
    cli.main()
    return received


def test_cli_forwards_memory_controls_and_explicit_batch_size(monkeypatch):
    received = _run_cli_capture(monkeypatch, ["--device", "cuda", "--memory-preset", "4", "--sort-window", "128",
                                "--max-padded-residues", "4096", "--oom-fallback", "cpu", "--batch_size", "7"])
    assert {key: received[key] for key in ("memory_preset", "sort_window", "max_padded_residues", "oom_fallback", "batch_size")} == {
        "memory_preset": "4", "sort_window": 128, "max_padded_residues": 4096,
        "oom_fallback": "cpu", "batch_size": 7,
    }


@pytest.mark.parametrize("device,available,preset,batch_size", [
    ("auto", True, "auto", None), ("cuda", True, "auto", None),
    ("auto", False, "auto", 32), ("cpu", True, "auto", 32),
    ("mps", True, "auto", 32), ("cuda", True, "off", 32),
])
def test_cli_defaults_preserve_fixed_mode_and_enable_cuda_auto(monkeypatch, device, available, preset, batch_size):
    received = _run_cli_capture(monkeypatch, ["--device", device, "--memory-preset", preset], cuda_available=available)
    assert received["batch_size"] == batch_size
    assert received["memory_preset"] == preset
    assert received["sort_window"] == 4096
    assert received["max_padded_residues"] is None
    assert received["oom_fallback"] == "error"


def test_cli_rejects_unknown_preset_during_parsing(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["deepkoala", "-i", "in", "-o", "out", "--memory-preset", "12"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_cli_explicit_gpu_preset_on_cpu_reports_actionable_failure(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["deepkoala", "-i", "in", "-o", "out", "--device", "cpu", "--memory-preset", "2"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert "require a CUDA device" in capsys.readouterr().out


@pytest.mark.parametrize("options", [["--memory-preset", "4"], ["--max-padded-residues", "512"], ["--oom-fallback", "cpu"]])
def test_cli_rejects_adaptive_controls_in_multi_mode(monkeypatch, capsys, options):
    monkeypatch.setattr(sys, "argv", ["deepkoala", "-i", "in", "-o", "out", "--multi", "--profiles_dir", "profiles", *options])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert "standard inference" in capsys.readouterr().out
