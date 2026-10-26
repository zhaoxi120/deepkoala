"""Streaming must preserve batching, predictions, and completed output files."""

import builtins
import io
import os
from pathlib import Path
import tarfile
import zipfile

import pandas as pd
import pytest
import torch

from deepkoala.data import ProteinDataset, collate_fn, get_dataloader


@pytest.mark.skipif(os.name != "posix", reason="POSIX filename-component limit; Windows whole-path limits differ")
@pytest.mark.parametrize("suffix", [".csv", ".csv.gz"])
def test_atomic_output_accepts_long_valid_filenames(tmp_path, suffix):
    from deepkoala.infer import _atomic_csv_output

    output = tmp_path / ("a" * 230 + suffix)
    output.write_text("previous result\n", encoding="utf-8")
    with _atomic_csv_output(output) as handle:
        handle.write("name,predict_label\nquery,K00001\n")

    assert pd.read_csv(output).to_dict("records") == [{"name": "query", "predict_label": "K00001"}]
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize(
    "newline,final_newline", [("\n", True), ("\n", False), ("\r\n", False)]
)
def test_streamed_batches_match_eager_records_and_keep_duplicate_ids(
    tmp_path, workers, newline, final_newline
):
    fasta = tmp_path / "proteins with spaces 蛋白.fasta"
    # Lowercase and spaces remain unknown tokens, as in the original reader.
    lines = [
        ">duplicate first description", "ACD", "x?",
        ">second", "W",
        ">duplicate second description", "MNP", "QRS",
        ">tab\tdescription", "A C",
        ">last", "Y",
    ]
    content = newline.join(lines) + (newline if final_newline else "")
    fasta.write_bytes(content.encode("utf-8"))
    eager = ProteinDataset(str(fasta))
    expected = [
        collate_fn([eager[index] for index in range(start, min(start + 2, len(eager)))])
        for start in range(0, len(eager), 2)
    ]

    actual = list(get_dataloader(str(fasta), batch_size=2, num_workers=workers))

    assert len(actual) == 3
    assert [list(names) for names, _, _ in actual] == [
        ["duplicate", "second"], ["duplicate", "tab"], ["last"]
    ]
    for (names, tokens, indices), (old_names, old_tokens, old_indices) in zip(actual, expected):
        assert names == old_names
        torch.testing.assert_close(tokens, old_tokens, rtol=0, atol=0)
        torch.testing.assert_close(indices, old_indices, rtol=0, atol=0)


def test_first_batch_reads_only_a_bounded_prefix_without_whole_file_reads(tmp_path, monkeypatch):
    fasta = tmp_path / "many.fasta"
    fasta.write_text("".join(f">protein_{index}\nAC\nDE\n" for index in range(1000)))
    consumed = []
    handles = []

    class CountedInput:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def __iter__(self):
            return self

        def __next__(self):
            line = next(self.handle)
            consumed.append(line)
            return line

        def readline(self, *args):
            line = self.handle.readline(*args)
            if line:
                consumed.append(line)
            return line

        def read(self, *args):
            pytest.fail("The streaming input path must not call read()")

        def readlines(self, *args):
            pytest.fail("The streaming input path must not call readlines()")

        def __getattr__(self, name):
            return getattr(self.handle, name)

    def wrap_opener(opener):
        def open_counted(file, *args, **kwargs):
            handle = opener(file, *args, **kwargs)
            if isinstance(file, (str, bytes, Path)) and Path(file) == fasta:
                handles.append(handle)
                return CountedInput(handle)
            return handle
        return open_counted

    monkeypatch.setattr(builtins, "open", wrap_opener(builtins.open))
    monkeypatch.setattr(io, "open", wrap_opener(io.open))

    loader = get_dataloader(str(fasta), batch_size=2, num_workers=0)
    assert consumed == []
    iterator = iter(loader)
    try:
        names, tokens, indices = next(iterator)
        assert names == ("protein_0", "protein_1")
        assert tokens.tolist() == [[2, 3, 4, 5], [2, 3, 4, 5]]
        assert indices.tolist() == [3, 3]
        assert 6 <= len(consumed) <= 12
    finally:
        # Close this intentionally abandoned input, regardless of iterator internals.
        for handle in handles:
            handle.close()


@pytest.mark.parametrize(
    "content",
    ["ACD\n>valid\nA\n", ">empty\n>valid\nA\n", ">empty\n", ">\nA\n"],
    ids=["sequence-before-header", "empty-internal-record", "empty-final-record", "empty-header"],
)
def test_streaming_rejects_invalid_records_with_a_useful_error(tmp_path, content):
    fasta = tmp_path / "malformed.fasta"
    fasta.write_text(content)
    with pytest.raises(ValueError, match=r"(?i)fasta|header|sequence|empty"):
        list(get_dataloader(str(fasta), batch_size=2, num_workers=0))


@pytest.mark.parametrize("workers", [0, 2])
def test_streaming_empty_input_yields_no_batches(tmp_path, workers):
    fasta = tmp_path / "empty.fasta"
    fasta.write_text("")
    assert list(get_dataloader(str(fasta), batch_size=2, num_workers=workers)) == []


def test_streaming_missing_input_reports_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(get_dataloader(str(tmp_path / "missing.fasta"), batch_size=2, num_workers=0))


def test_spawn_workers_preserve_record_order_and_final_partial_batch(tmp_path):
    fasta = tmp_path / "spawn input 蛋白.fasta"
    fasta.write_text(">duplicate\nACD\n>second\nW\n>duplicate\nMNPQ\n>fourth\nX\n>last\nY\n")
    eager = ProteinDataset(str(fasta))
    loader = get_dataloader(str(fasta), batch_size=2, num_workers=2)
    # Spawn starts a fresh interpreter, checking serialization without fork inheritance.
    loader.multiprocessing_context = "spawn"

    batches = list(loader)

    assert [list(names) for names, _, _ in batches] == [
        ["duplicate", "second"], ["duplicate", "fourth"], ["last"]
    ]
    for start, (names, tokens, indices) in zip(range(0, len(eager), 2), batches):
        old_names, old_tokens, old_indices = collate_fn(
            [eager[index] for index in range(start, min(start + 2, len(eager)))]
        )
        assert names == old_names
        torch.testing.assert_close(tokens, old_tokens, rtol=0, atol=0)
        torch.testing.assert_close(indices, old_indices, rtol=0, atol=0)


@pytest.mark.integration
@pytest.mark.parametrize("detail", [False, True])
@pytest.mark.parametrize("topk", [1, 2, 10])
def test_streaming_csv_preserves_schema_rows_and_per_sequence_counts(
    runtime_modules, tmp_path, detail, topk
):
    infer, _ = runtime_modules
    fasta = tmp_path / "duplicates.fasta"
    fasta.write_text('>duplicate\nACD\n>duplicate\nW\n>last,"quoted"\nAC\n')
    output = tmp_path / "results" / "output.csv"

    stats = infer.inference(
        str(fasta), str(output), batch_size=2, num_workers=0,
        device="cpu", detail=detail, topk=topk,
    )

    table = pd.read_csv(output, keep_default_na=False)
    k = min(topk, 2)
    assert stats == {"total": 3, "annotated": 3, "output": str(output)}
    assert table["name"].tolist() == [
        name for name in ["duplicate", "duplicate", 'last,"quoted"'] for _ in range(k)
    ]
    if detail:
        assert list(table.columns) == ["name", "predict_label", "probability", "threshold", "annotate"]
        assert table["predict_label"].tolist() == ["K00001", "K00002"][:k] * 3
        assert table["probability"].tolist() == pytest.approx([0.75, 0.25][:k] * 3)
        assert table["threshold"].tolist() == pytest.approx([0.7, 0.3][:k] * 3)
        assert table["annotate"].tolist() == ["*", ""][:k] * 3
    else:
        assert list(table.columns) == ["name", "predict_label"]
        assert table["predict_label"].tolist() == ["K00001", ""][:k] * 3


@pytest.mark.integration
@pytest.mark.parametrize("detail", [False, True])
def test_streaming_empty_inference_still_writes_one_header(runtime_modules, tmp_path, detail):
    infer, _ = runtime_modules
    fasta = tmp_path / "empty.fasta"
    fasta.write_text("")
    output = tmp_path / "empty.csv"

    stats = infer.inference(str(fasta), str(output), num_workers=0, device="cpu", detail=detail)

    header = "name,predict_label,probability,threshold,annotate" if detail else "name,predict_label"
    assert output.read_text() == header + "\n"
    assert stats == {"total": 0, "annotated": 0, "output": str(output)}


@pytest.mark.integration
@pytest.mark.parametrize("existing_output", [False, True])
def test_inference_failure_leaves_no_partial_output(
    runtime_modules, fasta_file, tmp_path, monkeypatch, existing_output
):
    infer, _ = runtime_modules
    destination = tmp_path / "destination"
    destination.mkdir()
    output = destination / "predictions.csv"
    previous = b"previous completed output\n"
    if existing_output:
        output.write_bytes(previous)
    files_before = set(destination.iterdir())
    original_forward = infer.GRUClassifier.forward
    calls = 0

    def fail_after_first_batch(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated inference failure after a complete batch")
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(infer.GRUClassifier, "forward", fail_after_first_batch)
    with pytest.raises(RuntimeError, match="simulated inference failure"):
        infer.inference(
            str(fasta_file), str(output), batch_size=2, num_workers=0, device="cpu", detail=True,
        )

    assert calls == 2
    assert set(destination.iterdir()) == files_before
    if existing_output:
        assert output.read_bytes() == previous
    else:
        assert not output.exists()


@pytest.mark.integration
def test_late_invalid_record_preserves_completed_output(runtime_modules, tmp_path):
    infer, _ = runtime_modules
    fasta = tmp_path / "late-invalid.fasta"
    fasta.write_text(">first\nACD\n>second\nW\n>empty\n")
    destination = tmp_path / "destination"
    destination.mkdir()
    output = destination / "predictions.csv"
    output.write_text("completed output\n")

    with pytest.raises(ValueError, match=r"(?i)fasta|header|sequence|empty"):
        infer.inference(str(fasta), str(output), batch_size=2, num_workers=0, device="cpu")

    assert output.read_text() == "completed output\n"
    assert list(destination.iterdir()) == [output]


@pytest.mark.integration
def test_inference_can_safely_replace_its_input_path(runtime_modules, fasta_file, tmp_path):
    infer, _ = runtime_modules
    reference = tmp_path / "reference.csv"
    kwargs = dict(batch_size=2, num_workers=0, device="cpu", detail=True, topk=2)
    expected_stats = infer.inference(str(fasta_file), str(reference), **kwargs)

    actual_stats = infer.inference(str(fasta_file), str(fasta_file), **kwargs)

    assert fasta_file.read_bytes() == reference.read_bytes()
    assert actual_stats == {**expected_stats, "output": str(fasta_file)}


@pytest.mark.integration
@pytest.mark.parametrize(
    "suffix", [".gz", ".bz2", ".xz", ".zip", ".tar", ".tar.gz", ".tar.bz2", ".tar.xz"]
)
def test_compressed_streaming_output_matches_plain_csv_and_has_one_archive_member(
    runtime_modules, fasta_file, tmp_path, suffix
):
    infer, _ = runtime_modules
    plain = tmp_path / "reference.csv"
    compressed = tmp_path / ("results.csv" + suffix)
    # Three records in two batches expose repeated headers or archive members.
    kwargs = dict(batch_size=2, num_workers=0, device="cpu", detail=True, topk=2)
    expected_stats = infer.inference(str(fasta_file), str(plain), **kwargs)

    actual_stats = infer.inference(str(fasta_file), str(compressed), **kwargs)

    pd.testing.assert_frame_equal(
        pd.read_csv(compressed, keep_default_na=False),
        pd.read_csv(plain, keep_default_na=False),
        check_exact=True,
    )
    assert actual_stats == {**expected_stats, "output": str(compressed)}
    if suffix == ".zip":
        with zipfile.ZipFile(compressed) as archive:
            assert archive.namelist() == ["results.csv"]
            assert archive.read("results.csv") == plain.read_bytes()
    elif suffix.startswith(".tar"):
        with tarfile.open(compressed, "r:*") as archive:
            members = archive.getmembers()
            assert [member.name for member in members] == ["results.csv"]
            assert members[0].isfile()
            with archive.extractfile(members[0]) as payload:
                assert payload.read() == plain.read_bytes()
