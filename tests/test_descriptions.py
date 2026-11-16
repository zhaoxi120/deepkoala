"""Description lookups and user-visible CSV behavior use synthetic resources only."""

import csv
import json
import shutil
import sys
import warnings

import pandas as pd
import pytest

from deepkoala import cli
from deepkoala.descriptions import load_label_descriptions


def _write_descriptions(path, entries, encoding="utf-8"):
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ko", "description"])
        writer.writerows(entries)


def _run(runtime_modules, multi, fasta, output, tmp_path, **kwargs):
    infer, infer_multi = runtime_modules
    options = dict(device="cpu", **kwargs)
    if multi:
        profiles = tmp_path / "profiles"
        profiles.mkdir(exist_ok=True)
        return infer_multi.inference_precision(
            str(fasta), str(output), profiles_dir=str(profiles), **options
        )
    return infer.inference(str(fasta), str(output), num_workers=0, batch_size=2, **options)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_description_loader_preserves_quoted_text_and_compound_order(tmp_path, encoding):
    path = tmp_path / "ko_description.csv"
    first = 'enzyme, "quoted"; 蛋白\nsecond line'
    second = "Other activity [EC:1.2.3.4]"
    _write_descriptions(path, [("K00001", first), ("K00002", second)], encoding)

    with warnings.catch_warnings(record=True) as caught:
        result = load_label_descriptions(path, ["K00001", "K00002+K00001"])

    assert not caught
    assert result == {
        "K00001": first,
        "K00002+K00001": f"K00002: {second} | K00001: {first}",
    }


@pytest.mark.parametrize(
    "content",
    [
        "",
        "KO,definition\nK00001,first\n",
        "ko,description,extra\nK00001,first,value\n",
        "ko,ko\nK00001,first\n",
        "ko,description\nK00001\n",
        "ko,description\nK00001,first,extra\n",
        "ko,description\nBAD0001,first\n",
        "ko,description\nK00001+K00002,first\n",
        "ko,description\nK00001,first\nK00001,second\n",
        "ko,description\nK00001,\n",
        "ko,description\nK00001,   \n",
        'ko,description\nK00001,"unterminated\n',
        'ko,description\nK00001,"quoted"trailing\n',
    ],
    ids=[
        "empty-file", "wrong-header", "extra-header", "duplicate-header",
        "missing-field", "extra-field", "invalid-ko", "compound-in-source",
        "duplicate-ko", "empty-description", "whitespace-description",
        "unclosed-quote", "invalid-quote",
    ],
)
def test_description_loader_rejects_invalid_tables(tmp_path, content):
    path = tmp_path / "ko_description.csv"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        load_label_descriptions(path, ["K00001"])


def test_description_loader_reports_unique_missing_components_once(tmp_path):
    path = tmp_path / "ko_description.csv"
    _write_descriptions(path, [("K00001", "first")])

    with pytest.warns(UserWarning) as caught:
        result = load_label_descriptions(
            path, ["K00001", "K00002", "K00001+K00002", "K00002+K00003"]
        )

    assert len(caught) == 1
    message = str(caught[0].message)
    assert "2" in message and "K00002" in message and "K00003" in message
    assert result == {
        "K00001": "first",
        "K00002": "",
        "K00001+K00002": "K00001: first | K00002: [description unavailable]",
        "K00002+K00003": (
            "K00002: [description unavailable] | K00003: [description unavailable]"
        ),
    }


def test_description_loader_bounds_warning_for_many_missing_kos(tmp_path):
    path = tmp_path / "ko_description.csv"
    _write_descriptions(path, [("K00001", "first")])
    missing = [f"K{index:05d}" for index in range(1000, 1100)]

    with pytest.warns(UserWarning) as caught:
        result = load_label_descriptions(path, missing)

    assert len(caught) == 1
    message = str(caught[0].message)
    assert "100" in message
    assert sum(ko in message for ko in missing) < len(missing)
    assert result == dict.fromkeys(missing, "")


@pytest.mark.integration
@pytest.mark.parametrize("multi", [False, True], ids=["standard", "multi"])
@pytest.mark.parametrize("variant", ["full", "frag"])
@pytest.mark.parametrize("detail", [False, True], ids=["simple", "detail"])
def test_descriptions_append_to_existing_results_without_changing_predictions(
    runtime_modules, fasta_file, tmp_path, toy_descriptions, multi, variant, detail
):
    plain = tmp_path / "without-description.csv"
    output = tmp_path / "with-description.csv"
    kwargs = dict(model=variant, detail=detail, topk=10)
    old_stats = _run(runtime_modules, multi, fasta_file, plain, tmp_path, **kwargs)

    stats = _run(
        runtime_modules, multi, fasta_file, output, tmp_path, description=True, **kwargs
    )

    previous = pd.read_csv(plain, keep_default_na=False)
    table = pd.read_csv(output, keep_default_na=False)
    assert table.columns.tolist() == previous.columns.tolist() + ["description"]
    pd.testing.assert_frame_equal(table.drop(columns="description"), previous)
    assert stats == {**old_stats, "output": str(output)}
    candidates = ["K00001", "K00002" if detail else ""]
    expected_labels = candidates * (2 if multi else 3) + ([""] if multi else [])
    assert table["predict_label"].tolist() == expected_labels
    assert table["description"].tolist() == [toy_descriptions.get(ko, "") for ko in expected_labels]


@pytest.mark.parametrize("multi", [False, True], ids=["standard", "multi"])
def test_top_one_description_has_one_prediction_per_sequence(
    runtime_modules, fasta_file, tmp_path, toy_descriptions, multi
):
    output = tmp_path / "top-one.csv"

    _run(runtime_modules, multi, fasta_file, output, tmp_path, description=True, topk=1)

    table = pd.read_csv(output, keep_default_na=False)
    assert table["name"].tolist() == ["alpha", "beta", "gamma"]
    assert table["description"].tolist() == [toy_descriptions["K00001"]] * (2 if multi else 3) + ([""] if multi else [])


@pytest.mark.parametrize("multi", [False, True], ids=["standard", "multi"])
@pytest.mark.parametrize("detail", [False, True], ids=["simple", "detail"])
def test_empty_fasta_has_description_header_and_zero_counts(
    runtime_modules, tmp_path, multi, detail
):
    fasta = tmp_path / "empty.fasta"
    fasta.write_text("", encoding="utf-8")
    output = tmp_path / "empty.csv"

    stats = _run(runtime_modules, multi, fasta, output, tmp_path, description=True, detail=detail)

    columns = ["name", "predict_label"]
    if detail:
        columns += ["probability", "threshold"] + (["start", "end"] if multi else []) + ["annotate"]
    columns += ["description"]
    assert output.read_text(encoding="utf-8") == ",".join(columns) + "\n"
    assert stats == {"total": 0, "annotated": 0, "output": str(output)}


@pytest.mark.parametrize("multi", [False, True], ids=["standard", "multi"])
@pytest.mark.parametrize("state", ["missing", "invalid"])
def test_bad_description_resource_fails_before_model_or_fasta_and_preserves_output(
    runtime_modules, isolated_runtime, tmp_path, monkeypatch, multi, state
):
    path = isolated_runtime / "resources" / "202501" / "ko_description.csv"
    if state == "missing":
        path.unlink()
    else:
        path.write_text("ko,description\nK00001,first\nK00001,duplicate\n", encoding="utf-8")
    output = tmp_path / "previous.csv"
    output.write_bytes(b"previous complete result\n")

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid descriptions must fail before loading weights or processing FASTA")

    infer, infer_multi = runtime_modules
    monkeypatch.setattr(infer.torch, "load", forbidden)
    monkeypatch.setattr(infer, "get_dataloader", forbidden)
    monkeypatch.setattr(infer_multi, "_read_fasta", forbidden)
    expected_error = FileNotFoundError if state == "missing" else ValueError

    with pytest.raises(expected_error):
        _run(runtime_modules, multi, tmp_path / "missing.fasta", output, tmp_path, description=True)

    assert output.read_bytes() == b"previous complete result\n"


@pytest.mark.parametrize("multi", [False, True], ids=["standard", "multi"])
@pytest.mark.parametrize("state", ["missing", "invalid"])
def test_description_disabled_ignores_unavailable_resource(
    runtime_modules, isolated_runtime, fasta_file, tmp_path, multi, state
):
    path = isolated_runtime / "resources" / "202501" / "ko_description.csv"
    if state == "missing":
        path.unlink()
    else:
        path.write_text("not a description table", encoding="utf-8")
    output = tmp_path / "default.csv"

    with warnings.catch_warnings(record=True) as caught:
        _run(runtime_modules, multi, fasta_file, output, tmp_path)

    assert not [warning for warning in caught if "description" in str(warning.message).lower()]
    assert pd.read_csv(output).columns.tolist() == ["name", "predict_label"]


@pytest.mark.parametrize("multi", [False, True], ids=["standard", "multi"])
def test_latest_requires_its_own_description_file_without_older_fallback(
    runtime_modules, isolated_runtime, fasta_file, tmp_path, multi
):
    resources = isolated_runtime / "resources"
    latest = resources / "202502"
    shutil.copytree(resources / "202501", latest)
    (latest / "ko_description.csv").unlink()
    output = tmp_path / "latest.csv"

    with pytest.raises(FileNotFoundError, match="202502"):
        _run(runtime_modules, multi, fasta_file, output, tmp_path, description=True, date="latest")
    assert not output.exists()

    _run(runtime_modules, multi, fasta_file, output, tmp_path, description=True, date="202501")
    assert "description" in pd.read_csv(output).columns


@pytest.mark.parametrize("multi", [False, True], ids=["standard", "multi"])
def test_compound_labels_and_missing_descriptions_reach_output(
    runtime_modules, isolated_runtime, fasta_file, tmp_path, toy_descriptions, multi
):
    config = isolated_runtime / "resources" / "202501" / "ko_config_full.json"
    config.write_text(json.dumps({
        "K00001+K00002": {"index": 0, "threshold": 0.7},
        "K99999": {"index": 1, "threshold": 0.3},
    }), encoding="utf-8")
    output = tmp_path / "compound.csv"

    with pytest.warns(UserWarning, match="K99999") as caught:
        _run(runtime_modules, multi, fasta_file, output, tmp_path, description=True, detail=True, topk=2)

    assert len([warning for warning in caught if issubclass(warning.category, UserWarning)]) == 1
    table = pd.read_csv(output, keep_default_na=False)
    expected = f"K00001: {toy_descriptions['K00001']} | K00002: {toy_descriptions['K00002']}"
    assert table["description"].tolist() == [expected, ""] * (2 if multi else 3) + ([""] if multi else [])
    assert table.loc[table["predict_label"] == "K99999", "description"].eq("").all()


@pytest.mark.parametrize("suffix", [".gz", ".zip", ".tar.gz"])
def test_multibatch_compressed_descriptions_round_trip(
    runtime_modules, fasta_file, tmp_path, toy_descriptions, suffix
):
    output = tmp_path / ("description.csv" + suffix)

    _run(runtime_modules, False, fasta_file, output, tmp_path, description=True, detail=True, topk=2)

    table = pd.read_csv(output, keep_default_na=False)
    assert table["name"].tolist() == ["alpha", "alpha", "beta", "beta", "gamma", "gamma"]
    assert table["description"].tolist() == list(toy_descriptions.values()) * 3


def test_recursive_multi_domain_descriptions_follow_each_domain(
    runtime_modules, tmp_path, monkeypatch, toy_descriptions
):
    _, infer_multi = runtime_modules
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "K00001.hmm").touch()
    monkeypatch.setattr(
        infer_multi, "_run_hmmsearch",
        lambda profile, sequence: (61, 110) if len(sequence) == 170 else (None, None),
    )
    fasta = tmp_path / "domains.fasta"
    fasta.write_text(">three-domains\n" + "A" * 170 + "\n", encoding="utf-8")
    output = tmp_path / "domains.csv"

    stats = _run(runtime_modules, True, fasta, output, tmp_path, description=True, detail=True, topk=2)

    table = pd.read_csv(output, keep_default_na=False)
    assert stats["total"] == stats["annotated"] == 1
    assert table["predict_label"].tolist() == ["K00001", "K00002"] * 3
    assert table["description"].tolist() == list(toy_descriptions.values()) * 3
    assert table["start"].iloc[::2].astype(int).tolist() == [61, 1, 111]
    assert table["end"].iloc[::2].astype(int).tolist() == [110, 60, 170]


@pytest.mark.parametrize("multi", [False, True], ids=["standard", "multi"])
@pytest.mark.parametrize("enabled", [False, True])
def test_cli_passes_description_flag_to_selected_inference(monkeypatch, tmp_path, multi, enabled):
    received = {}

    def fake_inference(**kwargs):
        received.update(kwargs)
        return {"total": 0, "annotated": 0}

    monkeypatch.setattr(cli, "inference_precision" if multi else "inference", fake_inference)
    argv = ["deepkoala", "-i", "in.fasta", "-o", str(tmp_path / "out.csv")]
    if multi:
        argv += ["--multi", "--profiles_dir", "profiles"]
    if enabled:
        argv += ["--description"]
    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    assert received.get("description", False) is enabled


@pytest.mark.integration
@pytest.mark.parametrize("multi", [False, True], ids=["standard", "multi"])
def test_cli_description_end_to_end(run_cli, fasta_file, tmp_path, toy_descriptions, multi):
    output = tmp_path / "cli-description.csv"
    args = [
        "-i", fasta_file, "-o", output, "--device", "cpu", "--num_workers", "0",
        "--description", "--detail", "--topk", "2", "--date", "202501",
    ]
    if multi:
        profiles = tmp_path / "profiles"
        profiles.mkdir()
        args += ["--multi", "--profiles_dir", profiles]

    result = run_cli(*args)

    assert result.returncode == 0, result.stdout + result.stderr
    table = pd.read_csv(output, keep_default_na=False)
    assert table.columns[-1] == "description"
    assert table["description"].tolist() == list(toy_descriptions.values()) * (2 if multi else 3) + ([""] if multi else [])
    assert f"Processed 3 sequences, annotated {2 if multi else 3}." in result.stdout
