"""Multi-domain regression tests with synthetic sequences and model weights."""

import pandas as pd
import pytest
import torch

from deepkoala import infer_multi


def test_classify_uses_real_model_probabilities_and_caps_topk(runtime_modules):
    model, _, _, _ = infer_multi._load_model("full", "latest", torch.device("cpu"))

    with torch.no_grad():
        indices, probabilities = infer_multi._classify(
            model, "ACDEFGHIKLMNPQRSTVWY" * 3, torch.device("cpu"), topk=10
        )

    assert indices == [0, 1]
    assert probabilities == pytest.approx([0.75, 0.25])
    assert sum(probabilities) == pytest.approx(1)


@pytest.mark.parametrize("length, expected_hits", [(0, 0), (1, 0), (49, 0), (50, 1)])
def test_minimum_fragment_length(monkeypatch, tmp_path, length, expected_hits):
    classified = []

    def classify(model, sequence, device, topk):
        classified.append(sequence)
        return [0], [0.9]

    monkeypatch.setattr(infer_multi, "_classify", classify)
    hits = infer_multi._annotate_sequence(
        "A" * length, object(), {0: "K00001"}, {"K00001": 0.8}, tmp_path, torch.device("cpu")
    )

    assert len(hits) == expected_hits
    assert len(classified) == expected_hits
    if expected_hits:
        assert (hits[0].start, hits[0].end) == (1, 50)


@pytest.mark.parametrize("probability, annotated", [(0.499, False), (0.5, True), (0.501, True)])
def test_threshold_is_inclusive_and_rejected_hits_have_no_bounds(
    monkeypatch, tmp_path, probability, annotated
):
    monkeypatch.setattr(infer_multi, "_classify", lambda *args, **kwargs: ([0], [probability]))
    calls = []
    (tmp_path / "K00001.hmm").touch()

    def hmmsearch(*args):
        calls.append(args)
        return 5, 55

    monkeypatch.setattr(infer_multi, "_run_hmmsearch", hmmsearch)
    hits = infer_multi._annotate_sequence(
        "A" * 60, object(), {0: "K00001"}, {"K00001": 0.5}, tmp_path,
        torch.device("cpu"), name="query",
    )

    assert len(hits) == 1
    assert hits[0].name == "query"
    assert hits[0].annotate == ("*" if annotated else "")
    assert (hits[0].start, hits[0].end) == ((5, 55) if annotated else (None, None))
    assert len(calls) == int(annotated)


@pytest.mark.parametrize("hmm_exists", [False, True])
def test_missing_profile_or_no_hmm_hit_falls_back_to_whole_fragment(
    monkeypatch, tmp_path, hmm_exists
):
    monkeypatch.setattr(infer_multi, "_classify", lambda *args, **kwargs: ([0], [0.9]))
    called = []
    if hmm_exists:
        (tmp_path / "K00001.hmm").touch()

    def no_hit(*args):
        called.append(args)
        return None, None

    monkeypatch.setattr(infer_multi, "_run_hmmsearch", no_hit)
    hits = infer_multi._annotate_sequence(
        "A" * 60, object(), {0: "K00001"}, {"K00001": 0.1}, tmp_path, torch.device("cpu")
    )

    assert [(hit.start, hit.end) for hit in hits] == [(1, 60)]
    assert len(called) == int(hmm_exists)


@pytest.mark.parametrize("left_length", [49, 50, 60])
def test_recursive_domains_preserve_one_based_offsets_and_topk(
    monkeypatch, tmp_path, left_length
):
    left, middle, right = "A" * left_length, "C" * 50, "D" * 60
    sequence = left + middle + right
    (tmp_path / "K00001.hmm").touch()
    fragments = []
    requested_topk = []

    def classify(model, fragment, device, topk):
        fragments.append(fragment)
        requested_topk.append(topk)
        return [0, 1], [0.9, 0.1]

    def hmmsearch(profile, fragment):
        if fragment == sequence:
            return left_length + 1, left_length + 50
        return None, None

    monkeypatch.setattr(infer_multi, "_classify", classify)
    monkeypatch.setattr(infer_multi, "_run_hmmsearch", hmmsearch)
    hits = infer_multi._annotate_sequence(
        sequence, object(), {0: "K00001", 1: "K00002"},
        {"K00001": 0.8, "K00002": 0.2}, tmp_path,
        torch.device("cpu"), name="multi", topk=2,
    )

    expected_fragments = [sequence] + ([left] if left_length >= 50 else []) + [right]
    expected_bounds = [(left_length + 1, left_length + 50)]
    if left_length >= 50:
        expected_bounds.append((1, left_length))
    expected_bounds.append((left_length + 51, len(sequence)))
    assert fragments == expected_fragments
    assert requested_topk == [2] * len(expected_fragments)
    assert [(hit.start, hit.end) for hit in hits[::2]] == expected_bounds
    assert all(hit.name == "multi" for hit in hits)
    assert all(hit.predict_label == "K00001" and hit.annotate == "*" for hit in hits[::2])
    assert all(hit.predict_label == "K00002" and hit.annotate == "" for hit in hits[1::2])
    assert all(hit.start is None and hit.end is None for hit in hits[1::2])


def test_domain_row_uses_missing_values_for_unlocalized_prediction():
    hit = infer_multi.DomainHit("query", "K00001", 0.25, 0.75, None, None)

    row = hit.as_row()

    assert set(row) == {"name", "predict_label", "probability", "threshold", "annotate", "start", "end"}
    assert row["name"] == "query"
    assert row["annotate"] == ""
    assert pd.isna(row["start"]) and pd.isna(row["end"])


@pytest.mark.parametrize("detail", [False, True])
def test_precision_csv_and_counts_use_real_model(runtime_modules, fasta_file, tmp_path, detail):
    profiles = tmp_path / "empty profiles"
    profiles.mkdir()
    output = tmp_path / "nested output" / "annotations.csv"

    result = infer_multi.inference_precision(
        str(fasta_file), str(output), profiles_dir=str(profiles), device="cpu", detail=detail, topk=10
    )

    assert result == {"total": 3, "annotated": 2, "output": str(output)}
    table = pd.read_csv(output)
    assert list(table["name"]) == ["alpha", "alpha", "beta", "beta", "gamma"]
    if detail:
        assert list(table.columns) == ["name", "predict_label", "probability", "threshold", "start", "end", "annotate"]
        assert list(table["predict_label"].iloc[:4]) == ["K00001", "K00002"] * 2
        assert table["probability"].iloc[:4].tolist() == pytest.approx([0.75, 0.25] * 2)
        assert table["threshold"].iloc[:4].tolist() == pytest.approx([0.7, 0.3] * 2)
        assert table["start"].iloc[[0, 2]].tolist() == [1, 1]
        assert table["end"].iloc[[0, 2]].tolist() == [63, 60]
        assert table["start"].iloc[[1, 3, 4]].isna().all()
        assert table["end"].iloc[[1, 3, 4]].isna().all()
        assert table["annotate"].iloc[[0, 2]].tolist() == ["*", "*"]
        assert table["annotate"].iloc[[1, 3, 4]].isna().all()
    else:
        assert list(table.columns) == ["name", "predict_label"]
        assert table["predict_label"].iloc[[0, 2]].tolist() == ["K00001", "K00001"]
        assert table["predict_label"].iloc[[1, 3, 4]].isna().all()


def test_precision_empty_fasta_writes_header_and_zero_counts(runtime_modules, tmp_path):
    fasta = tmp_path / "empty.fasta"
    fasta.write_text("", encoding="utf-8")
    output = tmp_path / "empty.csv"

    result = infer_multi.inference_precision(
        str(fasta), str(output), profiles_dir=str(tmp_path), device="cpu", detail=True
    )

    assert result["total"] == result["annotated"] == 0
    table = pd.read_csv(output)
    assert table.empty
    assert list(table.columns) == ["name", "predict_label", "probability", "threshold", "start", "end", "annotate"]


def test_annotation_api_with_real_model(runtime_modules, tmp_path):
    hits = infer_multi.annotate_precision("A" * 60, hmmer_dir=str(tmp_path), device="cpu")

    assert len(hits) == 1
    assert hits[0] == {"ko": "K00001", "prob": pytest.approx(0.75), "threshold": 0.7, "start": 1, "end": 60}


def test_precision_requires_profiles_before_loading_models(tmp_path):
    with pytest.raises(ValueError, match="profiles_dir must be provided"):
        infer_multi.inference_precision("missing.fasta", str(tmp_path / "out.csv"), device="cpu")
    with pytest.raises(ValueError, match="hmmer_dir must be provided"):
        infer_multi.annotate_precision("A" * 60, device="cpu")


@pytest.mark.parametrize("topk", [0, -1])
def test_precision_rejects_nonpositive_topk_before_loading_models(tmp_path, topk):
    with pytest.raises(ValueError, match="topk must be >= 1"):
        infer_multi.inference_precision(
            "missing.fasta", str(tmp_path / "out.csv"), profiles_dir=str(tmp_path), device="cpu", topk=topk
        )
