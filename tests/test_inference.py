import pandas as pd
import pytest


@pytest.mark.integration
@pytest.mark.parametrize("variant", ["full", "frag"])
def test_inference_simple_output_and_topk_limit(runtime_modules, fasta_file, tmp_path, variant):
    infer, _ = runtime_modules
    output = tmp_path / "nested" / "simple.csv"
    stats = infer.inference(str(fasta_file), str(output), model=variant, date="latest", device="cpu", num_workers=0, topk=10)
    frame = pd.read_csv(output, keep_default_na=False)
    assert stats == {"total": 3, "annotated": 3, "output": str(output)}
    assert list(frame.columns) == ["name", "predict_label"]
    assert frame["predict_label"].tolist() == ["K00001", ""] * 3
    assert frame["name"].tolist() == ["alpha", "alpha", "beta", "beta", "gamma", "gamma"]


@pytest.mark.integration
def test_empty_input_writes_header_and_zero_counts(runtime_modules, tmp_path):
    infer, _ = runtime_modules
    fasta = tmp_path / "empty.fasta"
    fasta.write_text("", encoding="utf-8")
    output = tmp_path / "empty.csv"
    stats = infer.inference(str(fasta), str(output), device="cpu", num_workers=0)
    assert stats["total"] == stats["annotated"] == 0
    assert pd.read_csv(output).empty
    assert list(pd.read_csv(output).columns) == ["name", "predict_label"]


@pytest.mark.parametrize("topk", [0, -1])
def test_invalid_topk_is_rejected(runtime_modules, fasta_file, tmp_path, topk):
    infer, _ = runtime_modules
    with pytest.raises(ValueError, match="topk"):
        infer.inference(str(fasta_file), str(tmp_path / "out.csv"), device="cpu", num_workers=0, topk=topk)


def test_incomplete_model_resources_fail_without_output(runtime_modules, isolated_runtime, fasta_file, tmp_path):
    infer, _ = runtime_modules
    (isolated_runtime / "resources" / "202501" / "weights_full.pt").unlink()
    output = tmp_path / "out.csv"
    with pytest.raises(FileNotFoundError):
        infer.inference(str(fasta_file), str(output), device="cpu", num_workers=0)
    assert not output.exists()
