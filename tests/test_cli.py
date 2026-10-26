import sys

import pandas as pd
import pytest

from deepkoala import cli


def test_cli_help(run_cli):
    result = run_cli("--help")
    assert result.returncode == 0, result.stderr
    assert "--device" in result.stdout
    assert "--profiles_dir" in result.stdout


@pytest.mark.integration
@pytest.mark.parametrize("workers", [0, 2])
def test_cli_cpu_inference_from_another_directory(run_cli, fasta_file, tmp_path, workers):
    output = tmp_path / "output results 输出" / "predictions.csv"
    result = run_cli(
        "-i", fasta_file, "-o", output, "--device", "cpu", "-d", "202501",
        "--num_workers", workers, "--batch_size", 2, "--detail", "--topk", 2,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    frame = pd.read_csv(output, keep_default_na=False)
    assert list(frame.columns) == ["name", "predict_label", "probability", "threshold", "annotate"]
    assert frame["name"].tolist() == ["alpha", "alpha", "beta", "beta", "gamma", "gamma"]
    assert frame["predict_label"].tolist() == ["K00001", "K00002"] * 3
    assert frame["probability"].tolist() == pytest.approx([0.75, 0.25] * 3)
    assert frame["annotate"].tolist() == ["*", ""] * 3
    assert "Processed 3 sequences, annotated 3." in result.stdout


@pytest.mark.parametrize("args", [[], ["-i", "missing", "-o", "out", "--device", "invalid"]])
def test_cli_usage_errors(run_cli, args):
    result = run_cli(*args)
    assert result.returncode == 2
    assert "error:" in result.stderr.lower()


@pytest.mark.integration
def test_cli_runtime_error_has_nonzero_exit(run_cli, tmp_path):
    output = tmp_path / "should-not-exist.csv"
    result = run_cli("-i", tmp_path / "missing.fasta", "-o", output, "-d", "202501", "--device", "cpu", "-nw", 0)
    assert result.returncode == 1
    assert "Error:" in result.stdout + result.stderr
    assert not output.exists()


@pytest.mark.parametrize("multi,device", [(False, "mps"), (True, "cpu")])
def test_cli_passes_device_to_inference(monkeypatch, tmp_path, multi, device):
    received = {}

    def fake_inference(**kwargs):
        received.update(kwargs)
        return {"total": 0, "annotated": 0}

    name = "inference_precision" if multi else "inference"
    monkeypatch.setattr(cli, name, fake_inference)
    argv = ["deepkoala", "-i", "input.fasta", "-o", str(tmp_path / "out.csv"), "--device", device]
    if multi:
        argv += ["--multi", "--profiles_dir", "profiles"]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert received["device"] == device


def test_cli_requires_profiles_for_multi_mode(monkeypatch, capsys):
    # Explicit empty value also overrides any machine-specific local default.
    monkeypatch.setattr(sys, "argv", ["deepkoala", "-i", "in", "-o", "out", "--multi", "--profiles_dir", ""])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert "--profiles_dir must be provided" in capsys.readouterr().out
