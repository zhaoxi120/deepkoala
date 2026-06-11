import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepkoala import cli

def test_cli_with_fixed_fasta(tmp_path: Path):
    # 1. Locate the fixed input FASTA file under tests/
    fasta_file = Path(__file__).parent / "test.fasta"
    assert fasta_file.exists(), "test.fasta not found in tests/ directory"

    # 2. Define the output CSV path inside pytest's temporary directory
    out_csv = tmp_path / "out.csv"

    # 3. Run the CLI using subprocess
    result = subprocess.run(
        [
            "python", "-m", "deepkoala.cli",
            "-i", str(fasta_file),
            "-o", str(out_csv),
            "-m", "full",
            "-d", "latest",
        ],
        capture_output=True,
        text=True,
    )

    # 4. Validate the execution result
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert out_csv.exists(), "Output CSV was not created"

    # Optional: check if CSV file contains expected header
    csv_text = out_csv.read_text()
    assert "name" in csv_text


def test_cli_help():
    # This test does not require resources or weights,
    # it only verifies that the CLI help command works.
    result = subprocess.run(
        ["python", "-m", "deepkoala.cli", "-h"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_cli_passes_device_to_inference(monkeypatch, tmp_path):
    received = {}

    def fake_inference(**kwargs):
        received.update(kwargs)
        return {"total": 0, "annotated": 0}

    monkeypatch.setattr(cli, "inference", fake_inference)
    monkeypatch.setattr(
        sys,
        "argv",
        ["deepkoala", "-i", "input.fasta", "-o", str(tmp_path / "out.csv"), "--device", "mps"],
    )

    cli.main()

    assert received["device"] == "mps"


def test_cli_passes_device_to_multi_domain_inference(monkeypatch, tmp_path):
    received = {}

    def fake_inference_precision(**kwargs):
        received.update(kwargs)
        return {"total": 0, "annotated": 0}

    monkeypatch.setattr(cli, "inference_precision", fake_inference_precision)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "deepkoala",
            "-i",
            "input.fasta",
            "-o",
            str(tmp_path / "out.csv"),
            "--multi",
            "--profiles_dir",
            "profiles",
            "--device",
            "cpu",
        ],
    )

    cli.main()

    assert received["device"] == "cpu"
