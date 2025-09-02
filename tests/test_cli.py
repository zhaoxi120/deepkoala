import subprocess
from pathlib import Path

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
            "-m", "full_length",
            "-d", "latest",
            "-of", "simple",
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
