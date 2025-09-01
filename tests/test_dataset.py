import importlib.util
import pathlib
import pytest


spec = importlib.util.spec_from_file_location(
    "deepkoala_module",
    pathlib.Path(__file__).resolve().parents[1] / "deepkoala" / "deepkoala.py",
)
deepkoala = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deepkoala)

ProteinDataset = deepkoala.ProteinDataset


def test_tokenize_handles_lowercase(tmp_path):
    fasta = tmp_path / "sample.fasta"
    fasta.write_text(">seq1\nACD\n")
    dataset = ProteinDataset(str(fasta))
    assert dataset.tokenize("acd") == [2, 3, 4]
