import importlib.util
import pathlib
import pytest


spec = importlib.util.spec_from_file_location(
    "deepkoala_module",
    pathlib.Path(__file__).resolve().parents[1] / "deepkoala" / "deepkoala.py",
)
deepkoala = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deepkoala)

find_latest_date = deepkoala.find_latest_date


def test_find_latest_date_latest(tmp_path):
    (tmp_path / "202401").mkdir()
    (tmp_path / "202402").mkdir()
    result = find_latest_date("latest", str(tmp_path))
    assert result == "202402"


def test_find_latest_date_specific(tmp_path):
    (tmp_path / "202401").mkdir()
    result = find_latest_date("202401", str(tmp_path))
    assert result == "202401"


def test_find_latest_date_invalid_format(tmp_path):
    with pytest.raises(SystemExit):
        find_latest_date("20-01", str(tmp_path))
