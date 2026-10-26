import json

import pytest
import torch

from deepkoala.utils import (
    DateResolveError,
    find_latest_date,
    load_ko_config,
    resolve_device,
)


def set_backend_availability(monkeypatch, *, cuda: bool, mps: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def test_resolve_device_auto_prefers_cuda(monkeypatch):
    set_backend_availability(monkeypatch, cuda=True, mps=True)

    assert resolve_device("auto") == torch.device("cuda")


def test_resolve_device_auto_uses_mps_when_cuda_is_unavailable(monkeypatch):
    set_backend_availability(monkeypatch, cuda=False, mps=True)

    assert resolve_device(None) == torch.device("mps")


def test_resolve_device_auto_uses_cpu_when_accelerators_are_unavailable(monkeypatch):
    set_backend_availability(monkeypatch, cuda=False, mps=False)

    assert resolve_device("auto") == torch.device("cpu")


def test_resolve_device_rejects_explicit_unavailable_mps(monkeypatch):
    set_backend_availability(monkeypatch, cuda=False, mps=False)

    with pytest.raises(ValueError, match="mps"):
        resolve_device("mps")


def test_resolve_device_accepts_explicit_cpu(monkeypatch):
    set_backend_availability(monkeypatch, cuda=False, mps=False)

    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_accepts_explicit_available_cuda(monkeypatch):
    set_backend_availability(monkeypatch, cuda=True, mps=False)

    assert resolve_device(torch.device("cuda")) == torch.device("cuda")


def test_resolve_device_explicit_cpu_overrides_available_accelerators(monkeypatch):
    set_backend_availability(monkeypatch, cuda=True, mps=True)

    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_rejects_explicit_unavailable_cuda(monkeypatch):
    set_backend_availability(monkeypatch, cuda=False, mps=False)

    with pytest.raises(ValueError, match="cuda.*not available"):
        resolve_device("cuda")


def test_resolve_device_accepts_explicit_available_mps(monkeypatch):
    set_backend_availability(monkeypatch, cuda=False, mps=True)

    assert resolve_device("mps") == torch.device("mps")


def test_resolve_device_preserves_explicit_cuda_index(monkeypatch):
    set_backend_availability(monkeypatch, cuda=True, mps=False)

    assert resolve_device("cuda:1") == torch.device("cuda:1")


def test_resolve_device_auto_handles_a_build_without_mps_backend(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delattr(torch.backends, "mps", raising=False)

    assert resolve_device("auto") == torch.device("cpu")


@pytest.mark.parametrize("device", ["invalid-device", "meta"])
def test_resolve_device_rejects_unknown_or_unsupported_backends(device):
    with pytest.raises(ValueError, match="Unsupported device"):
        resolve_device(device)


def test_latest_resource_date_uses_only_version_directories(tmp_path):
    for version in ("202412", "202501", "202409"):
        (tmp_path / version).mkdir()
    (tmp_path / "209912").write_text("A file is not a resource release.", encoding="utf-8")
    (tmp_path / "notes").mkdir()
    (tmp_path / "202501_backup").mkdir()

    assert find_latest_date("latest", str(tmp_path)) == "202501"
    assert find_latest_date("202409", str(tmp_path)) == "202409"


def test_latest_resource_date_reports_no_available_versions(tmp_path):
    (tmp_path / "README.txt").write_text("No model releases here.", encoding="utf-8")

    with pytest.raises(DateResolveError, match="No valid YYYYMM directories"):
        find_latest_date("latest", str(tmp_path))


def test_explicit_missing_resource_date_reports_the_requested_version(tmp_path):
    (tmp_path / "202501").mkdir()

    with pytest.raises(DateResolveError, match="202502.*not found"):
        find_latest_date("202502", str(tmp_path))


@pytest.mark.parametrize("date", ["2025-01", "20251", "../202501", ""])
def test_invalid_resource_date_is_rejected(tmp_path, date):
    with pytest.raises(DateResolveError, match="Invalid date format"):
        find_latest_date(date, str(tmp_path))


def test_ko_config_uses_explicit_class_indices_and_preserves_thresholds(tmp_path):
    config = tmp_path / "ko_config_full.json"
    # JSON order deliberately differs from classifier index order.
    config.write_text(
        json.dumps(
            {
                "K00002": {"index": 1, "threshold": 0.875},
                "K00001": {"index": 0, "threshold": 0.125},
            }
        ),
        encoding="utf-8",
    )

    ko_to_index, index_to_ko, thresholds = load_ko_config(str(config))

    assert ko_to_index == {"K00001": 0, "K00002": 1}
    assert index_to_ko == {0: "K00001", 1: "K00002"}
    assert thresholds == {"K00001": 0.125, "K00002": 0.875}


def test_missing_ko_config_reports_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ko_config(str(tmp_path / "missing.json"))


def test_malformed_ko_config_reports_invalid_json(tmp_path):
    config = tmp_path / "invalid.json"
    config.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_ko_config(str(config))
