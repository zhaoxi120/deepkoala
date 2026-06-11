import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepkoala.utils import resolve_device


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
