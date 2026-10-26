"""Self-contained test resources; never load the user's installed models."""

import json
import math
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from deepkoala.model import GRUClassifier


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def record_environment(record_testsuite_property):
    from importlib.metadata import version

    record_testsuite_property("platform", platform.platform())
    record_testsuite_property("architecture", platform.machine())
    record_testsuite_property("python", sys.version)
    for package in ("torch", "numpy", "pandas", "pytest"):
        record_testsuite_property(package, version(package))


def pytest_addoption(parser):
    parser.addoption("--run-hardware", action="store_true", help="Run real accelerator tests")
    parser.addoption("--hardware-device", choices=("cuda", "mps"), help="Required accelerator")
    parser.addoption("--run-hmmer", action="store_true", help="Run tests requiring HMMER on PATH")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-hardware") and not config.getoption("--hardware-device"):
        raise pytest.UsageError("--run-hardware requires --hardware-device cuda or mps")
    for item in items:
        for marker, option in (("hardware", "--run-hardware"), ("hmmer", "--run-hmmer")):
            if marker in item.keywords and not config.getoption(option):
                item.add_marker(pytest.mark.skip(reason=f"Opt in with {option}; no real hardware/tool coverage by default"))


@pytest.fixture(scope="session", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="session")
def toy_resources(tmp_path_factory):
    resources = tmp_path_factory.mktemp("synthetic-models") / "resources"
    version = resources / "202501"
    version.mkdir(parents=True)
    config = {
        "K00001": {"index": 0, "threshold": 0.7},
        "K00002": {"index": 1, "threshold": 0.3},
    }
    # A real GRU checkpoint with known output, not biological weights.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        classifier = GRUClassifier(128, 2, 2)
        with torch.no_grad():
            for parameter in classifier.parameters():
                parameter.zero_()
            classifier.cls.bias[0] = math.log(3)
        for variant in ("full", "frag"):
            (version / f"ko_config_{variant}.json").write_text(json.dumps(config), encoding="utf-8")
            torch.save(classifier.state_dict(), version / f"weights_{variant}.pt")
    return resources


@pytest.fixture
def fasta_file(tmp_path):
    target = tmp_path / "input sequences 蛋白.fasta"
    shutil.copyfile(ROOT / "tests" / "fixtures" / "synthetic.fasta", target)
    return target


@pytest.fixture
def isolated_runtime(tmp_path, toy_resources):
    runtime = tmp_path / "isolated runtime"
    shutil.copytree(ROOT / "deepkoala", runtime / "deepkoala", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(toy_resources, runtime / "resources")
    return runtime


@pytest.fixture
def runtime_modules(monkeypatch, isolated_runtime):
    from deepkoala import infer, infer_multi

    # Redirect source-relative resource lookup only for this test process.
    for module in (infer, infer_multi):
        monkeypatch.setattr(module, "__file__", str(isolated_runtime / "deepkoala" / Path(module.__file__).name))
    return infer, infer_multi


@pytest.fixture
def run_cli(isolated_runtime, tmp_path):
    outside = tmp_path / "outside repository"
    outside.mkdir()
    env = os.environ.copy()
    env.update(PYTHONPATH=str(isolated_runtime), PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def run(*args):
        return subprocess.run(
            [sys.executable, "-m", "deepkoala.cli", *map(str, args)],
            cwd=outside, env=env, capture_output=True, text=True, encoding="utf-8", timeout=90,
        )

    return run


@pytest.fixture
def hardware_device(request):
    name = request.config.getoption("--hardware-device")
    if name == "cuda":
        available = torch.cuda.is_available()
    elif name == "mps":
        available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else:
        pytest.fail("Specify --hardware-device cuda or mps")
    if not available:
        pytest.fail(f"Requested hardware test device {name!r} is not available")
    if name == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        pytest.fail("Disable PYTORCH_ENABLE_MPS_FALLBACK for genuine MPS validation")
    return torch.device(name)
