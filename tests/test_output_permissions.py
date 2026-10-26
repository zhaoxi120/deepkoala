"""Atomic publication keeps normal output permissions and private partial data."""

import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from deepkoala.infer import _atomic_csv_output


posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX file-mode behavior")
ROOT = Path(__file__).resolve().parents[1]


@posix_only
@pytest.mark.parametrize("mode", [0o640, 0o444], ids=["group_readable", "read_only"])
def test_atomic_output_preserves_existing_mode_and_keeps_partial_private(tmp_path, mode):
    output = tmp_path / "annotations.csv"
    output.write_text("name,predict_label\nold,K00001\n", encoding="utf-8")
    output.chmod(mode)

    with _atomic_csv_output(output) as handle:
        temporary = [path for path in tmp_path.iterdir() if path != output]
        assert len(temporary) == 1
        assert stat.S_IMODE(temporary[0].stat().st_mode) == 0o600
        handle.write("name,predict_label\nnew,K00002\n")
        assert "old,K00001" in output.read_text(encoding="utf-8")

    assert stat.S_IMODE(output.stat().st_mode) == mode
    assert output.read_text(encoding="utf-8") == "name,predict_label\nnew,K00002\n"
    assert list(tmp_path.iterdir()) == [output]


@posix_only
def test_new_atomic_output_uses_the_same_mode_as_a_normal_text_file(tmp_path):
    reference = tmp_path / "normal.csv"
    reference.write_text("reference\n", encoding="utf-8")
    output = tmp_path / "annotations.csv"

    with _atomic_csv_output(output) as handle:
        assert not output.exists()
        handle.write("name,predict_label\nquery,K00001\n")

    assert stat.S_IMODE(output.stat().st_mode) == stat.S_IMODE(reference.stat().st_mode)


def test_atomic_output_also_works_when_fchmod_is_unavailable(tmp_path, monkeypatch):
    # Windows exposes chmod but does not expose the Unix-only fchmod API.
    monkeypatch.delattr(os, "fchmod", raising=False)
    output = tmp_path / "annotations.csv"
    with _atomic_csv_output(output) as handle:
        handle.write("name,predict_label\nquery,K00001\n")

    assert output.read_text(encoding="utf-8") == "name,predict_label\nquery,K00001\n"


@posix_only
@pytest.mark.parametrize("mode", [0o640, 0o444], ids=["group_readable", "read_only"])
def test_failed_atomic_output_preserves_existing_content_and_mode(tmp_path, mode):
    output = tmp_path / "annotations.csv"
    original = "name,predict_label\noriginal,K00001\n"
    output.write_text(original, encoding="utf-8")
    output.chmod(mode)

    with pytest.raises(RuntimeError, match="inference failed"):
        with _atomic_csv_output(output) as handle:
            handle.write("name,predict_label\npartial,K00002\n")
            raise RuntimeError("inference failed")

    assert output.read_text(encoding="utf-8") == original
    assert stat.S_IMODE(output.stat().st_mode) == mode
    assert list(tmp_path.iterdir()) == [output]


@posix_only
@pytest.mark.parametrize("mask, expected", [("077", 0o600), ("002", 0o664)])
def test_atomic_output_honors_umask_in_an_isolated_process(tmp_path, mask, expected):
    script = """
import json
import os
from pathlib import Path
import stat
import sys
from deepkoala.infer import _atomic_csv_output

directory = Path(sys.argv[1])
os.umask(int(sys.argv[2], 8))
normal = directory / 'normal.csv'
normal.write_text('reference\\n', encoding='utf-8')
output = directory / 'annotations.csv'
with _atomic_csv_output(output) as handle:
    temporary = [path for path in directory.iterdir() if path != normal]
    assert len(temporary) == 1
    assert stat.S_IMODE(temporary[0].stat().st_mode) == 0o600
    handle.write('name,predict_label\\nquery,K00001\\n')
assert len(list(directory.iterdir())) == 2
print(json.dumps({'normal': stat.S_IMODE(normal.stat().st_mode),
                  'atomic': stat.S_IMODE(output.stat().st_mode)}))
"""
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), mask],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"normal": expected, "atomic": expected}
