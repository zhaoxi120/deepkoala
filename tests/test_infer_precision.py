import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepkoala import infer_precision


def test_run_hmmsearch_returns_none_on_no_hits(monkeypatch, tmp_path):
    hmm_file = tmp_path / "dummy.hmm"
    hmm_file.write_text("")

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(subprocess, "run", fake_run)

    start, end = infer_precision._run_hmmsearch(hmm_file, "A" * 60)

    assert start is None and end is None


def test_annotate_sequence_fallback_without_hmm_bounds(monkeypatch, tmp_path):
    sequence = "A" * 60
    hmm_dir = tmp_path
    hmm_file = hmm_dir / "KO.hmm"
    hmm_file.write_text("")

    monkeypatch.setattr(infer_precision, "_classify", lambda *args, **kwargs: (0, 0.9))
    monkeypatch.setattr(infer_precision, "_run_hmmsearch", lambda *args, **kwargs: (None, None))

    model = object()
    idx2ko = {0: "KO"}
    thresholds = {"KO": 0.1}

    hits = infer_precision._annotate_sequence(
        sequence,
        model,
        idx2ko,
        thresholds,
        hmm_dir,
        None,
    )

    assert len(hits) == 1
    assert hits[0].start == 1
    assert hits[0].end == len(sequence)
