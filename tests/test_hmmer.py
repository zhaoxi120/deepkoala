"""HMMER parsing contracts, plus opt-in tests of the real external program."""

from pathlib import Path
import random
import shutil
import subprocess

import pytest

from deepkoala import infer_multi


def domtbl_row(start=7, end=53):
    # HMMER domtblout: target, accession, tlen, query, accession, qlen,
    # full E/score/bias, domain index/count, c-E/i-E/score/bias,
    # hmm from/to, alignment from/to, envelope from/to, accuracy, description.
    return f"query - 60 profile - 50 1e-20 70.0 0.0 1 1 1e-20 1e-20 70.0 0.0 1 50 {start} {end} 5 55 0.98 synthetic\n"


@pytest.fixture
def controlled_hmmsearch(monkeypatch, tmp_path):
    """Simulate only process output; parsing and FASTA creation remain real.

    The fake child writes the output file itself. Production still creates,
    closes, reads, and removes its real temporary files.
    """
    real_temporary_directory = infer_multi.tempfile.TemporaryDirectory
    state = {"output": "", "error": None, "calls": [], "fasta_paths": [],
             "domtbl_paths": [], "fasta_contents": []}

    def temporary_directory(*args, **kwargs):
        kwargs["dir"] = tmp_path
        return real_temporary_directory(*args, **kwargs)

    def run(command, **kwargs):
        state["calls"].append((command, kwargs))
        fasta_path = Path(command[-1])
        domtbl_path = Path(command[command.index("--domtblout") + 1])
        state["fasta_paths"].append(fasta_path)
        state["domtbl_paths"].append(domtbl_path)
        state["fasta_contents"].append(fasta_path.read_text(encoding="utf-8"))
        assert not domtbl_path.exists()
        if state["error"] is not None:
            raise state["error"]
        domtbl_path.write_text(state["output"], encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(infer_multi.tempfile, "TemporaryDirectory", temporary_directory)
    monkeypatch.setattr(infer_multi.subprocess, "run", run)
    return state


def test_hmmsearch_reads_alignment_not_envelope_or_profile_coordinates(controlled_hmmsearch, tmp_path):
    state = controlled_hmmsearch
    state["output"] = "# HMMER output\n" + domtbl_row(7, 53) + domtbl_row(10, 20)

    assert infer_multi._run_hmmsearch(tmp_path / "profile.hmm", "A" * 60) == (7, 53)
    assert state["fasta_contents"][0] == ">query\n" + "A" * 60 + "\n"
    assert len(state["calls"]) == 1
    _, kwargs = state["calls"][0]
    assert kwargs["check"] is True


def test_hmmsearch_passes_literal_paths_without_a_shell(controlled_hmmsearch, tmp_path):
    state = controlled_hmmsearch
    state["output"] = domtbl_row(7, 53)
    profile = tmp_path / "profile with spaces;literal$(text).hmm"

    assert infer_multi._run_hmmsearch(profile, "A" * 60) == (7, 53)

    command, kwargs = state["calls"][0]
    assert command == ["hmmsearch", "--noali", "--domtblout",
                       str(state["domtbl_paths"][0]), str(profile),
                       str(state["fasta_paths"][0])]
    assert not kwargs.get("shell", False)
    assert kwargs["stdout"] == subprocess.DEVNULL


@pytest.mark.parametrize("output", ["", "# no hits\n", "\n# comment\nnot a domain row\n"])
def test_successful_hmmsearch_without_domain_rows_returns_no_bounds(controlled_hmmsearch, tmp_path, output):
    controlled_hmmsearch["output"] = output

    assert infer_multi._run_hmmsearch(tmp_path / "profile.hmm", "A" * 60) == (None, None)


def test_hmmsearch_skips_short_lines_before_a_valid_domain(controlled_hmmsearch, tmp_path):
    controlled_hmmsearch["output"] = "\nshort line\n# comment\n" + domtbl_row(1, 60)

    assert infer_multi._run_hmmsearch(tmp_path / "profile.hmm", "A" * 60) == (1, 60)


@pytest.mark.parametrize("start, end", [("bad", 53), (7, "bad")])
def test_invalid_hmm_coordinates_raise_parse_error(controlled_hmmsearch, tmp_path, start, end):
    controlled_hmmsearch["output"] = domtbl_row(start, end)

    with pytest.raises(ValueError, match="invalid literal"):
        infer_multi._run_hmmsearch(tmp_path / "profile.hmm", "A" * 60)


def test_exit_one_currently_returns_no_bounds(controlled_hmmsearch, tmp_path):
    # Characterizes current behavior; exit code 1 is not proof of a valid
    # zero-hit search. A successful zero-hit search is tested independently.
    controlled_hmmsearch["error"] = subprocess.CalledProcessError(1, "hmmsearch")

    assert infer_multi._run_hmmsearch(tmp_path / "profile.hmm", "A" * 60) == (None, None)


@pytest.mark.parametrize("returncode", [2, 127])
def test_non_one_process_failures_are_not_hidden(controlled_hmmsearch, tmp_path, returncode):
    controlled_hmmsearch["error"] = subprocess.CalledProcessError(returncode, "hmmsearch", stderr=b"failed")

    with pytest.raises(subprocess.CalledProcessError) as raised:
        infer_multi._run_hmmsearch(tmp_path / "profile.hmm", "A" * 60)

    assert raised.value.returncode == returncode
    assert raised.value.stderr == b"failed"


def test_missing_executable_exception_reports_fallback(controlled_hmmsearch, tmp_path, capsys):
    controlled_hmmsearch["error"] = FileNotFoundError("hmmsearch")

    assert infer_multi._run_hmmsearch(tmp_path / "profile.hmm", "A" * 60) == (None, None)
    assert "hmmsearch not found" in capsys.readouterr().out


@pytest.mark.parametrize(
    "output, error, expected_error",
    [
        (domtbl_row(), None, None),
        ("", None, None),
        ("", FileNotFoundError("hmmsearch"), None),
        ("", subprocess.CalledProcessError(1, "hmmsearch"), None),
        ("", subprocess.CalledProcessError(2, "hmmsearch"), subprocess.CalledProcessError),
        (domtbl_row("bad"), None, ValueError),
    ],
    ids=["hit", "no-hit", "missing-executable", "exit-one", "process-error", "parse-error"],
)
def test_hmmsearch_removes_temporary_files_on_success_and_failure(
    controlled_hmmsearch, tmp_path, output, error, expected_error
):
    controlled_hmmsearch["output"] = output
    controlled_hmmsearch["error"] = error
    if expected_error is None:
        infer_multi._run_hmmsearch(tmp_path / "profile.hmm", "A" * 60)
    else:
        with pytest.raises(expected_error):
            infer_multi._run_hmmsearch(tmp_path / "profile.hmm", "A" * 60)

    assert controlled_hmmsearch["fasta_paths"]
    assert not any(path.exists() for path in controlled_hmmsearch["fasta_paths"])
    assert not any(path.exists() for path in controlled_hmmsearch["domtbl_paths"])
    assert not any(path.parent.exists() for path in controlled_hmmsearch["fasta_paths"])


@pytest.fixture
def real_hmm_profile(tmp_path, monkeypatch):
    executables = {name: shutil.which(name) for name in ("hmmbuild", "hmmsearch")}
    missing = [name for name, executable in executables.items() if executable is None]
    if missing:
        pytest.fail("--run-hmmer requires tools on PATH: " + ", ".join(missing))
    rng = random.Random(713)
    sequence = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(180))
    alignment = tmp_path / "reference.fasta"
    alignment.write_text(">reference\n" + sequence + "\n", encoding="ascii")
    profile = tmp_path / "reference.hmm"
    subprocess.run(
        [executables["hmmbuild"], "--amino", str(profile), str(alignment)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    original_run = subprocess.run
    returncodes = []

    def run_with_available_executable(command, **kwargs):
        # Replace only the executable; keep all production argument boundaries.
        assert not kwargs.get("shell", False)
        command = [executables["hmmsearch"], *command[1:]]
        kwargs.setdefault("timeout", 60)
        try:
            completed = original_run(command, **kwargs)
        except subprocess.CalledProcessError as error:
            returncodes.append(error.returncode)
            raise
        returncodes.append(completed.returncode)
        return completed

    monkeypatch.setattr(infer_multi.subprocess, "run", run_with_available_executable)
    # Confine generated temporary FASTA files to pytest's managed test folder.
    monkeypatch.setattr(infer_multi.tempfile, "tempdir", str(tmp_path))
    return profile, sequence, returncodes


@pytest.mark.hmmer
def test_real_hmmsearch_finds_synthetic_reference(real_hmm_profile):
    profile, sequence, returncodes = real_hmm_profile

    assert infer_multi._run_hmmsearch(profile, sequence) == (1, len(sequence))
    assert returncodes == [0]


@pytest.mark.hmmer
def test_real_hmmsearch_reports_zero_hits(real_hmm_profile):
    profile, _, returncodes = real_hmm_profile

    assert infer_multi._run_hmmsearch(profile, "A" * 180) == (None, None)
    # Exit code 1 is swallowed by production code; it must never count as a
    # successful zero-hit integration test (including Windows file-open errors).
    assert returncodes == [0]


@pytest.mark.hmmer
def test_real_hmmsearch_accepts_profile_path_with_spaces(real_hmm_profile, tmp_path, monkeypatch):
    profile, sequence, returncodes = real_hmm_profile
    spaced_profile = tmp_path / "profile with spaces.hmm"
    shutil.copyfile(profile, spaced_profile)
    spaced_temporary_directory = tmp_path / "temporary files with spaces"
    spaced_temporary_directory.mkdir()
    monkeypatch.setattr(infer_multi.tempfile, "tempdir", str(spaced_temporary_directory))

    assert infer_multi._run_hmmsearch(spaced_profile, sequence) == (1, len(sequence))
    assert returncodes == [0]
    assert not list(spaced_temporary_directory.iterdir())
