# Testing DeepKOALA

The test suite checks the existing inference scripts without changing their
prediction logic. Run it from the repository root with Python 3.11 or 3.12.
Tests use temporary files and small synthetic inputs rather than private FASTA
files, published model weights, KOfam downloads, or a GPU.

GitHub CI is limited to lightweight platform compatibility checks. Large-input
performance comparisons, seven-preset VRAM stress tests and real GPU OOM tests
are manual local validation, not pull-request, scheduled or release jobs.
The current implementation receives one local large-input comparison; its raw
FASTA, checkpoints, process logs and measurements stay in the ignored
`test-results/` directory. Do not upload these as CI artifacts or make CI depend
on their availability. Future performance investigations can be run locally when
needed; they are not recurring CI requirements.
Links into `test-results/` in validation reports refer to locally retained records,
not files published in the GitHub repository or workflow artifacts.
The [final cleanup validation](final-validation.md) records the cleanup checks;
the subsequent [known-issue fixes](known-issue-fixes.md) record the FASTA ID and
HMMER temporary-file repairs and the remaining limitations.

## Install in a separate environment

For Linux or macOS:

```bash
python3.11 -m venv .venv-test
source .venv-test/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-test.txt
python -m pytest
```

For Windows PowerShell:

```powershell
py -3.11 -m venv .venv-test
.\.venv-test\Scripts\python.exe -m pip install --upgrade pip
.\.venv-test\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv-test\Scripts\python.exe -m pytest
```

Python 3.12 can be substituted in either example. These commands create a test
environment; they do not upgrade dependencies in an existing analysis environment.
The test requirements include the unchanged runtime requirements plus pytest 8.
Installing these dependencies requires package access;
the default test cases themselves do not download models or dependencies.

To match the CI baseline, install PyTorch 2.4.1 before the test requirements.
On Linux and Windows, use its CPU wheel:

```console
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
```

On Apple Silicon macOS, use `python -m pip install torch==2.4.1` instead. For
hardware tests, install the PyTorch build appropriate to the device. The CPU CI
matrix does not validate Intel macOS or every supported Python/PyTorch combination.

## Run and interpret the suite

```console
python -m pytest
python -m pytest -m known_issue
python -m pytest --junitxml=test-results/junit.xml
```

`pytest.ini` discovers tests under `tests/` and uses `-ra` to show skipped and
expected-failure reasons. Warnings are not globally disabled. Always review the
summary: an exit code of zero can include explicitly documented expected failures.

| Marker | Purpose | Default behavior |
| --- | --- | --- |
| `integration` | Exercise collaborating components or subprocesses using synthetic inputs | Run |
| `known_issue` | Assert desired behavior for a specific existing defect | Run with a strict expected failure where the defect remains |
| `hardware` | Execute inference on an explicitly selected real CUDA or MPS device | Skip unless `--run-hardware` is supplied |
| `hmmer` | Execute real HMMER programs on synthetic inputs | Skip unless `--run-hmmer` is supplied |
| `benchmark` | Local performance or memory measurement | Excluded from CI; current benchmark scripts live outside `tests/` |

An `XFAIL` records a known defect; it is not a passing functionality check.
Expected failures are strict, so a test that unexpectedly succeeds becomes an
`XPASS(strict)` failure. After fixing a defect, remove or narrow its expected-failure
marker and keep the regression assertion. Do not add broad skips or catch-all
expected failures to make a run green. To see the underlying failure while working
on a known issue, use `python -m pytest -m known_issue --runxfail`.

## What the tests establish

The default suite covers FASTA parsing and tokenization, batching, device
selection, CLI argument forwarding and subprocess invocation, inference output
and counts with synthetic models, domain-boundary logic with controlled HMMER
responses. Synthetic checkpoints exercise the
code path without requiring the production `resources/` directory.

The supported CI usage path is a source checkout, runtime dependencies installed
from `requirements.txt`, and `python -m deepkoala.cli`. Wheel construction,
installation, metadata and packaged-resource checks have been removed from the
suite. This changes the testing scope; it does not add standalone wheel support.

These checks establish software behavior for the tested inputs. They do not
measure KO assignment accuracy, reproduce the paper, validate published weights,
benchmark throughput, or certify production model results on a GPU. Those tasks
require separately recorded model versions, data provenance, and reference results.

## Existing script constraints

All strict expected failures carry the `known_issue` marker. The current list is:

| Existing defect | Regression coverage | When it runs |
| --- | --- | --- |
| HMM profile paths containing spaces are not quoted in the shell command | `test_hmmer.py`: execute real HMMER using a profile path containing spaces | Explicit `--run-hmmer` only |

The former FASTA-ID mismatch and HMMER temporary-FASTA cleanup failures are now
fixed and run as ordinary required tests. ID checks cover eager, streaming,
adaptive-window and multi-domain readers with spaces/tabs and empty headers.
Cleanup checks cover hits, no hits, missing executables, handled/non-handled
process errors and malformed coordinates. Their `xfail` markers were removed.
After removal of the wheel tests, the lightweight CI selection has no expected
failures. The opt-in HMMER path-with-spaces test remains a strict expected failure.

The inference code currently locates model files in `resources/YYYYMM/` next to
the package directory. Use the documented source-checkout layout and install
dependencies from `requirements.txt`; removing wheel tests does not change model
lookup or add runtime dependencies/model resources to the wheel.

Real HMMER tests for a normal hit and for zero hits are not marked as expected
failures on Windows. They require the subprocess to exit successfully as well as
return the expected coordinates. Windows may expose the existing implementation's
attempt to let HMMER write to an already-open temporary file. Such an error must
fail an explicitly requested run; a process failure must not be counted as a
successful zero-hit result.

The two-worker DataLoader check requires working subprocess creation and local
interprocess communication, including Unix-domain sockets on POSIX. A sandbox
that denies these operations can cause a failure or timeout. Re-run in an
environment that permits them; this is not a `known_issue` expected failure and
must not be used to claim that multiprocessing passed.

Local copies may contain machine-specific defaults for model dates, profile
directories, or the `hmmsearch` executable. Tests pass explicit arguments or
control dependencies where appropriate; passing isolated tests does not certify
that a local default path exists on another machine. This testing change does not
rewrite those inference scripts or their local settings.

## Opt in to external tools and real hardware

Run CUDA and MPS checks separately on machines that actually provide the requested
device:

```console
python -m pytest -m hardware --run-hardware --hardware-device cuda
python -m pytest -m hardware --run-hardware --hardware-device mps
```

`--run-hardware` requires `--hardware-device`. A missing or unavailable requested
device must fail the run rather than silently skip or fall back to CPU. Synthetic
model inference checks backend execution; it still does not validate biological
accuracy with the production models.

To exercise HMMER, first make both `hmmbuild` and `hmmsearch` available on `PATH`,
then run:

```console
python -m pytest -m hmmer --run-hmmer
```

After opting in, missing required executables must fail the run. These tests use
synthetic sequences and an HMM built for the test, so they do not require the
KOfam database. Record the operating system, Python/PyTorch or HMMER version,
device, command, and test summary when reporting an external-tool or hardware run.

A local Linux run on 2026-10-02 with HMMER 3.4 verified the normal-hit and zero-hit
cases. The profile-path-with-spaces case remained a strict expected failure.
This local result does not establish Windows or macOS HMMER compatibility.

## GitHub Actions

The [test workflow](../.github/workflows/tests.yml) runs on pull requests, pushes
to `main`, and manual dispatch. It defines four CPU jobs:

| Platform | Python | PyTorch | Purpose |
| --- | --- | --- | --- |
| Linux | 3.11 | 2.4.1 | Baseline compatibility |
| Windows | 3.11 | 2.4.1 | Paths, subprocesses and file handling |
| Apple Silicon macOS | 3.11 | 2.4.1 | macOS/ARM CPU compatibility |
| Linux | 3.12 | Latest allowed by `>=2.4.1,<3` | Additional runtime-version compatibility |

Each job uses small synthetic models and FASTA inputs to check dependency
installation, imports, CLI help and error exit codes, `full`/`frag` CPU inference,
CSV output, Unicode/space-containing paths, and worker/spawn behavior. Existing
fast logic regression checks remain included. The generic
no-`fchmod` output test also runs on Windows; only POSIX permission-bit checks
are skipped there. Wheel tests are not part of this workflow.

Sparse checkout includes the package, tests and root configuration files;
production model resources and local benchmark directories are not checked out.
Each job runs:

```console
python -m pytest tests -m "not hardware and not hmmer and not benchmark" --junitxml=test-results/junit.xml
```

Only the small `test-results/junit.xml` report is uploaded, even when tests fail.
There are no CI throughput targets, large datasets, preset stress jobs, benchmark
artifact uploads or GPU/PBS submissions. The same lightweight scope applies to
manual workflow dispatch. JUnit suite properties record the actual OS,
architecture, Python, PyTorch, NumPy, pandas, and pytest versions; explicit
accelerator runs also record the device. GitHub-hosted macOS CPU execution and mocked device
selection are not evidence of a successful MPS run. The workflow does not opt in
to real hardware or HMMER tests. A configured matrix is a validation plan; only
completed workflow results show which jobs passed for a particular commit.

The workflow uses the v6 action series with Node 24. The official
[checkout](https://github.com/actions/checkout/tree/v6),
[setup-python](https://github.com/actions/setup-python/tree/v6), and
[upload-artifact](https://github.com/actions/upload-artifact/tree/v6) documentation
specifies Actions Runner 2.327.1 or later. The workflow targets current
[GitHub-hosted runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners);
check this minimum if adapting it to a self-hosted runner.
