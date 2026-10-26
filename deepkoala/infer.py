from contextlib import contextmanager
import io
import os
from pathlib import Path
import secrets
import stat
import tarfile
import tempfile
import zipfile

import pandas as pd
from pandas.io.common import get_handle, infer_compression
import torch
from tqdm import tqdm

from .adaptive import AdaptivePredictor, predict_tensors
from .batching import PRESETS
from .data import get_dataloader
from .model import GRUClassifier
from .utils import find_latest_date, load_ko_config, resolve_device


@contextmanager
def _csv_handle(temporary, output):
    compression = infer_compression(str(output), "infer")
    if compression == "zip":
        # pandas buffers ZIP/TAR payloads in BytesIO; stream archives ourselves.
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            with archive.open(output.stem, "w", force_zip64=True) as member:
                with io.TextIOWrapper(member, encoding="utf-8", newline="") as handle:
                    yield handle
    elif compression == "tar":
        # A TAR header needs the final size. Spool CSV to disk, never to RAM.
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="", dir=output.parent) as handle:
            yield handle
            handle.flush()
            size = os.fstat(handle.fileno()).st_size
            handle.buffer.seek(0)
            suffix = output.suffix
            codec = {".gz": "gz", ".bz2": "bz2", ".xz": "xz"}.get(suffix, "")
            name = output.name[:-len(".tar" + suffix)] if codec else output.stem
            with tarfile.open(temporary, "w:" + codec) as archive:
                member = tarfile.TarInfo(name)
                member.size = size
                archive.addfile(member, handle.buffer)
    else:
        with get_handle(temporary, "w", compression=compression) as handles:
            yield handles.handle


@contextmanager
def _atomic_csv_output(output_path):
    """Publish the streamed CSV only after every batch succeeds.

    Keeping the temporary file alongside the destination also permits input
    and output to be the same path without truncating the FASTA being read.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_mode = stat.S_IMODE(output.stat().st_mode)
    except FileNotFoundError:
        output_mode = None
    while True:
        # A valid destination may already approach the filesystem's name limit.
        # Compression is selected from output by _csv_handle, not this name.
        temporary = output.parent / f".deepkoala-{secrets.token_hex(12)}.tmp"
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            break
        except FileExistsError:
            continue
    try:
        try:
            # Capture normal creation permissions without changing the process
            # umask. Keep incomplete output private until it is published.
            if output_mode is None:
                output_mode = stat.S_IMODE(os.fstat(fd).st_mode)
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            else:
                os.chmod(temporary, 0o600)
        finally:
            os.close(fd)
        # Match to_csv's extension-based compression and text handling.
        with _csv_handle(temporary, output) as handle:
            yield handle
        os.chmod(temporary, output_mode)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def inference(
    input_path: str,
    output_path: str,
    model: str = "full",
    date: str = "latest",
    batch_size: int | None = None,
    num_workers: int = 2,
    detail: bool = False,
    device: str | torch.device | None = None,
    topk: int = 1,
    memory_preset: str = "auto",
    sort_window: int = 4096,
    max_padded_residues: int | None = None,
    oom_fallback: str = "error",
):
    """Stream predictions to CSV, publishing the output after a successful run.

    CUDA uses measured adaptive batches by default; CPU/MPS and preset ``off``
    keep fixed batches. Return per-sequence counts and optional CUDA batch stats.
    """
    device = resolve_device(device)
    if memory_preset not in ("auto", "off", *PRESETS):
        raise ValueError("Unknown memory_preset; choose auto, off, 2, 4, 8, 16, 24, 32, or gt32")
    if memory_preset not in ("auto", "off") and device.type != "cuda":
        raise ValueError("Explicit GPU memory presets require a CUDA device")
    adaptive = device.type == "cuda" and memory_preset != "off"
    if oom_fallback not in ("error", "cpu"):
        raise ValueError("oom_fallback must be error or cpu")
    for name, value in (("batch_size", batch_size), ("sort_window", sort_window),
                        ("max_padded_residues", max_padded_residues)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    if topk < 1:
        raise ValueError("topk must be >= 1")
    batch_size = batch_size if batch_size is not None else (512 if adaptive else 64)
    resources_dir = Path(__file__).resolve().parent.parent / "resources"

    db_date = find_latest_date(date, resources_dir)
    ko_cfg = resources_dir / db_date / f"ko_config_{model}.json"
    weights = resources_dir / db_date / f"weights_{model}.pt"

    ko2idx, idx2ko, threshold = load_ko_config(str(ko_cfg))
    model = GRUClassifier(128, 2, len(ko2idx))
    checkpoint = torch.load(str(weights), map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint)
    del checkpoint
    model.to(device)

    def cpu_model_factory():
        fallback = GRUClassifier(128, 2, len(ko2idx))
        fallback.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True))
        return fallback.eval()

    def prediction_chunks(predictor):
        if predictor is not None:
            for predictions in predictor.iter_predictions(input_path, sort_window):
                yield ([r.name for r, _, _ in predictions],
                       [indices for _, indices, _ in predictions],
                       [values for _, _, values in predictions])
                del predictions
        else:
            loader = get_dataloader(input_path, batch_size, num_workers)
            for names, seqs, lens in loader:
                indices, values = predict_tensors(model, seqs, lens, device, topk)
                yield names, indices, values

    total_sequences, annotated_sequences = 0, 0
    columns = ["name", "predict_label", "probability", "threshold", "annotate"]
    output_columns = columns if detail else columns[:2]

    model.eval()
    with torch.no_grad(), _atomic_csv_output(output_path) as output:
        predictor = AdaptivePredictor(
            model, device, topk, memory_preset=memory_preset,
            max_batch_size=batch_size, max_padded_residues=max_padded_residues,
            oom_fallback=oom_fallback, cpu_model_factory=cpu_model_factory,
        ) if adaptive else None
        pd.DataFrame(columns=output_columns).to_csv(output, index=False)
        for seq_names, top_idx, top_prob in tqdm(prediction_chunks(predictor), 'Inference Progress'):
            rows = []

            for seq_name, idx_row, prob_row in zip(
                seq_names, top_idx, top_prob
            ):
                total_sequences += 1
                seq_annotated = False

                for pred_idx, pred_prob in zip(idx_row, prob_row):
                    pred = idx2ko[pred_idx]
                    th = threshold[pred]
                    mark = '*' if pred_prob >= th else ''

                    rows.append((seq_name, pred, pred_prob, th, mark))

                    if mark:
                        seq_annotated = True

                if seq_annotated:
                    annotated_sequences += 1

            df = pd.DataFrame(rows, columns=columns).round(4)
            if not detail:
                df.loc[df["annotate"] != "*", "predict_label"] = pd.NA
                df = df[output_columns]
            df.to_csv(output, index=False, header=False)
    summary = {
        "total": total_sequences,
        "annotated": annotated_sequences,
        "output": str(output_path),
    }
    if predictor is not None:
        summary["batching"] = predictor.summary()
    return summary
