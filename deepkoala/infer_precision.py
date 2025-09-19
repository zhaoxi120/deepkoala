from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple
import subprocess
import tempfile

import pandas as pd
import torch
import torch.nn.functional as F

from .data import AA_VOCAB
from .model import GRUClassifier
from .utils import find_latest_date, load_ko_config

__all__ = [
    "annotate_precision",
    "inference_precision",
]


DEFAULT_PROFILES_DIR = "/usr/appli/freeware/kofamscan/1.3.0/db/profiles"


@dataclass
class DomainHit:
    """Container describing a detected domain on the original sequence."""

    name: str | None
    predict_label: str
    probability: float
    threshold: float
    start: int
    end: int

    @property
    def annotate(self) -> str:
        return "*" if self.probability >= self.threshold else ""

    def as_row(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "predict_label": self.predict_label,
            "probability": self.probability,
            "threshold": self.threshold,
            "annotate": self.annotate,
            "start": self.start,
            "end": self.end,
        }


def _tokenize(seq: str) -> torch.Tensor:
    """Convert amino acid sequence into indices for the classifier."""

    tokens = [AA_VOCAB.get(aa, 1) for aa in seq]
    return torch.tensor(tokens, dtype=torch.long)


def _classify(model: GRUClassifier, seq: str, device: torch.device) -> Tuple[int, float]:
    """Return (predicted_index, probability) for a fragment."""

    tensor = _tokenize(seq).unsqueeze(0).to(device)
    lens = torch.tensor([tensor.size(1) - 1], dtype=torch.long, device=device)
    logits = model(tensor, lens)
    prob = F.softmax(logits, dim=1)[0]
    mx, idx = torch.max(prob, dim=0)
    return idx.item(), mx.item()


def _run_hmmsearch(hmm_file: Path, seq: str) -> Tuple[int | None, int | None]:
    """Execute ``hmmsearch`` on ``seq`` against ``hmm_file`` and parse bounds."""

    with tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False) as tmp_fa:
        tmp_fa.write(">query\n")
        tmp_fa.write(seq + "\n")
        fasta_path = tmp_fa.name
    with tempfile.NamedTemporaryFile("r") as domtbl:
        cmd = [
            "hmmsearch",
            "--domtblout",
            domtbl.name,
            str(hmm_file),
            fasta_path,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            return None, None
        domtbl.seek(0)
        for line in domtbl:
            if line.startswith("#"):
                continue
            cols = line.strip().split()
            if len(cols) >= 19:
                start = int(cols[17])
                end = int(cols[18])
                return start, end
    return None, None


def _read_fasta(path: str) -> Iterator[Tuple[str, str]]:
    """Yield ``(identifier, sequence)`` pairs from ``path``."""

    with open(path) as handle:
        seq_id: str | None = None
        chunks: List[str] = []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    yield seq_id, "".join(chunks)
                seq_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if seq_id is not None:
            yield seq_id, "".join(chunks)


def _load_model(
    mode: str, date: str, device: torch.device
) -> Tuple[GRUClassifier, Dict[str, int], Dict[int, str], Dict[str, float]]:
    resources_dir = Path(__file__).resolve().parent.parent / "resources"
    db_date = find_latest_date(date, str(resources_dir))
    cfg_path = resources_dir / db_date / f"ko_config_{mode}.json"
    weight_path = resources_dir / db_date / f"weights_{mode}.pt"

    ko2idx, idx2ko, thresholds = load_ko_config(str(cfg_path))
    model = GRUClassifier(128, 2, len(ko2idx)).to(device)
    checkpoint = torch.load(str(weight_path), map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model, ko2idx, idx2ko, thresholds


def _annotate_sequence(
    sequence: str,
    model: GRUClassifier,
    idx2ko: Dict[int, str],
    thresholds: Dict[str, float],
    hmm_dir: Path,
    device: torch.device,
    *,
    name: str | None = None,
) -> List[DomainHit]:
    hits: List[DomainHit] = []
    queue: List[Tuple[str, int]] = [(sequence, 0)]

    while queue:
        frag, offset = queue.pop(0)
        if len(frag) < 50:
            continue
        pred_idx, prob = _classify(model, frag, device)
        ko = idx2ko[pred_idx]
        thr = thresholds[ko]
        if prob < thr:
            continue
        hmm_file = hmm_dir / f"{ko}.hmm"
        start, end = None, None
        if hmm_file.exists():
            start, end = _run_hmmsearch(hmm_file, frag)
        if start is None or end is None:
            dom_start = offset + 1
            dom_end = offset + len(frag)
        else:
            dom_start = offset + start
            dom_end = offset + end
        hits.append(
            DomainHit(
                name=name,
                predict_label=ko,
                probability=prob,
                threshold=thr,
                start=dom_start,
                end=dom_end,
            )
        )
        left = frag[: dom_start - offset - 1]
        right = frag[dom_end - offset :]
        if len(left) >= 50:
            queue.append((left, offset))
        if len(right) >= 50:
            queue.append((right, dom_end))
    return hits


def annotate_precision(
    sequence: str,
    *,
    hmmer_dir: str | None = None,
    date: str = "latest",
    mode: str = "full_length",
    device: torch.device | None = None,
) -> List[Dict[str, float]]:
    """Annotate domains in ``sequence`` using precision mode."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, idx2ko, thresholds = _load_model(mode, date, device)
    hmm_dir = Path(hmmer_dir or DEFAULT_PROFILES_DIR)

    with torch.no_grad():
        hits = _annotate_sequence(sequence, model, idx2ko, thresholds, hmm_dir, device)

    return [
        {
            "ko": hit.predict_label,
            "prob": hit.probability,
            "threshold": hit.threshold,
            "start": hit.start,
            "end": hit.end,
        }
        for hit in hits
    ]


def inference_precision(
    input_path: str,
    output_path: str,
    *,
    mode: str = "full_length",
    date: str = "latest",
    profiles_dir: str | None = None,
    output_format: str = "detail",
    device: torch.device | None = None,
) -> Dict[str, object]:
    """Run precision inference on ``input_path`` and write CSV to ``output_path``."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, idx2ko, thresholds = _load_model(mode, date, device)
    hmm_dir = Path(profiles_dir or DEFAULT_PROFILES_DIR)

    total_sequences = 0
    annotated_sequences = 0
    rows: List[Dict[str, object]] = []

    with torch.no_grad():
        for seq_id, sequence in _read_fasta(input_path):
            total_sequences += 1
            hits = _annotate_sequence(sequence, model, idx2ko, thresholds, hmm_dir, device, name=seq_id)
            if hits:
                annotated_sequences += 1
                rows.extend(hit.as_row() for hit in hits)
            else:
                rows.append(
                    {
                        "name": seq_id,
                        "predict_label": pd.NA,
                        "probability": pd.NA,
                        "threshold": pd.NA,
                        "annotate": "",
                        "start": pd.NA,
                        "end": pd.NA,
                    }
                )

    df = pd.DataFrame(
        rows,
        columns=["name", "predict_label", "probability", "threshold", "annotate", "start", "end"],
    )

    if output_format == "simple":
        df.loc[df["annotate"] != "*", "predict_label"] = pd.NA
        df = df[["name", "predict_label"]]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return {
        "total": total_sequences,
        "annotated": annotated_sequences,
        "output": str(output_path),
    }
