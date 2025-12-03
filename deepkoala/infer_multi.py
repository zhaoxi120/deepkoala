from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
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


@dataclass
class DomainHit:
    """Container describing a detected domain on the original sequence."""

    name: str | None
    predict_label: str
    probability: float
    threshold: float
    start: Optional[int]
    end: Optional[int]

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
            "start": self.start if self.start is not None else pd.NA,
            "end": self.end if self.end is not None else pd.NA,
        }


def _tokenize(seq: str) -> torch.Tensor:
    """Convert amino acid sequence into indices for the classifier."""

    tokens = [AA_VOCAB.get(aa, 1) for aa in seq]
    return torch.tensor(tokens, dtype=torch.long)


def _classify(
    model: GRUClassifier, seq: str, device: torch.device, top_k: int
) -> Tuple[List[int], List[float]]:
    """Return ``(indices, probabilities)`` for the top-k predictions of a fragment."""

    tensor = _tokenize(seq).unsqueeze(0).to(device)
    lens = torch.tensor([tensor.size(1) - 1], dtype=torch.long, device=device)
    logits = model(tensor, lens)
    prob = F.softmax(logits, dim=1)[0]
    k = min(top_k, prob.size(0))
    top_prob, top_idx = torch.topk(prob, k=k)
    return top_idx.tolist(), top_prob.tolist()


def _run_hmmsearch(hmm_file: Path, seq: str) -> Tuple[int | None, int | None]:
    """Execute ``hmmsearch`` on ``seq`` against ``hmm_file`` and parse bounds."""

    with tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False) as tmp_fa:
        tmp_fa.write(">query\n")
        tmp_fa.write(seq + "\n")
        fasta_path = tmp_fa.name
    with tempfile.NamedTemporaryFile("r") as domtbl:
        cmd = [
            f"hmmsearch --noali --domtblout {domtbl.name} {hmm_file} {fasta_path}"
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            print("hmmsearch not found; skipping boundary detection.")
            return None, None
        except subprocess.CalledProcessError as exc:
            if exc.returncode == 1:
                return None, None
            raise
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
    model: str, date: str, device: torch.device
) -> Tuple[GRUClassifier, Dict[str, int], Dict[int, str], Dict[str, float]]:
    resources_dir = Path(__file__).resolve().parent.parent / "resources"
    db_date = find_latest_date(date, str(resources_dir))
    cfg_path = resources_dir / db_date / f"ko_config_{model}.json"
    weight_path = resources_dir / db_date / f"weights_{model}.pt"

    ko2idx, idx2ko, thresholds = load_ko_config(str(cfg_path))
    classifier = GRUClassifier(128, 2, len(ko2idx)).to(device)
    checkpoint = torch.load(str(weight_path), map_location=device)
    classifier.load_state_dict(checkpoint)
    classifier.eval()
    return classifier, ko2idx, idx2ko, thresholds


def _annotate_sequence(
    sequence: str,
    model: GRUClassifier,
    idx2ko: Dict[int, str],
    thresholds: Dict[str, float],
    hmm_dir: Path,
    device: torch.device,
    *,
    name: str | None = None,
    top_k: int = 1,
) -> List[DomainHit]:
    hits: List[DomainHit] = []
    queue: List[Tuple[str, int]] = [(sequence, 0)]

    while queue:
        frag, offset = queue.pop(0)
        if len(frag) < 50:
            continue
        pred_indices, probs = _classify(model, frag, device, top_k=top_k)
        if isinstance(pred_indices, int):
            pred_indices = [pred_indices]
        if isinstance(probs, (int, float)):
            probs = [float(probs)]
        for rank, (pred_idx, prob) in enumerate(zip(pred_indices, probs)):
            ko = idx2ko[pred_idx]
            thr = thresholds[ko]
            if rank == 0:
                if prob < thr:
                    hits.append(
                        DomainHit(
                            name=name,
                            predict_label=ko,
                            probability=prob,
                            threshold=thr,
                            start=None,
                            end=None,
                        )
                    )
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
            else:
                hits.append(
                    DomainHit(
                        name=name,
                        predict_label=ko,
                        probability=prob,
                        threshold=thr,
                        start=None,
                        end=None,
                    )
                )
    return hits


def annotate_precision(
    sequence: str,
    *,
    hmmer_dir: str | None = None,
    date: str = "latest",
    model: str = "full",
    device: torch.device | None = None,
) -> List[Dict[str, float]]:
    """Annotate domains in ``sequence`` using multi-domain mode."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not hmmer_dir:
        raise ValueError("hmmer_dir must be provided for multi-domain annotation")
    classifier, _, idx2ko, thresholds = _load_model(model, date, device)
    hmm_dir = Path(hmmer_dir)

    with torch.no_grad():
        hits = _annotate_sequence(sequence, classifier, idx2ko, thresholds, hmm_dir, device)

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
    model: str = "full",
    date: str = "latest",
    profiles_dir: str | None = None,
    detail: bool = False,
    device: torch.device | None = None,
    top_k: int = 1,
) -> Dict[str, object]:
    """Run multi-domain inference on ``input_path`` and write CSV to ``output_path``."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not profiles_dir:
        raise ValueError("profiles_dir must be provided when running multi-domain inference")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    classifier, _, idx2ko, thresholds = _load_model(model, date, device)
    hmm_dir = Path(profiles_dir)

    total_sequences = 0
    annotated_sequences = 0
    rows: List[Dict[str, object]] = []

    with torch.no_grad():
        for seq_id, sequence in _read_fasta(input_path):
            total_sequences += 1
            hits = _annotate_sequence(
                sequence,
                classifier,
                idx2ko,
                thresholds,
                hmm_dir,
                device,
                name=seq_id,
                top_k=top_k,
            )
            if hits:
                if any(hit.annotate == "*" for hit in hits):
                    annotated_sequences += 1
                rows.extend(hit.as_row() for hit in hits)
            else:
                rows.append(
                    {
                        "name": seq_id,
                        "predict_label": pd.NA,
                        "probability": pd.NA,
                        "threshold": pd.NA,
                        "start": pd.NA,
                        "end": pd.NA,
                        "annotate": "",
                    }
                )

    df = pd.DataFrame(
        rows,
        columns=["name", "predict_label", "probability", "threshold", "start", "end", "annotate"],
    )
    df = df.round(4)

    if not detail:
        df.loc[df["annotate"] != "*", "predict_label"] = pd.NA
        df = df[["name", "predict_label"]]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return {
        "total": total_sequences,
        "annotated": annotated_sequences,
        "output": str(output_path),
    }
