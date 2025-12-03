from pathlib import Path
import torch, torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from .data import get_dataloader
from .model import GRUClassifier
from .utils import load_ko_config, find_latest_date

def inference(
    input_path: str,
    output_path: str,
    model: str = "full",
    date: str = "latest",
    batch_size: int = 64,
    num_workers: int = 2,
    detail: bool = False,
    device: torch.device | None = None,
    topk: int = 1,
):
    
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resources_dir = Path(__file__).resolve().parent.parent / "resources"

    db_date = find_latest_date(date, resources_dir)
    ko_cfg = resources_dir / db_date / f"ko_config_{model}.json"
    weights = resources_dir / db_date / f"weights_{model}.pt"

    ko2idx, idx2ko, threshold = load_ko_config(str(ko_cfg))
    model = GRUClassifier(128, 2, len(ko2idx)).to(device)
    checkpoint = torch.load(str(weights), map_location=device)
    model.load_state_dict(checkpoint)

    if topk < 1:
        raise ValueError("topk must be >= 1")

    names, labels, probs, thrs, ann = [], [], [], [], []
    total_sequences, annotated_sequences = 0, 0

    model.eval()
    with torch.no_grad():
        loader = get_dataloader(input_path, batch_size, num_workers)
        for seq_names, seqs, lens in tqdm(loader, 'Inference Progress'):
            seqs, lens = seqs.to(device), lens.to(device)
            out = model(seqs, lens)
            prob = F.softmax(out, dim=1)
            k = min(topk, prob.size(1))
            top_prob, top_idx = torch.topk(prob, k=k, dim=1)

            for seq_name, idx_row, prob_row in zip(
                seq_names, top_idx.tolist(), top_prob.tolist()
            ):
                total_sequences += 1
                seq_annotated = False

                for pred_idx, pred_prob in zip(idx_row, prob_row):
                    pred = idx2ko[pred_idx]
                    th = threshold[pred]
                    mark = '*' if pred_prob >= th else ''

                    names.append(seq_name)
                    labels.append(pred)
                    probs.append(pred_prob)
                    thrs.append(th)
                    ann.append(mark)

                    if mark:
                        seq_annotated = True

                if seq_annotated:
                    annotated_sequences += 1


    df = pd.DataFrame(
        {
            "name": names,
            "predict_label": labels,
            "probability": probs,
            "threshold": thrs,
            "annotate": ann,
        }
    )
    df = df.round(4)
    
    if not detail:
        df.loc[df["annotate"] != "*", "predict_label"] = pd.NA
        df = df[["name","predict_label"]]
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return {
        "total": total_sequences,
        "annotated": annotated_sequences,
        "output": str(output_path),
    }
