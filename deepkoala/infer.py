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
    top_k: int = 1,
):
    
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resources_dir = "./resources"

    db_date = find_latest_date(date, resources_dir)
    ko_cfg = Path(resources_dir)/db_date/f"ko_config_{model}.json"
    weights = Path(resources_dir)/db_date/f"weights_{model}.pt"

    ko2idx, idx2ko, threshold = load_ko_config(str(ko_cfg))
    model = GRUClassifier(128, 2, len(ko2idx)).to(device)
    checkpoint = torch.load(str(weights), map_location=device)
    model.load_state_dict(checkpoint)

    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    names, labels, probs, thrs, ann, top_preds = [], [], [], [], [], []
    total, ann_cnt = 0, 0

    model.eval()
    with torch.no_grad():
        loader = get_dataloader(input_path, batch_size, num_workers)
        for seq_names, seqs, lens in tqdm(loader, 'Inference Progress'):
            seqs, lens = seqs.to(device), lens.to(device)
            out = model(seqs, lens)
            prob = F.softmax(out, dim=1)
            k = min(top_k, prob.size(1))
            top_prob, top_idx = torch.topk(prob, k=k, dim=1)

            for seq_name, idx_row, prob_row in zip(
                seq_names, top_idx.tolist(), top_prob.tolist()
            ):
                primary_idx = idx_row[0]
                primary_prob = prob_row[0]
                names.append(seq_name)
                pred = idx2ko[primary_idx]
                labels.append(pred)
                probs.append(primary_prob)
                th = threshold[pred]
                thrs.append(th)
                mark = '*' if primary_prob >= th else ''
                ann.append(mark)
                total += 1
                ann_cnt += 1 if mark else 0


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
    return {"total": total, "annotated": ann_cnt, "output": str(output_path)}
