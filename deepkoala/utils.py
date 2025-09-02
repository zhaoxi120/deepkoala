import os, re, sys, json
from typing import Dict, Tuple

def load_ko_config(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        classes = json.load(f)
    ko2idx = {ko: v["index"] for ko, v in classes.items()}
    idx2ko = {v["index"]: ko for ko, v in classes.items()}
    thresholds = {ko: v["threshold"] for ko, v in classes.items()}
    return ko2idx, idx2ko, thresholds


def find_latest_date(date_str: str, folder_path: str) -> str:
    pat = re.compile(r'^\d{6}$')
    if date_str == "latest":
        cands = [n for n in os.listdir(folder_path) if pat.match(n) and os.path.isdir(os.path.join(folder_path, n))]
        if not cands: raise DateResolveError("No valid YYYYMM directories.")
        return max(cands)
    if pat.match(date_str):
        if os.path.isdir(os.path.join(folder_path, date_str)): return date_str
        raise DateResolveError(f"Specified date '{date_str}' not found.")
    raise DateResolveError("Invalid date format (need 'YYYYMM' or 'latest').")