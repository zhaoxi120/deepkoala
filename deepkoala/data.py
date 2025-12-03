from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

AA_VOCAB = {
    '<pad>': 0,
    '<unk>': 1,
    'A': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7,
    'H': 8,
    'I': 9,
    'K': 10,
    'L': 11,
    'M': 12,
    'N': 13,
    'P': 14,
    'Q': 15,
    'R': 16,
    'S': 17,
    'T': 18,
    'V': 19,
    'W': 20,
    'Y': 21,
}


class ProteinDataset(Dataset):
    def __init__(self, fasta_path: str):
        self.fasta_path = fasta_path
        self.entries = self._load()

    def _load(self):
        with open(self.fasta_path, 'r') as f:
            content = f.read()[1:]
        return [e for e in content.split('\n>') if e.strip()]

    def tokenize(self, seq: str):
        return [AA_VOCAB.get(a, 1) for a in seq]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        header, seq = self.entries[idx].split('\n', 1)
        name = header.split(' ')[0]
        seq = self.tokenize(seq.replace('\n', ''))
        t = torch.tensor(seq, dtype=torch.long)
        return name, t, len(t) - 1


def collate_fn(batch):
    names, seqs, lens = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    return names, padded, torch.tensor(lens, dtype=torch.long)


def get_dataloader(path: str, batch_size: int, num_workers: int = 0):
    return DataLoader(
        ProteinDataset(path),
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=num_workers,
    )
