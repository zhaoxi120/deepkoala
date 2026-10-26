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


def _fasta_identifier(header: str) -> str:
    """Use the first whitespace-delimited token as the ID in every reader."""
    fields = header.split(maxsplit=1)
    if not fields:
        raise ValueError("Empty FASTA header")
    return fields[0]


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
        name = _fasta_identifier(header)
        seq = self.tokenize(seq.replace('\n', ''))
        t = torch.tensor(seq, dtype=torch.long)
        return name, t, len(t) - 1


def collate_fn(batch):
    names, seqs, lens = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    return names, padded, torch.tensor(lens, dtype=torch.long)


class _FastaRecords:
    """Read records once, in the parent process, as DataLoader sampler keys.

    Sending records rather than integer indices lets workers tokenize batches
    without loading or independently scanning the entire input. DataLoader's
    bounded prefetch queue preserves the original batch and record order.
    """

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with open(self.path, "r") as handle:
            header = None
            chunks = []
            for line_number, line in enumerate(handle, 1):
                line = line.rstrip("\r\n")
                if line.startswith(">"):
                    if header is not None:
                        yield self._record(header, chunks)
                    header, chunks = line[1:], []
                elif header is None:
                    if line.strip():
                        raise ValueError(
                            f"FASTA sequence before a header at line {line_number}"
                        )
                else:
                    chunks.append(line)
            if header is not None:
                yield self._record(header, chunks)

    @staticmethod
    def _record(header, chunks):
        name = _fasta_identifier(header)
        sequence = "".join(chunks)
        if not sequence:
            raise ValueError(f"Empty FASTA sequence for {name!r}")
        return name, sequence


class _ProteinRecords(Dataset):
    """Tokenize the records supplied by the streaming sampler."""

    def __getitem__(self, record):
        name, sequence = record
        tokens = torch.tensor([AA_VOCAB.get(aa, 1) for aa in sequence], dtype=torch.long)
        return name, tokens, len(tokens) - 1


def get_dataloader(path: str, batch_size: int, num_workers: int = 0):
    """Stream fixed-size batches with bounded worker prefetching.

    Unlike the eager, indexable ProteinDataset API, this loader does not know
    the total number of records without reading the file.
    """
    return DataLoader(
        _ProteinRecords(),
        sampler=_FastaRecords(path),
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=num_workers,
    )
