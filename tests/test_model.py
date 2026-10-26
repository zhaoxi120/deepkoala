"""CPU numerical checks using deterministic, nonzero synthetic weights."""

import pytest
import torch

from deepkoala.model import GRUClassifier


@pytest.mark.parametrize("length", [1, 3, 5])
def test_padded_batch_matches_individual_sequence(length):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        model = GRUClassifier(16, 2, 3).eval()
    sequence = torch.tensor([2, 3, 4, 5, 6][:length], dtype=torch.long)
    padded = torch.zeros((2, 5), dtype=torch.long)
    padded[0] = torch.tensor([7, 8, 9, 10, 11])
    padded[1, :length] = sequence
    with torch.no_grad():
        individual = model(sequence.unsqueeze(0), torch.tensor([length - 1]))
        batched = model(padded, torch.tensor([4, length - 1]))
    assert batched.shape == (2, 3)
    assert torch.isfinite(batched).all()
    torch.testing.assert_close(batched[1], individual[0], rtol=1e-5, atol=1e-6)
