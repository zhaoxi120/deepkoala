import copy

import pytest
import torch

from deepkoala.model import GRUClassifier


@pytest.mark.hardware
def test_real_accelerator_gru_matches_cpu(hardware_device, record_testsuite_property):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        cpu = GRUClassifier(16, 2, 3).eval()
    accelerator = copy.deepcopy(cpu).to(hardware_device).eval()
    tokens = torch.tensor([[2, 3, 4, 5, 6], [21, 1, 0, 0, 0]], dtype=torch.long)
    indices = torch.tensor([4, 1], dtype=torch.long)
    with torch.no_grad():
        expected = cpu(tokens, indices)
        actual = accelerator(tokens.to(hardware_device), indices.to(hardware_device))
    assert actual.device.type == hardware_device.type
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
    assert actual.argmax(dim=1).cpu().tolist() == expected.argmax(dim=1).tolist()
    record_testsuite_property("device", str(hardware_device))
    if hardware_device.type == "cuda":
        record_testsuite_property("device_name", torch.cuda.get_device_name(hardware_device))
