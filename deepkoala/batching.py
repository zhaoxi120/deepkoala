"""Memory ceilings and bounded, length-aware protein batches.

The 80% figures are scheduling budgets, not requests to reserve memory or
process-wide allocator limits. The budget includes the model, cached allocator
blocks, inference tensors, and a caller-chosen CUDA allowance. These helpers do
not query CUDA or infer a residue budget from bytes; the caller must use
measurements of the actual model to choose that budget.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Iterator


GIB = 2 ** 30


def _integer(value, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


@dataclass(frozen=True)
class MemoryPreset:
    name: str
    capacity_gib: int

    @property
    def budget_bytes(self) -> int:
        """80% of nominal capacity, including model and runtime overhead."""
        return self.capacity_gib * GIB * 4 // 5


# The >32 GiB preset is exactly twice the 32 GiB preset. Actual hardware and
# currently free memory still cap this nominal 64 GiB profile in plan_memory.
PRESETS = {
    str(capacity): MemoryPreset(str(capacity), capacity)
    for capacity in (2, 4, 8, 16, 24, 32)
}
PRESETS["gt32"] = MemoryPreset("gt32", 64)


def select_preset(total_bytes: int, requested: str = "auto") -> MemoryPreset:
    """Choose a nominal capacity, allowing 5% device-reported shortfall.

    Devices reporting more than 32 GiB use the nominal 64 GiB profile. Smaller
    devices use the highest qualifying profile, with 2 GiB as the floor; the
    actual device ceiling is always enforced separately by plan_memory.
    """
    total_bytes = _integer(total_bytes, "total_bytes", positive=True)
    if requested != "auto":
        if not isinstance(requested, str) or requested not in PRESETS:
            raise ValueError(
                "requested preset must be 'auto', '2', '4', '8', '16', "
                "'24', '32', or 'gt32'"
            )
        return PRESETS[requested]
    if total_bytes > 32 * GIB:
        return PRESETS["gt32"]
    for capacity in (32, 24, 16, 8, 4, 2):
        if capacity * GIB * 95 <= total_bytes * 100:
            return PRESETS[str(capacity)]
    return PRESETS["2"]


@dataclass(frozen=True)
class MemoryPlan:
    preset: MemoryPreset
    budget_bytes: int
    total_bytes: int
    free_bytes: int


def plan_memory(
    total_bytes: int,
    free_bytes: int,
    reserved_bytes: int = 0,
    requested: str = "auto",
) -> MemoryPlan:
    """Calculate an absolute scheduling budget from preset and free memory.

    ``reserved_bytes`` is this process's existing allocator reservation. It is
    already included in occupied device memory, so it is added to 80% of the
    currently free bytes rather than counted a second time against that free
    amount. Total, free, and reserved values may be separate device snapshots.
    """
    total_bytes = _integer(total_bytes, "total_bytes", positive=True)
    free_bytes = _integer(free_bytes, "free_bytes")
    reserved_bytes = _integer(reserved_bytes, "reserved_bytes")
    if free_bytes > total_bytes:
        raise ValueError("free_bytes cannot exceed total_bytes")
    if reserved_bytes > total_bytes:
        raise ValueError("reserved_bytes cannot exceed total_bytes")
    preset = select_preset(total_bytes, requested)
    budget_bytes = min(
        preset.budget_bytes,
        total_bytes * 4 // 5,
        reserved_bytes + free_bytes * 4 // 5,
    )
    if budget_bytes <= 0:
        raise ValueError("No positive GPU memory budget is available")
    return MemoryPlan(preset, budget_bytes, total_bytes, free_bytes)


@dataclass(frozen=True)
class ProteinRecord:
    index: int
    name: str
    sequence: str

    @property
    def length(self) -> int:
        return len(self.sequence)


def iter_windows(
    input_path,
    window_size: int = 4096,
) -> Iterator[list[ProteinRecord]]:
    """Read FASTA once into count-bounded windows, preserving original ordinals.

    Sequence lengths do not shorten a sorting window or truncate a record.
    Host memory therefore scales with the sum of lengths in one window, while
    GPU batches are constrained separately by the adaptive predictor. After a
    yield, release the completed window before reading another one; callers
    should likewise release their references before requesting the next window.
    """
    from .data import _FastaRecords

    window_size = _integer(window_size, "window_size", positive=True)
    window = []
    for index, (name, sequence) in enumerate(_FastaRecords(input_path)):
        record = ProteinRecord(index, name, sequence)
        window.append(record)
        if len(window) == window_size:
            yield window
            window = []
    if window:
        yield window
