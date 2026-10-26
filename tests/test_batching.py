"""GPU-independent contracts for memory presets and bounded protein batching."""

import math
import weakref

import pytest

from deepkoala.batching import (
    GIB,
    PRESETS,
    iter_windows,
    plan_memory,
    select_preset,
)


@pytest.mark.parametrize("name,capacity", [(str(n), n) for n in (2, 4, 8, 16, 24, 32)] + [("gt32", 64)])
def test_seven_presets_have_exact_eighty_percent_ceilings(name, capacity):
    preset = PRESETS[name]
    assert len(PRESETS) == 7
    assert preset.name == name
    assert preset.capacity_gib == capacity
    assert preset.budget_bytes == int(capacity * 0.8 * GIB)
    assert select_preset(8 * GIB, requested=name) == preset


@pytest.mark.parametrize("capacity", [2, 4, 8, 16, 24, 32])
def test_auto_detects_nominal_capacity_and_five_percent_shortfall(capacity):
    assert select_preset(capacity * GIB).name == str(capacity)
    boundary = math.ceil(capacity * GIB * 95 / 100)
    assert select_preset(boundary).name == str(capacity)
    if capacity > 2:
        assert select_preset(boundary - 1).capacity_gib < capacity


def test_auto_capacity_floor_and_large_device_boundary():
    assert select_preset(GIB).name == "2"
    assert select_preset(32 * GIB + 1).name == "gt32"


@pytest.mark.parametrize("capacity,ceiling", [(40, 32), (48, 38.4), (80, 51.2)])
def test_large_preset_is_clamped_by_actual_capacity(capacity, ceiling):
    plan = plan_memory(capacity * GIB, capacity * GIB)
    assert plan.preset.name == "gt32"
    assert plan.budget_bytes == int(ceiling * GIB)
    assert plan.total_bytes == plan.free_bytes == capacity * GIB


def test_large_preset_nominal_budget_is_twice_32_gib_before_byte_rounding():
    assert PRESETS["gt32"].capacity_gib == 2 * PRESETS["32"].capacity_gib
    assert PRESETS["gt32"].budget_bytes - 2 * PRESETS["32"].budget_bytes in (0, 1)


def test_busy_gpu_uses_free_memory_plus_existing_process_reservation():
    plan = plan_memory(24 * GIB, 5 * GIB, reserved_bytes=GIB)
    assert plan.preset.name == "24"
    assert plan.budget_bytes == 5 * GIB


def test_explicit_preset_still_respects_free_and_actual_device_limits():
    assert plan_memory(4 * GIB, 4 * GIB, requested="gt32").budget_bytes == int(3.2 * GIB)
    assert plan_memory(24 * GIB, GIB, requested="32").budget_bytes == int(0.8 * GIB)
    assert plan_memory(24 * GIB, 24 * GIB, requested="2").budget_bytes == int(1.6 * GIB)


def test_zero_free_memory_can_reuse_existing_allocator_reservation():
    assert plan_memory(8 * GIB, 0, reserved_bytes=GIB).budget_bytes == GIB


@pytest.mark.parametrize("total", [0, -1, True, 2.0, "8", None])
def test_invalid_total_capacity_is_rejected(total):
    with pytest.raises(ValueError, match="total_bytes"):
        select_preset(total)


@pytest.mark.parametrize("requested", ["64", "AUTO", "", None, 8, []])
def test_invalid_preset_is_rejected(requested):
    with pytest.raises(ValueError, match="requested preset"):
        select_preset(8 * GIB, requested)


@pytest.mark.parametrize("kwargs,match", [
    ({"total_bytes": -1}, "total_bytes"),
    ({"free_bytes": -1}, "free_bytes"),
    ({"free_bytes": 9 * GIB}, "free_bytes"),
    ({"free_bytes": 1.5}, "free_bytes"),
    ({"reserved_bytes": -1}, "reserved_bytes"),
    ({"reserved_bytes": 9 * GIB}, "reserved_bytes"),
    ({"reserved_bytes": True}, "reserved_bytes"),
    ({"free_bytes": 0}, "No positive GPU memory budget"),
])
def test_invalid_memory_plan_is_rejected(kwargs, match):
    arguments = {"total_bytes": 8 * GIB, "free_bytes": 8 * GIB}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        plan_memory(**arguments)


def _write_fasta(tmp_path, lengths):
    path = tmp_path / "proteins.fasta"
    path.write_text("".join(f">duplicate description {i}\n{'A' * length}\n" for i, length in enumerate(lengths)))
    return path


def test_default_window_size_and_duplicate_names_preserve_global_ordinals(tmp_path):
    path = _write_fasta(tmp_path, [1] * 4099)
    windows = list(iter_windows(path))
    assert [len(window) for window in windows] == [4096, 3]
    flattened = [record for window in windows for record in window]
    assert [record.index for record in flattened] == list(range(4099))
    assert {record.name for record in flattened} == {"duplicate"}


def test_windows_use_record_count_even_above_four_million_residues(tmp_path):
    path = _write_fasta(tmp_path, [16_384] * 257)
    windows = list(iter_windows(path, window_size=256))
    assert [len(window) for window in windows] == [256, 1]
    # An implicit four-million-residue cap would split the first window early.
    assert sum(record.length for record in windows[0]) == 4_194_304
    flattened = [record for window in windows for record in window]
    assert [record.index for record in flattened] == list(range(257))
    assert all(record.sequence == "A" * 16_384 for record in flattened)


def test_windows_release_completed_records_before_reading_next_window(monkeypatch):
    from deepkoala import data

    calls = []
    first_record = None

    class CountingRecords:
        def __init__(self, path):
            calls.append(path)

        def __iter__(self):
            for index in range(4):
                if index == 2:
                    assert first_record() is None
                calls.append(index)
                yield "duplicate", "AA"

    monkeypatch.setattr(data, "_FastaRecords", CountingRecords)
    windows = iter_windows("one-pass", window_size=2)
    first_window = next(windows)
    first_record = weakref.ref(first_window[0])
    assert calls == ["one-pass", 0, 1]
    del first_window
    assert [record.index for record in next(windows)] == [2, 3]
    assert list(windows) == []
    assert calls == ["one-pass", 0, 1, 2, 3]


def test_windows_normalize_ids_and_preserve_sequence_tokenization(tmp_path):
    path = tmp_path / "multiline.fasta"
    path.write_bytes(b"\r\n>name\tdescription more\r\nAC\r\nD X\r\n>name description\r\nW")
    records = list(iter_windows(path))[0]
    assert [(r.name, r.sequence) for r in records] == [("name", "ACD X"), ("name", "W")]


@pytest.mark.parametrize("content,match", [("AA\n>name\nAA\n", "before a header"), (">\nAA\n", "Empty FASTA header"), (">name\n", "Empty FASTA sequence")])
def test_windows_preserve_parser_validation(tmp_path, content, match):
    path = tmp_path / "invalid.fasta"
    path.write_text(content)
    with pytest.raises(ValueError, match=match):
        list(iter_windows(path))


def test_empty_fasta_has_no_windows(tmp_path):
    path = _write_fasta(tmp_path, [])
    assert list(iter_windows(path)) == []


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_window_size_is_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="window_size"):
        list(iter_windows(tmp_path / "unused.fasta", window_size=value))
