"""Input contracts tested with synthetic sequences and no model resources."""

import pytest
import torch

from deepkoala.data import ProteinDataset, collate_fn, get_dataloader
from deepkoala.batching import iter_windows
from deepkoala.infer_multi import _read_fasta


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize("final_newline", [False, True])
def test_fasta_reads_multiline_records_in_paths_with_spaces_and_unicode(
    tmp_path, newline, final_newline
):
    directory = tmp_path / "protein inputs 蛋白"
    directory.mkdir()
    fasta = directory / "short sequences.fasta"
    content = newline.join([">first description", "AC", "D", ">second", "WY"])
    if final_newline:
        content += newline
    fasta.write_bytes(content.encode("utf-8"))

    dataset = ProteinDataset(str(fasta))

    assert len(dataset) == 2
    first_name, first_tokens, first_end = dataset[0]
    second_name, second_tokens, second_end = dataset[1]
    assert (first_name, second_name) == ("first", "second")
    assert first_tokens.tolist() == [2, 3, 4]
    assert second_tokens.tolist() == [20, 21]
    # The model consumes the last residue index, not the number of residues.
    assert (first_end, second_end) == (2, 1)
    assert first_tokens.dtype == torch.long
    assert list(_read_fasta(str(fasta))) == [("first", "ACD"), ("second", "WY")]


def test_unknown_residues_use_unknown_token_without_becoming_padding(tmp_path):
    fasta = tmp_path / "unknown.fasta"
    fasta.write_text(">sequence\nACDX?\n", encoding="utf-8")

    _, tokens, last_index = ProteinDataset(str(fasta))[0]

    assert tokens.tolist() == [2, 3, 4, 1, 1]
    assert last_index == 4


def test_collate_pads_short_sequences_and_preserves_last_residue_positions(tmp_path):
    fasta = tmp_path / "lengths.fasta"
    fasta.write_text(">long\nACD\n>short\nW\n", encoding="utf-8")
    dataset = ProteinDataset(str(fasta))

    names, tokens, last_indices = collate_fn([dataset[0], dataset[1]])

    assert names == ("long", "short")
    assert tokens.tolist() == [[2, 3, 4], [20, 0, 0]]
    assert last_indices.dtype == torch.long
    assert tokens[torch.arange(2), last_indices].tolist() == [4, 20]


def test_dataloader_keeps_record_order_and_the_final_partial_batch(tmp_path):
    fasta = tmp_path / "batches.fasta"
    fasta.write_text(">one\nAC\n>two\nD\n>three\nWY\n", encoding="utf-8")

    batches = list(get_dataloader(str(fasta), batch_size=2, num_workers=0))

    assert [tuple(names) for names, _, _ in batches] == [("one", "two"), ("three",)]
    assert [tokens.shape[0] for _, tokens, _ in batches] == [2, 1]
    assert batches[-1][1].tolist() == [[20, 21]]


def test_empty_fasta_yields_no_records(tmp_path):
    fasta = tmp_path / "empty.fasta"
    fasta.write_text("", encoding="utf-8")

    assert len(ProteinDataset(str(fasta))) == 0
    assert list(_read_fasta(str(fasta))) == []
    assert list(get_dataloader(str(fasta), batch_size=2, num_workers=0)) == []


def test_missing_fasta_reports_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        ProteinDataset(str(tmp_path / "missing.fasta"))


@pytest.mark.parametrize("header", [
    "sequence\tdescription", "sequence\t\tdescription", "sequence description",
    " \tsequence \t description", "sequence",
], ids=["tab", "tabs", "space", "mixed-whitespace", "id-only"])
def test_all_fasta_readers_use_the_same_whitespace_delimited_id(tmp_path, header):
    fasta = tmp_path / "tab_header.fasta"
    fasta.write_text(f">{header}\nACD\n", encoding="utf-8")

    single_model_name = ProteinDataset(str(fasta))[0][0]
    multi_model_name = next(_read_fasta(str(fasta)))[0]
    streamed_names, _, _ = next(iter(get_dataloader(str(fasta), batch_size=1, num_workers=0)))
    window = next(iter_windows(fasta))

    assert single_model_name == multi_model_name == streamed_names[0] == window[0].name == "sequence"


@pytest.mark.parametrize("header", ["", " \t "])
def test_all_fasta_readers_reject_empty_identifiers(tmp_path, header):
    fasta = tmp_path / "empty_header.fasta"
    fasta.write_text(f">{header}\nACD\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Empty FASTA header"):
        ProteinDataset(str(fasta))[0]
    with pytest.raises(ValueError, match="Empty FASTA header"):
        list(_read_fasta(str(fasta)))
    with pytest.raises(ValueError, match="Empty FASTA header"):
        list(get_dataloader(str(fasta), batch_size=1, num_workers=0))
    with pytest.raises(ValueError, match="Empty FASTA header"):
        list(iter_windows(fasta))
