"""Reproducible random FASTA smoke tests with synthetic model resources only."""

import random
import re

import pandas as pd
import pytest


@pytest.fixture
def random_fasta(tmp_path):
    rng = random.Random(20261116)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    # Deliberately unsorted: exercise padding across batches and the 50 aa
    # minimum for multi-domain inference, including both adjacent lengths.
    lengths = (127, 49, 1, 64, 50, 17, 129, 51, 63)
    records = [
        (f"random_{index:02d}", "".join(rng.choice(alphabet) for _ in range(length)))
        for index, length in enumerate(lengths)
    ]
    fasta = tmp_path / "random sequences.fasta"
    with fasta.open("w", encoding="utf-8") as handle:
        for name, sequence in records:
            handle.write(f">{name} synthetic random sequence\n")
            for start in range(0, len(sequence), 37):
                handle.write(sequence[start:start + 37] + "\n")
    return fasta, records


@pytest.mark.integration
@pytest.mark.parametrize("variant", ["full", "frag"])
@pytest.mark.parametrize("multi", [False, True], ids=["standard", "multi"])
def test_random_sequence_cli_preserves_predictions_with_descriptions(
    run_cli, random_fasta, tmp_path, toy_descriptions, variant, multi
):
    fasta, records = random_fasta
    args = [
        "-i", fasta, "--model", variant, "--date", "202501", "--device", "cpu",
        "--num_workers", "0", "--batch_size", "3", "--detail", "--topk", "2",
    ]
    if multi:
        # An empty directory exercises the documented no-profile path without
        # invoking HMMER or using biological profiles in compatibility CI.
        profiles = tmp_path / "empty profiles"
        profiles.mkdir()
        args += ["--multi", "--profiles_dir", profiles]

    frames = []
    summaries = []
    for description in (False, True):
        output = tmp_path / f"description-{description}.csv"
        result = run_cli(*args, "-o", output, *(["--description"] if description else []))
        assert result.returncode == 0, result.stdout + result.stderr
        summaries.append(re.findall(r"Processed (\d+) sequences, annotated (\d+)\.", result.stdout))
        frames.append(pd.read_csv(output, keep_default_na=False))

    plain, described = frames
    assert described.columns.tolist() == plain.columns.tolist() + ["description"]
    pd.testing.assert_frame_equal(described.drop(columns="description"), plain)
    expected_names = [
        name
        for name, sequence in records
        for _ in range(1 if multi and len(sequence) < 50 else 2)
    ]
    assert described["name"].tolist() == expected_names
    assert set(described["predict_label"]) - {""} == set(toy_descriptions)
    assert described["description"].tolist() == [
        toy_descriptions.get(label, "") for label in described["predict_label"]
    ]
    if multi:
        short_names = [name for name, sequence in records if len(sequence) < 50]
        assert described.loc[described["name"].isin(short_names), "predict_label"].eq("").all()

    assert summaries[0] == summaries[1]
    annotated_count = described.loc[described["annotate"] == "*", "name"].nunique()
    assert summaries[1] == [(str(len(records)), str(annotated_count))]
