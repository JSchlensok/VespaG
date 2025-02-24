import sys
from pathlib import Path

import pytest
from Bio import SeqIO

test_data_dir = Path("tests/test_data")

@pytest.fixture()
def fasta() -> Path:
    return test_data_dir / "test.fasta"

@pytest.fixture()
def esm2_score_files(fasta) -> list[Path]:
    return [test_data_dir / f"scores_esm2/{rec.id}.csv" for rec in SeqIO.parse(str(fasta), "fasta")]

@pytest.fixture()
def prott5_score_files(fasta) -> list[Path]:
    return [test_data_dir / f"scores_prott5/{rec.id}.csv" for rec in SeqIO.parse(str(fasta), "fasta")]


@pytest.fixture()
def output_dir(tmp_path) -> Path:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
