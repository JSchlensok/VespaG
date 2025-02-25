import sys
from pathlib import Path

import numpy as np
import pytest
from Bio import SeqIO
from jaxtyping import Float

test_data_dir = Path("tests/test_data")

ScoreList = Float[np.typing.NDArray, "100"]

@pytest.fixture(scope="module")
def fasta() -> Path:
    return test_data_dir / "test.fasta"

@pytest.fixture(scope="module")
def score_files(fasta) -> dict[str, list[Path]]:
    return {
        "esm2": [test_data_dir / f"scores_esm2/{rec.id}.csv" for rec in SeqIO.parse(str(fasta), "fasta")],
        "prott5": [test_data_dir / f"scores_prott5/{rec.id}.csv" for rec in SeqIO.parse(str(fasta), "fasta")]
    }

@pytest.fixture()
def output_dir(tmp_path) -> Path:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
