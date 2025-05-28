import polars as pl
import polars.testing
import pytest
from typer.testing import CliRunner

from vespag.__main__ import app

runner = CliRunner()


@pytest.mark.parametrize("embedding_type", ["esm2", "prott5"])
def test_predict(output_dir, fasta, score_files, embedding_type):
    result = runner.invoke(app, ["predict", "-i", str(fasta), "-o", str(output_dir), "--embedding-type", embedding_type])

    assert result.exit_code == 0

    for score_file in score_files[embedding_type]:
        assert score_file.exists()

        ground_truth_df = pl.read_csv(score_file)
        generated_df = pl.read_csv(output_dir / score_file.name)
        pl.testing.assert_frame_equal(generated_df, ground_truth_df, rtol=3e-05, atol=2e-07)