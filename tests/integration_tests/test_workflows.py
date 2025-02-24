import polars as pl
import polars.testing
from typer.testing import CliRunner

from vespag.__main__ import app

runner = CliRunner()


def test_predict(output_dir, fasta, score_files):
    result = runner.invoke(app, ["predict", "-i", str(fasta), "-o", str(output_dir)])

    assert result.exit_code == 0

    for score_file in score_files:
        assert score_file.exists()

        ground_truth_df = pl.read_csv(score_file)
        generated_df = pl.read_csv(output_dir / score_file.name)
        pl.testing.assert_frame_equal(generated_df, ground_truth_df)
