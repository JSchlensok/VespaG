import warnings
from pathlib import Path

import polars as pl
import typer
import yaml
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm.rich import tqdm
from typing import Annotated

from vespag.predict import generate_predictions
from vespag.utils import download, setup_logger, unzip
from vespag.utils.proteingym import PROTEINGYM_CHANGED_FILENAMES

app = typer.Typer()


@app.command()
def proteingym(
    output_path: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output",
            help="Output path. Defaults to ./output/proteingym217 or ./output/proteingym87",
        ),
    ],
    dms_reference_file: Annotated[
        Path, typer.Option("--reference-file", help="Path of DMS reference file")
    ] = None,
    dms_directory: Annotated[
        Path,
        typer.Option(
            "--dms-directory", help="Path of directory containing per-DMS score files"
        ),
    ] = None,
    embedding_file: Annotated[
        Path,
        typer.Option(
            "-e",
            "--embeddings",
            help="Path to pre-generated input embeddings. Embeddings will be generated from scratch if no path is provided",
        ),
    ] = None,
    id_map_file: Annotated[
        Path,
        typer.Option(
            "--id-map",
            help="CSV file mapping embedding IDs to FASTA IDs if they're different",
        ),
    ] = None,
    normalize_scores: Annotated[
        bool,
        typer.Option(
            "--normalize/--dont-normalize", help="Whether to normalize scores to [0, 1]"
        ),
    ] = True,
    legacy_mode: Annotated[
        bool,
        typer.Option(
            "--v1/--v2",
            help="Whether to evaluate on the first version (87 DMS) of ProteinGym",
        ),
    ] = False,
):
    logger = setup_logger()
    warnings.filterwarnings("ignore", message="rich is experimental/alpha")

    benchmark_name, benchmark_version = (
        ("proteingym217", "v2") if not legacy_mode else ("proteingym87", "v1")
    )
    config = yaml.safe_load((Path.cwd() / "params.yaml").open("r"))["eval"][
        "proteingym"
    ]

    if not dms_reference_file:
        dms_reference_file = Path.cwd() / f"data/test/{benchmark_name}/reference.csv"
        download(
            config["reference_file"][benchmark_version],
            dms_reference_file,
            "Downloading reference file",
            remove_bar=True,
        )

    if not dms_directory:
        dms_directory = Path.cwd() / f"data/test/{benchmark_name}/raw_dms_files/"
        zip_file = dms_directory / "DMS.zip"
        download(
            config["dms_files"], zip_file, "Downloading DMS files", remove_bar=True
        )
        unzip(zip_file, dms_directory, "Extracting DMS files", remove_bar=True)
        zip_file.unlink()

    if not output_path:
        output_path = Path.cwd() / f"output/{benchmark_name}"
    output_path.mkdir(parents=True, exist_ok=True)

    sequence_file = output_path / "sequences.fasta"
    reference_df = pl.read_csv(dms_reference_file)
    if legacy_mode:
        new_filenames = pl.from_records(
            [
                {"DMS_id": key, "DMS_filename": val}
                for key, val in PROTEINGYM_CHANGED_FILENAMES.items()
            ]
        )
        reference_df = (
            reference_df.join(new_filenames, on="DMS_id", how="left")
            .with_columns(
                pl.col("DMS_filename_right").fill_null(pl.col("DMS_filename"))
            )
            .drop("DMS_filename")
            .rename({"DMS_filename_right": "DMS_filename"})
        )
    sequences = [
        SeqRecord(id=row["DMS_id"], seq=Seq(row["target_seq"]))
        for row in reference_df.iter_rows(named=True)
    ]
    logger.info(f"Writing {len(sequences)} sequences to {sequence_file}")
    SeqIO.write(sequences, sequence_file, "fasta")

    logger.info(f"Parsing mutation files from {dms_directory}")
    mutation_file = output_path / "mutations.txt"
    dms_files = {
        row["DMS_id"]: pl.read_csv(dms_directory / row["DMS_filename"])
        for row in reference_df.iter_rows(named=True)
    }
    pl.concat(
        [
            df.with_columns(pl.lit(dms_id).alias("DMS_id")).select(["DMS_id", "mutant"])
            for dms_id, df in dms_files.items()
        ]
    ).write_csv(mutation_file)

    logger.info("Generating predictions")
    generate_predictions(
        fasta_file=sequence_file,
        output_path=output_path,
        embedding_file=embedding_file,
        mutation_file=mutation_file,
        id_map_file=id_map_file,
        single_csv=True,
        normalize_scores=normalize_scores,
    )

    mutation_file.unlink()
    sequence_file.unlink()

    prediction_file = output_path / "vespag_scores_all.csv"
    all_preds = pl.read_csv(prediction_file)

    logger.info(
        "Computing Spearman correlations between experimental and predicted scores"
    )
    records = []
    for dms_id, dms_df in tqdm(list(dms_files.items()), leave=False):
        dms_df = dms_df.join(
            all_preds.filter(pl.col("Protein") == dms_id),
            left_on="mutant",
            right_on="Mutation",
        )
        spearman = dms_df.select(
            pl.corr("DMS_score", "VespaG", method="spearman")
        ).item()
        records.append({"DMS_id": dms_id, "spearman": spearman})
    result_csv_path = output_path / "VespaG_Spearman_per_DMS.csv"
    result_df = pl.from_records(records)
    logger.info(f"Writing results to {result_csv_path}")
    logger.info(f"Mean Spearman r: {result_df['spearman'].mean():.3f}")
    result_df.write_csv(result_csv_path)
