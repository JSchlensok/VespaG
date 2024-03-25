import csv
import shutil
import statistics
import subprocess
from pathlib import Path

import h5py
import polars as pl
import rich.progress as progress
import torch
import typer
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TextColumn, track
from typing_extensions import Annotated

from src.vespa2.utils import setup_logger


def main(
    gemme_dir: Annotated[Path, typer.Argument()],
    output_dir: Annotated[Path, typer.Argument()],
    output_h5_path: Annotated[Path, typer.Option("--h5-path")] = None,
    unprocessed_fasta: Annotated[Path, typer.Option("--fasta")] = None,
):
    logger = setup_logger()
    output_dir.mkdir(exist_ok=True, parents=True)
    gemme_output_dir = output_dir / "raw_gemme_predictions"
    gemme_output_dir.mkdir(exist_ok=True)

    output_h5_path = output_h5_path or output_dir / "gemme_predictions.h5"

    # Extract files if necessary
    if gemme_dir.suffix == ".tar.gz":
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn()
        ) as progress:

                task = progress.add_task(f"Extracting {gemme_prediction_archive} to {output_dir}", total=None)
                subprocess.run(["tar", "-xzvf", str(gemme_prediction_archive), "--strip-components=1", "-C", str(gemme_output_dir)], capture_output=True)
    else:
        shutil.copytree(gemme_dir, gemme_output_dir, dirs_exist_ok=True)
    
    # Ensure folder structure
    single_files = gemme_output_dir.glob("*_normPred_evolCombi.txt")
    for file in track(list(single_files), description="Ensuring consistent folder structure"):
        protein_name = str(file.stem).replace("_normPred_evolCombi", "")
        new_directory = gemme_output_dir / protein_name
        new_directory.mkdir(exist_ok=True)
        file.rename(new_directory / file.name)

    # Filter out faulty GEMME preds
    bad_execution_file = gemme_output_dir / "bad_jet.txt"
    if not bad_execution_file.exists():
        bad_execution_file = gemme_output_dir / "bad_execution.txt"
        if not bad_execution_file.exists():
            # manually re-run quality control script
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn()
            ) as progress:

                task = progress.add_task(f"Checking GEMME prediction quality", total=None)
                subprocess.run(["src/vespa2/data/gemme/check_gemme_quality.sh", str(gemme_output_dir)], capture_output=True)

    if bad_execution_file.exists():
        bad_gemme_preds = (
            pl.read_csv(bad_execution_file, separator=" ", has_header=False, columns=[0], new_columns=["id"])
            .with_columns(
                pl.col("id").str.replace("/", "")
            )
            .to_series()
            .to_list()
        )
    else:
        # if no issues: no file created
        bad_gemme_preds = []

    gemme_folders = [dir for dir in gemme_output_dir.iterdir() if dir.is_dir()]
    records = []
    protein_names = []
    missing_preds = []
    pred_files = []

    unprocessed_fasta_records = [rec for rec in SeqIO.parse(unprocessed_fasta, "fasta")] if unprocessed_fasta else None

    def parse_gemme_directory(dir: Path, unprocessed_fasta_records: list[SeqRecord] = None) -> None:
        """
        Perform checks on GEMME pred directory.
        If everything is fine, add its query protein to the list of valid records.

        Defining it in this scope is the easiest way to access shared memory for multithreading.
        """
        status = "fine"
        if dir.stem in bad_gemme_preds:
            return

        try:
            pred_file = next(iter(dir.glob("*_normPred_evolCombi.txt")))
            pred_files.append(pred_file)
        except StopIteration:
            status = "missing prediction file"

        # TODO check for NAs

        if unprocessed_fasta_records is None:
          fasta_file = next(iter(dir.glob("*.fasta")))
          query_protein = [rec for rec in SeqIO.parse(fasta_file, "fasta")][0]
        else:
          query_protein = [rec for rec in unprocessed_fasta_records if rec.id == dir.stem][0]

        if status == "missing prediction file":
            missing_preds.append({"protein_id": protein_id, "folder": dir.stem})
        elif status == "too many NAs":
            pass
        else:
            records.append(query_protein)
        
        protein_names.append({"folder": dir.stem, "protein_id": protein_id})

    with joblib_progress(
        f"Preprocessing GEMME predictions",
        total=len(gemme_folders),
    ):
        Parallel(n_jobs=-1, require="sharedmem")(
            delayed(parse_gemme_directory)(gemme_dir, unprocessed_fasta_records)
            for gemme_dir in gemme_folders
        )

    map_file = output_dir / "folder_protein_ids.csv"
    logger.info(f"Writing protein IDs to {map_file}")
    pl.from_records(protein_names).sort(by="folder").write_csv(map_file)

    pl.from_records(missing_preds).write_csv(output_dir / "missing_preds.csv")

    # Write remaining valid records to FASTA file
    fasta_output_file = output_dir / "preprocessed.fasta"
    logger.info(f"Writing all sequences to {fasta_output_file}")
    SeqIO.write(records, fasta_output_file, "fasta")

    # Write statistics of valid records
    statistics_file = output_dir / "statistics.txt"
    logger.info(f"Writing summary statistics to {statistics_file}")
    lengths = [len(rec) for rec in records]
    with open(statistics_file, "w+") as f:
        f.write(f"Number of sequences: {len(lengths)}" + "\n")
        f.write(f"Number of variants: {sum(lengths):,}" + "\n")
        f.write(f"Minimal sequence length: {min(lengths)}" + "\n")
        f.write(f"Maximal sequence length: {max(lengths)}" + "\n")
        f.write(f"Median sequence length: {statistics.median(lengths)}")

    # Load predictions into H5 file
    with h5py.File(output_h5_path, "w") as hdf:
        for file in progress.track(
            pred_files,
            description="Loading GEMME predictions into HDF5 file",
        ):
            protein_id = str(file.stem).replace("_normPred_evolCombi", '')
            raw = file.open().readlines()
            temp_file = None
            if raw[0].startswith('"V1"'):
              raw[0] = '"to_aa" ' + raw[0]
              temp_file = file.parent / (file.stem + "_mod.txt")
              temp_file.open("w+").writelines(raw)
            data = torch.tensor(pl.read_csv(temp_file, separator=" ", null_values="NA").transpose()[1:].to_numpy().astype("double"))
            if temp_file:
              temp_file.unlink()
            hdf.create_dataset(name=protein_id, data=data)

if __name__ == "__main__":
    typer.run(main)

