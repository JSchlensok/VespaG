from pathlib import Path
from typing import Annotated

import h5py
import pandas as pd
import typer
from rich import progress

app = typer.Typer()


def store_gemme_as_h5(gemme_folder: Path, output_file: Path) -> None:
    with h5py.File(output_file, "w") as hdf:
        for file in progress.track(
            list(gemme_folder.glob("*_normPred_evolCombi.txt")),
            description=f"Loading GEMME score files from {gemme_folder}",
        ):
            protein_id = file.stem.replace("_normPred_evolCombi", "")
            data = pd.read_csv(file, sep=" ").transpose().to_numpy().astype("double")
            hdf.create_dataset(name=protein_id, data=data)


@app.command()
def load(
    gemme_folder: Annotated[
        Path, typer.Argument(help="Directory with raw GEMME predictions as txt files")
    ],
    output_file: Annotated[Path, typer.Argument(help="Path of output H5 file")],
):
    store_gemme_as_h5(gemme_folder, output_file)


@app.command()
def foo():
    # This is just here to make Typer behave as it doesn't accept just one command
    print("bar")


if __name__ == "__main__":
    app()
