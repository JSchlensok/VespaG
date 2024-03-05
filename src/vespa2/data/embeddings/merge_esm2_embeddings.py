import os
import shutil
from pathlib import Path

import h5py
import rich.progress as progress
import torch
import typer
from typing_extensions import Annotated


def main(
    esm_file_directory: Annotated[Path, typer.Argument()],
    h5_file: Annotated[Path, typer.Argument()],
    split_ids: Annotated[bool, typer.Option("--split-ids")] = False
):
    with h5py.File(h5_file, "w") as hdf:
        for embedding_file in progress.track(
                list(esm_file_directory.rglob("*.pt")),
            description="Merging embedding files into one HDF5 file",
        ):
            data = torch.load(embedding_file)
            label = data["label"]
            if split_ids:
                label = label.split('|')[1]
            else:
                label = label.split(' ')[0]
            embedding = next(iter(data["representations"].values()))
            hdf.create_dataset(name=label, data=embedding)
            os.remove(embedding_file)

    shutil.rmtree(esm_file_directory)


if __name__ == "__main__":
    typer.run(main)
