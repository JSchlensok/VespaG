from pathlib import Path
from typing import Union

import typer
from Bio import SeqIO
from typing_extensions import Annotated

from src.vespag.data.embeddings import Embedder, save_embeddings
from src.vespag.runner.type_hinting import EmbeddingType

model_names = {
    "esm2": "facebook/esm2_t36_3B_UR50D",
    "prott5": "Rostlab/prot_t5_xl_uniref50",
}


def main(
    input_fasta_file: Annotated[Path, typer.Argument(help="Path of input FASTA file")],
    output_h5_file: Annotated[
        Path, typer.Argument(help="Path for saving HDF5 file with computed embeddings")
    ],
    cache_dir: Annotated[
        Path,
        typer.Option(
            "-c", "--cache-dir", help="Custom path to download model checkpoints to"
        ),
    ],
    embedding_type: Annotated[
        EmbeddingType,
        typer.Option(
            "-e",
            "--embedding-type",
            case_sensitive=False,
            help="Type of embeddings to generate",
        ),
    ] = EmbeddingType.esm2,
    pretrained_path: Annotated[
        Union[Path, str],
        typer.Option(
            "--pretrained-path", help="Path or URL of pretrained transformer."
        ),
    ] = None,
):
    if embedding_type and not pretrained_path:
        pretrained_path = model_names[embedding_type]

    sequences = {rec.id: str(rec.seq) for rec in SeqIO.parse(input, "fasta")}
    embedder = Embedder(pretrained_path, cache_dir)
    embeddings = Embedder.embed(sequences)
    Embedder.save_embeddings(embeddings, output_h5_file)


if __name__ == "__main__":
    typer.run(main)
