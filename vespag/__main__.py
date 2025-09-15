from pathlib import Path
from typing import Annotated

import typer

from .data.embeddings import generate_embeddings
from .eval import eval
from .predict import generate_predictions
from .training.train import train as run_training
from .utils.type_hinting import EmbeddingType

app = typer.Typer()

app.add_typer(eval.app, name="eval")


@app.command()
def predict(
    fasta_file: Annotated[
        Path,
        typer.Option(
            "-i",
            "--input",
            help="Path to FASTA-formatted file containing protein sequence(s)",
        ),
    ],
    output_path: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output",
            help="Path for saving created CSV and/or H5 files. Defaults to ./output",
        ),
    ] = None,
    embedding_file: Annotated[
        Path | None,
        typer.Option(
            "-e",
            "--embeddings",
            help="Path to pre-generated input embeddings. Embeddings will be generated from scratch if no path is provided.",
        ),
    ] = None,
    mutation_file: Annotated[
        Path | None,
        typer.Option("--mutation-file", help="CSV file specifying specific mutations to score"),
    ] = None,
    single_csv: Annotated[
        bool,
        typer.Option(
            "--single-csv/--multi-csv",
            help="Whether to return one CSV file for all proteins instead of a single file for each protein",
        ),
    ] = False,
    no_csv: Annotated[
        bool,
        typer.Option("--no-csv/--csv", help="Whether no CSV output should be produced at all"),
    ] = False,
    h5_output: Annotated[
        bool,
        typer.Option(
            "--h5-output/--no-h5-output",
            help="Whether a file containing predictions in HDF5 format should be created",
        ),
    ] = False,
    zero_based_mutations: Annotated[
        bool,
        typer.Option(
            "--zero-idx/--one-idx",
            help="Whether to enumerate the sequence starting at 0",
        ),
    ] = False,
    transform_scores: Annotated[
        bool,
        typer.Option(
            "--transform/--dont-transform",
            help="Whether to transform scores to same distribution as GEMME scores",
        ),
    ] = False,
    normalize_scores: Annotated[
        bool,
        typer.Option(
            "--normalize/--dont-normalize",
            help="Whether to transform scores to [0, 1] range",
        ),
    ] = True,
    embedding_type: Annotated[
        EmbeddingType,
        typer.Option("--embedding-type", help="Type of pLM used for generating embeddings"),
    ] = EmbeddingType.esm2,
) -> None:
    generate_predictions(
        fasta_file=fasta_file,
        output_path=output_path,
        embedding_file=embedding_file,
        mutation_file=mutation_file,
        single_csv=single_csv,
        no_csv=no_csv,
        h5_output=h5_output,
        zero_based_mutations=zero_based_mutations,
        normalize=normalize_scores,
        transform=transform_scores,
        embedding_type=embedding_type,
    )


@app.command()
def embed(
    input_fasta_file: Annotated[Path, typer.Argument(help="Path of input FASTA file")],
    output_h5_file: Annotated[Path, typer.Argument(help="Path for saving HDF5 file with computed embeddings")],
    cache_dir: Annotated[
        Path,
        typer.Option("-c", "--cache-dir", help="Custom path to download model checkpoints to"),
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
        str | None,
        typer.Option("--pretrained-path", help="Path or URL of pretrained transformer"),
    ] = None,
):
    generate_embeddings(input_fasta_file, output_h5_file, cache_dir, embedding_type, pretrained_path)


@app.command()
def train(
    model_config_key: Annotated[str, typer.Option("--model")],
    datasets: Annotated[list[str], typer.Option("--dataset")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")],
    embedding_type: Annotated[str, typer.Option("--embedding-type", "-e")],
    compute_full_train_loss: Annotated[bool, typer.Option("--full-train-loss")] = False,
    sampling_strategy: Annotated[str, typer.Option("--sampling-strategy")] = "basic",
    wandb_config: Annotated[tuple[str, str] | None, typer.Option("--wandb")] = None,
    limit_cache: Annotated[bool, typer.Option("--limit-cache")] = False,
    use_full_dataset: Annotated[bool, typer.Option("--use-full-dataset")] = False,
):
    run_training(
        model_config_key=model_config_key,
        datasets=datasets,
        output_dir=output_dir,
        embedding_type=embedding_type,
        compute_full_train_loss=compute_full_train_loss,
        sampling_strategy=sampling_strategy,
        wandb_config=wandb_config,
        limit_cache=limit_cache,
        use_full_dataset=use_full_dataset,
    )


if __name__ == "__main__":
    app(prog_name="vespag")
