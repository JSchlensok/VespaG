import contextlib
import os
import warnings
from itertools import islice
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import rich.progress as progress
import torch
from Bio import SeqIO

from vespag.data.embeddings import generate_embeddings
from vespag.utils import (
    DEFAULT_MODEL_PARAMETERS,
    GEMME_ALPHABET,
    get_device,
    load_model,
    mask_non_mutations,
    normalize_scores,
    read_mutation_file,
    setup_logger,
)
from vespag.utils.type_hinting import *

BATCH_SIZE = 100  # TODO parametrize


def chunk_dict(d: dict, chunk_size: int) -> list[dict]:
    """Yield successive n-sized chunks from d."""
    it = iter(d)
    for i in range(0, len(d), chunk_size):
        yield {k: d[k] for k in islice(it, chunk_size)}


def generate_predictions(
    fasta_file: Path,
    output_path: Path | None,
    embedding_file: Path | None = None,
    mutation_file: Path | None = None,
    single_csv: bool = False,
    no_csv: bool = False,
    h5_output: bool = False,
    zero_based_mutations: bool = False,
    transform: bool = False,
    normalize: bool = True,
    clip_to_one: bool = True,
    embedding_type: EmbeddingType = EmbeddingType.esm2,
) -> None:
    logger = setup_logger()
    warnings.filterwarnings("ignore", message="rich is experimental/alpha")

    # Set default output path
    output_path = output_path or Path.cwd() / "output"
    if not output_path.exists():
        logger.info(f"Creating output directory {output_path}")
        output_path.mkdir(parents=True)

    device = get_device()
    params = DEFAULT_MODEL_PARAMETERS
    params["embedding_type"] = embedding_type
    model = load_model(**params).eval().to(device, dtype=torch.float)

    sequences = {rec.id: str(rec.seq) for rec in SeqIO.parse(fasta_file, "fasta")}

    if embedding_file:
        logger.info(f"Loading pre-computed embeddings from {embedding_file}")

    else:
        embedding_file = output_path / f"{embedding_type.value}_embeddings.h5"
        if "HF_HOME" in os.environ:
            plm_cache_dir = Path(os.environ["HF_HOME"])
        else:
            plm_cache_dir = Path.cwd() / f".{embedding_type.value}_cache"
            plm_cache_dir.mkdir(exist_ok=True)
        generate_embeddings(fasta_file, embedding_file, embedding_type=embedding_type, cache_dir=plm_cache_dir)

    h5_output_path = output_path / "vespag_scores.h5"

    # TODO parse multi-mutations
    if mutation_file:
        logger.info("Parsing mutational landscape")
        mutations_per_protein = read_mutation_file(mutation_file, one_indexed=not zero_based_mutations)
    else:
        mutations_per_protein = None

    # TODO test if new implementation is really faster than old one since I ended up reusing most of the old code
    # TODO check if progress bar is smooth enough
    with (
        progress.Progress(
            *progress.Progress.get_default_columns(),
            progress.TimeElapsedColumn(),
            progress.TextColumn("Current protein: {task.fields[current_protein]}"),
        ) as pbar,
        torch.no_grad(),
        h5py.File(h5_output_path, "w") if h5_output else contextlib.nullcontext() as h5_file,
    ):
        prediction_progress = pbar.add_task(
            "Computing predictions", total=sum(map(len, sequences.values())), current_protein="None"
        )
        for batch_sequences in chunk_dict(sequences, BATCH_SIZE):
            embeddings = {
                id: torch.from_numpy(np.array(emb[()], dtype=np.float32))
                for id, emb in h5py.File(embedding_file).items()
                if id in batch_sequences.keys()
            }
            for protein_id, sequence in batch_sequences.items():
                pbar.update(prediction_progress, current_protein=protein_id)
                embedding = embeddings[protein_id].to(device).unsqueeze(0)
                y = model(embedding).squeeze(0).cpu().numpy()
                if normalize:
                    y = normalize_scores(y, transform, clip_to_one=clip_to_one)
                if h5_output:
                    h5_file.create_dataset(protein_id, data=y, dtype=np.float16)

                protein_df = pl.from_records(
                    [
                        {"Mutation": f"{wt_aa}{i + 1}{GEMME_ALPHABET[j]}", "VespaG": score}
                        for i, wt_aa in enumerate(sequence)
                        for j, score in enumerate(y[i])
                        if wt_aa != GEMME_ALPHABET[j]
                    ]
                )
                if mutations_per_protein:
                    sav_df = protein_df.filter(~pl.col("Mutation").str.contains(":"))

                    multi_mutations_df = (
                        protein_df.filter(pl.col("Mutation").str.contains(":"))
                        .lazy()
                        .with_columns(pl.col("Mutation").str.split(":").alias("single_mutations"))
                        .explode("single_mutations")
                        .join(sav_df.lazy(), left_on="single_mutations", right_on="Mutation", how="left")
                        .group_by("Mutation")
                        .agg(pl.col("VespaG").sum().alias("VespaG"))
                        .sort("Mutation")
                        .collect()
                    )

                    protein_df = pl.concat([sav_df, multi_mutations_df])

                if not no_csv:
                    protein_df.write_csv(output_path / (protein_id + ".csv"), float_precision=4)

            pbar.advance(prediction_progress, sum(map(len, batch_sequences.values())))

        if single_csv and not no_csv:
            logger.info("Generating single CSV output")
            pl.concat(
                [
                    pl.scan_csv(output_path / (protein_id + ".csv"))
                    .with_columns(pl.lit(protein_id).alias("Protein"))
                    .select(["Protein", "Mutation", "VespaG"])
                    for protein_id in sequences.keys()
                    if os.stat(output_path / (protein_id + ".csv")).st_size > 16  # just header
                ]
            ).sink_csv(output_path / "vespag_scores_all.csv", float_precision=4)
            logger.info("Tidying up")
            for protein_id in sequences.keys():
                (output_path / (protein_id + ".csv")).unlink()
