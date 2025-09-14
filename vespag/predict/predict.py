import csv
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
from tqdm.rich import tqdm

from vespag.data.embeddings import generate_embeddings
from vespag.utils import (
    AMINO_ACIDS,
    DEFAULT_MODEL_PARAMETERS,
    GEMME_ALPHABET,
    SAV,
    compute_mutation_score,
    generate_sav_landscape,
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
    for i in range(0, len(d), chunk_size):
        yield {k: d[k] for k in islice(iter(d), chunk_size)}


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
    embedding_type: EmbeddingType = "esm2",
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

    with (
        progress.Progress(
            *progress.Progress.get_default_columns(),
            progress.TimeElapsedColumn(),
            progress.TextColumn("Current protein: {task.description}"),
        ) as pbar,
        torch.no_grad(),
        h5py.File(h5_output_path, "w") as h5_file
    ):
        prediction_progress = pbar.add_task(
            "Computing predictions",
            total = sum(map(len, sequences.values()))
        )
        for batch_sequences in chunk_dict(sequences, BATCH_SIZE):
            embeddings = {
                id: torch.from_numpy(np.array(emb[()], dtype=np.float32))
                for id, emb in h5py.File(embedding_file).items()
                if id in batch_sequences.keys()
            }
            for protein_id, sequence in batch_sequences.items():
                pbar.update(prediction_progress, description=protein_id)
                embedding = embeddings[protein_id].to(device).unsqueeze(0)
                y = model(embedding).squeeze(0)
                y = mask_non_mutations(y, sequence).cpu().numpy()
                if normalize:
                    y = normalize_scores(y)
                h5_file.create_dataset(protein_id, data=y, dtype=np.float16)
                
                # TODO parse mutation file
                # TODO score multi-mutations
                # TODO concatenate into one big file if requested
                # TODO transform scores if necessary
                pl.from_records(
                    [
                        {
                        "Mutation": f"{wt_aa}{i+1}{GEMME_ALPHABET[j]}",
                        "VespaG": score
                        }
                        for i, wt_aa in enumerate(sequence)
                        for j, score in enumerate(y[i])
                        if wt_aa != GEMME_ALPHABET[j]
                    ]
                ).write_csv(output_path / (protein_id + ".csv"))

            pbar.advance(prediction_progress, sum(map(len, batch_sequences.values())))

            # TODO remove h5 output if it's not needed
"""
    if mutation_file:
        logger.info("Parsing mutational landscape")
        mutations_per_protein = read_mutation_file(mutation_file, one_indexed=not zero_based_mutations)
    else:
        logger.info("Generating mutational landscape")
        mutations_per_protein = generate_sav_landscape(
            sequences=sequences, zero_based_mutations=zero_based_mutations, tqdm=True
        )

    vespag_scores = {}
    scores_per_protein = {}
    with (
        progress.Progress(
            progress.TextColumn("[progress.description]Computing"),
            progress.BarColumn(),
            progress.TaskProgressColumn(),
            progress.TimeElapsedColumn(),
            progress.TextColumn("Current protein: {task.description}"),
        ) as pbar,
        torch.no_grad(),
        h5py.File(output_h5_file, "w") as h5_file
    ):
        logger.info("Generating predictions")
        prediction_progress = pbar.add_task(
            "Generating predictions",
            total=sum([len(mutations) for mutations in mutations_per_protein.values()]),
        )
        for id, sequence in sequences.items():
            pbar.update(prediction_progress, description=id)
            embedding = embeddings[id].to(device).unsqueeze(0)
            y = model(embedding).squeeze(0)
            y = mask_non_mutations(y, sequence)
            vespag_scores[id] = y.detach().numpy()
            pbar.advance(prediction_progress, len(mutations_per_protein[id]))

        if normalize_scores:
            normalizer = ScoreNormalizer("minmax")
            normalizer.fit(np.concatenate([y.flatten() for y in vespag_scores.values()]))
        else:
            normalizer = None

        pbar.remove_task(prediction_progress)

        logger.info("Scoring mutations")
        scoring_progress = pbar.add_task(
            "Scoring mutations",
            total=sum([len(mutations) for mutations in mutations_per_protein.values()]),
        )
        for id, y in vespag_scores.items():
            pbar.update(scoring_progress, description=id)
            scores_per_protein[id] = {
                mutation: compute_mutation_score(
                    y,
                    mutation,
                    transform=transform_scores,
                    embedding_type=embedding_type,
                    normalizer=normalizer,
                    pbar=pbar,
                    progress_id=scoring_progress,
                )
                for mutation in mutations_per_protein[id]
            }
        pbar.remove_task(scoring_progress)

    if not h5_output:
        # TODO delete h5 file
        logger.info(f"Serializing predictions to {h5_output_path}")
        with h5py.File(h5_output_path, "w") as f:
            for id, vespag_prediction in tqdm(vespag_scores.items(), leave=False):
                f.create_dataset(id, data=vespag_prediction)

    if not no_csv:
        logger.info("Generating CSV output")
        if not single_csv:
            for protein_id, mutations in tqdm(scores_per_protein.items(), leave=False):
                output_file = output_path / (protein_id + ".csv")
                with output_file.open("w+") as f:
                    f.write("Mutation,VespaG\n")
                    f.writelines([f"{sav!s},{score}\n" for sav, score in mutations.items()])
        else:
            output_file = output_path / "vespag_scores_all.csv"
            with output_file.open("w+") as f:
                f.write("Protein,Mutation,VespaG\n")
                f.writelines(
                    list(
                        tqdm(
                            [
                                f"{protein_id},{sav!s},{score}\n"
                                for protein_id, mutations in scores_per_protein.items()
                                for sav, score in mutations.items()
                            ],
                            leave=False,
                        )
                    )
                )
"""
