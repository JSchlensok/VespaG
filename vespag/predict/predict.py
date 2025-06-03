import csv
import os
import warnings
from pathlib import Path

import h5py
import numpy as np
import rich.progress as progress
import torch
from Bio import SeqIO
from tqdm.rich import tqdm

from vespag.data.embeddings import generate_embeddings
from vespag.utils import (
    AMINO_ACIDS,
    DEFAULT_MODEL_PARAMETERS,
    SAV,
    ScoreNormalizer,
    compute_mutation_score,
    generate_sav_landscape,
    get_device,
    load_model,
    mask_non_mutations,
    read_mutation_file,
    setup_logger,
)
from vespag.utils.type_hinting import *


def generate_predictions(
    fasta_file: Path,
    output_path: Path | None,
    embedding_file: Path | None = None,
    mutation_file: Path | None = None,
    id_map_file: Path | None = None,
    single_csv: bool = False,
    no_csv: bool = False,
    h5_output: bool = False,
    zero_based_mutations: bool = False,
    transform_scores: bool = True,
    normalize_scores: bool = True,
    embedding_type: EmbeddingType = "esm2",
) -> None:
    logger = setup_logger()
    warnings.filterwarnings("ignore", message="rich is experimental/alpha")

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

    embeddings = {
        id: torch.from_numpy(np.array(emb[()], dtype=np.float32))
        for id, emb in tqdm(
            h5py.File(embedding_file).items(),
            desc="Loading embeddings",
            leave=False,
        )
    }

    if id_map_file:
        id_map = {row[0]: row[1] for row in csv.reader(id_map_file.open("r"))}
        for from_id, to_id in id_map.items():
            embeddings[to_id] = embeddings[from_id]
            del embeddings[from_id]

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

    if h5_output:
        h5_output_path = output_path / "vespag_scores_all.h5"
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
