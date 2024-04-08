import gc
from pathlib import Path

import h5py
import polars as pl
import re
import rich.progress as progress
import torch.multiprocessing as mp
import torch.optim.lr_scheduler
import torchmetrics as tm
import typer
import wandb
from Bio import SeqIO
from dvc.api import params_show
from tqdm import tqdm
from typing_extensions import Annotated

from src.vespa2.utils import MeanModel, Mutation, get_device, get_precision, load_model, setup_logger


# dms_reference_file and protein_reference_file can take URLs as well as paths, but Typer doesn't support Union types yet, hence the str annotation
def main(
        model_config_keys: Annotated[list[str], typer.Option("--model")],
        checkpoint_dirs: Annotated[list[Path], typer.Option("--checkpoint-dir")],
        sequence_file: Annotated[Path, typer.Option("--sequence-file")],
        embedding_type: Annotated[str, typer.Option("--embedding-type")],
        embedding_file: Annotated[Path, typer.Option("--embedding-file")],
        output_path: Annotated[Path, typer.Option("--output-path", "-o")],
):
    logger = setup_logger()
    device = get_device()
    precision = get_precision()
    dtype = torch.float if precision == "float" else torch.half
    logger.info(f"Using device {str(device)} with precision {precision}")

    params = params_show()
    alphabet = params["gemme"]["alphabet"]
    mutation_df = pl.read_csv("data/test/mega_dataset_v1_rasp/rasp_variants.csv")

    if len(model_config_keys) == 1:
        logger.info("Loading model")
        model = load_model(model_config_keys[0], params["models"], checkpoint_dirs[0], embedding_type)
        model = model.to(device=device, dtype=dtype)

    else:
        logger.info("Loading models for mean model")
        models = [
            load_model(config_key, params["models"], checkpoint_dir, embedding_type).to(
                device=device, dtype=dtype
            )
            for config_key, checkpoint_dir in zip(model_config_keys, checkpoint_dirs)
        ]
        model = MeanModel(*models)

    logger.info("Loading evaluation data")
    embeddings = {
        key: torch.tensor(data[()], device=device).to(dtype=dtype)
        for key, data in h5py.File(embedding_file, "r").items()
    }

    records = []
    all_aas = set(alphabet)
    wt_sequences = {rec.id: rec.seq for rec in SeqIO.parse(sequence_file, "fasta")}

    def score_mutation(preds: torch.Tensor, mutation_str: str) -> float:
            return sum([
            preds[sav.position][alphabet.index(sav.to_aa)]
            for sav in Mutation.from_mutation_string(mutation_str, one_indexed=True)
            ])

    with torch.no_grad():
        with progress.Progress(
            *progress.Progress.get_default_columns(), progress.TimeElapsedColumn()
        ) as pbar:
            overall_progress = pbar.add_task("Overall", total=len(embeddings))
            for protein_id, emb in embeddings.items():
                pred = model(emb).cpu().numpy()
                mutations = mutation_df.filter(pl.col("pdbid") == protein_id)
                records.extend([
                    {
                        "pdbid": protein_id,
                        "mutation": row["variant"],
                        "VespaG": score_mutation(pred, row["variant"])
                    }
                    for row in mutations.iter_rows(named=True)
                ])
                pbar.advance(overall_progress)
            pbar.remove_task(overall_progress)

    results_df = pl.from_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.write_csv(output_path)

    gc.collect()
    torch.cuda.empty_cache()
    wandb.finish()


if __name__ == "__main__":
    typer.run(main)
