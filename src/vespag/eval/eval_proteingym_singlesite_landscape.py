import gc
from pathlib import Path

import h5py
import polars as pl
import rich.progress as progress
import torch.multiprocessing as mp
import torch.optim.lr_scheduler
import torchmetrics as tm
import typer
import wandb
from Bio import SeqIO
from tqdm import tqdm
from typing_extensions import Annotated

from src.vespag.utils import MeanModel, Mutation, get_device, get_precision
from src.vespag.utils import load_model as load_model_from_config
from src.vespag.utils import setup_logger


def load_model(config_key: str, params: dict, checkpoint_dir: Path, embedding_type: str) -> torch.nn.Module:
    architecture = params[config_key]["architecture"]
    model_parameters = params[config_key]["model_parameters"]
    model = load_model_from_config(architecture, model_parameters, embedding_type)

    with open(checkpoint_dir / "wandb_run_id.txt", "r") as f:
        wandb_run_id = f.read()
    api = wandb.Api()
    artifact = api.artifact(
        f"jschlensok/vespa2/model-{wandb_run_id}:best", type="model"
    )
    artifact_dir = artifact.download()
    checkpoint_file = next(Path(artifact_dir).glob("state_dict.pt"))
    model.load_state_dict(torch.load(checkpoint_file))

    return model

# dms_reference_file and protein_reference_file can take URLs as well as paths, but Typer doesn't support Union types yet, hence the str annotation
def main(
        model_config_keys: Annotated[list[str], typer.Option("--model")],
        checkpoint_dirs: Annotated[list[Path], typer.Option("--checkpoint-dir")],
        embedding_type: Annotated[str, typer.Option("--embedding-type")],
        fasta_path: Annotated[Path, typer.Option("--fasta")],
        embedding_file: Annotated[Path, typer.Option("--embedding-file")],
        output_path: Annotated[Path, typer.Option("--output-path", "-o")],
):
    logger = setup_logger()
    device = get_device()
    precision = get_precision()
    dtype = torch.float if precision == "float" else torch.half
    logger.info(f"Using device {str(device)} with precision {precision}")

    gemme_alphabet = "ACDEFGHIKLMNPQRSTVWY"
    protein_reference_file = Path("/mnt/project/schlensok/VESPA2/ProteinGym/MSA_Files_Elodie/neffTab_full.csv")
    dms_reference_file = Path("data/proteingym_substitutions/reference.csv")
    dms_directory = Path("/mnt/project/schlensok/VESPA2/ProteinGym/ProteinGym_substitutions/")

    params = {
        "fnn": {
            "architecture": "fnn",
            "model_parameters": {
                "hidden_dims": [256, 64],
                "dropout_rate": None
            },
            "training_parameters": {
                "learning_rate": 0.0001,
                "batch_size": {
                    "training": 25000,
                    "validation": 8192
                },
                "epochs": 200,
                "val_every_epoch": 1,
                "checkpoint_every_epoch": None
            },
        },
        "cnn": {
            "architecture": "cnn",
            "model_parameters": {
                "n_channels": 256,
                "kernel_size": 7,
                "padding": 3,
                "fully_connected_layers": [ 256, 64 ],
                "dropout": {
                    "fnn": None,
                    "cnn": .2
                }
            },
            "training_parameters": {
                "learning_rate": .0001,
                "batch_size": {
                    "training": 25000,
                    "validation": 8192
                },
                "epochs": 200,
                "val_every_epoch": 1,
                "checkpoint_every_epoch": None
            }
        }
    }

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

    wildtype_sequences = {rec.id.split('|')[1]: str(rec.seq) for rec in SeqIO.parse(fasta_path, "fasta")}

    dfs = []
    for seq_id, seq in tqdm(list(wildtype_sequences.items())):
        embedding = embeddings[seq_id]
        pred = model(embedding).cpu().numpy()

        def score_mutation(mutation_string: str) -> float:
            pbar.advance(overall_progress)
            return sum(
                [
                    pred[sav.position][gemme_alphabet.index(sav.to_aa)]
                    for sav in Mutation.from_mutation_string(
                    mutation_string, one_indexed=True
                )
                ]
            )

        mutations = [f"{wt_aa}{i+1}{to_aa}" for i, wt_aa in enumerate(seq) for to_aa in set(alphabet) - {wt_aa}]
        dms_pred = [score_mutation(mutation) for mutation in mutations]
        df = (
            pl.DataFrame(dms_pred.numpy())
            .with_columns([
                pl.lit(seq_id).alias("protein_id"),
                pl.Series(mutations).alias("mutation")
            ])
        )
        dfs.append(df)

    pl.concat(dfs).write_csv(output_path)

    gc.collect()
    torch.cuda.empty_cache()
    wandb.finish()


if __name__ == "__main__":
    typer.run(main)
