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
from dvc.api import params_show
from typing_extensions import Annotated

from src.vespa2.utils import MeanModel, Mutation, get_device, get_precision
from src.vespa2.utils import load_model as load_model_from_config
from src.vespa2.utils import setup_logger


def load_model(config_key: str, params: dict, checkpoint_dir: Path, embedding_type: str) -> torch.nn.Module:
    architecture = params[config_key]["architecture"]
    model_parameters = params[config_key]["model_parameters"]
    model = load_model_from_config(architecture, model_parameters, embedding_type)

    with open(checkpoint_dir / "wandb_run_id.txt", "r") as f:
        wandb_run_id = f.read()
    """
    api = wandb.Api()
    artifact = api.artifact(
        f"jschlensok/vespa2/model-{wandb_run_id}:best", type="model"
    )
    artifact_dir = artifact.download()
    checkpoint_file = next(Path(artifact_dir).glob("state_dict.pt"))
    """
    checkpoint_file = "./checkpoints/all/esm2/fnn_1_layer/naive_sampling/final/epoch-200/state_dict.pt"
    model.load_state_dict(torch.load(checkpoint_file))

    return model

# dms_reference_file and protein_reference_file can take URLs as well as paths, but Typer doesn't support Union types yet, hence the str annotation
def main(
        model_config_keys: Annotated[list[str], typer.Option("--model")],
        checkpoint_dirs: Annotated[list[Path], typer.Option("--checkpoint-dir")],
        embedding_type: Annotated[str, typer.Option("--embedding-type")],
        embedding_file: Annotated[Path, typer.Option("--embedding-file")],
        pred_output_path: Annotated[Path, typer.Option("--pred-output")],
        spearman_output_path: Annotated[Path, typer.Option("--spearman-output")]
):
    logger = setup_logger()
    device = get_device()
    precision = get_precision()
    dtype = torch.float if precision == "float" else torch.half
    logger.info(f"Using device {str(device)} with precision {precision}")

    params = params_show()
    gemme_alphabet = params["gemme"]["alphabet"]
    protein_reference_file = params["eval"]["proteingym"]["reference_files"]["per_protein"]
    dms_reference_file = params["eval"]["proteingym"]["reference_files"]["per_dms"]
    dms_directory = Path(params["eval"]["proteingym"]["dms_directory"])

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
    proteingym_reference_df = pl.read_csv(dms_reference_file)

    embeddings = {
        key: torch.tensor(data[()], device=device).to(dtype=dtype)
        for key, data in h5py.File(embedding_file, "r").items()
    }

    with progress.Progress(
            progress.TextColumn(
                "[progress.description]Evaluation ProteinGym"
            ),
            progress.BarColumn(),
            progress.TaskProgressColumn(),
            progress.TimeElapsedColumn(),
            progress.TextColumn("Current DMS: {task.description}")
        ) as pbar, torch.no_grad():
        print()
        overall_progress = pbar.add_task(
            f"Evaluation ProteinGym",
            total=proteingym_reference_df["DMS_total_number_mutants"].sum(),
        )

        per_dms_results = []
        raw_preds = []

        for dms in proteingym_reference_df.iter_rows(named=True):
            protein_id = dms["UniProt_ID"]
            dms_id = dms["DMS_id"]
            wildtype_sequence = dms["target_seq"]
            embedding = embeddings[protein_id]

            pbar.update(overall_progress, description=dms_id)

            pred = model(embedding).cpu().numpy()
            # Set scores of non-mutations to be 0
            pred[
                torch.arange(len(wildtype_sequence)),
                torch.tensor([gemme_alphabet.index(aa) for aa in wildtype_sequence]),
            ] = 0.

            mutation_df = pl.read_csv(dms_directory / dms["DMS_filename"])

            # For multi-mutations, simply sum up their SAVs
            # TODO define this in cleaner fashion
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


            dms_pred = torch.tensor(
                [score_mutation(mutation_string) for mutation_string in mutation_df["mutant"]],
                dtype=torch.float,
            )

            dms_exp = torch.tensor(mutation_df["DMS_score"])
            spearman = tm.functional.spearman_corrcoef(dms_pred, dms_exp).item()
            pearson = tm.functional.pearson_corrcoef(dms_pred, dms_exp).item()

            per_dms_results.append(
                {"DMS_id": dms_id, "spearman": spearman, "pearson": pearson}
            )

            raw_preds.append(
                pl.DataFrame({
                    "DMS_id": dms_id,
                    "mutation": mutation_df["mutant"],
                    "DMS_score": dms_exp.numpy(),
                    "VespaG": dms_pred.numpy()
                })
            )


    if spearman_output_path:
        spearman_output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df = pl.from_records(per_dms_results)
        results_df.write_csv(spearman_output_path)

    if pred_output_path:
        pred_output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df = pl.concat(raw_preds)
        results_df.write_csv(pred_output_path)

    gc.collect()
    torch.cuda.empty_cache()
    wandb.finish()


if __name__ == "__main__":
    typer.run(main)

