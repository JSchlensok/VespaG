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

from src.vespa2.training.dataset import PerResidueDataset
from src.vespa2.utils import MeanModel, get_device, get_precision, load_model, setup_logger


# dms_reference_file and protein_reference_file can take URLs as well as paths, but Typer doesn't support Union types yet, hence the str annotation
def main(
        model_config_keys: Annotated[list[str], typer.Option("--model")],
        checkpoint_dirs: Annotated[list[Path], typer.Option("--checkpoint-dir")],
        embedding_type: Annotated[str, typer.Option("--embedding-type")],
        output_path: Annotated[Path, typer.Option("--output-path", "-o")],
):
    logger = setup_logger()
    device = get_device()
    precision = get_precision()
    dtype = torch.float if precision == "float" else torch.half
    logger.info(f"Using device {str(device)} with precision {precision}")

    params = params_show()
    batch_size = params["eval"]["batch_size"]

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
    max_len = 4096 if embedding_type == "esm2" else 99999
    datasets = {
        dataset_name: PerResidueDataset(
            dataset_params["embeddings"][embedding_type],
            dataset_params["gemme"],
            dataset_params["splits"]["val"],
            precision=precision,
            device=device,
            max_len=max_len
        )
        for dataset_name, dataset_params in params["datasets"]["train"].items()
    }

    dataloaders = {
        name: torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        for name, dataset in datasets.items()
    }

    criterion = torch.nn.MSELoss()

    loss_per_dataset = {}
    records = []

    with progress.Progress(
            progress.TextColumn(
                "[progress.description]Evaluation validation data"
            ),
            progress.BarColumn(),
            progress.TaskProgressColumn(),
            progress.TimeElapsedColumn(),
            progress.TextColumn("Current dataset: {task.description}")
        ) as pbar, torch.no_grad():
        print()
        overall_progress = pbar.add_task(
            f"Evaluation validation data",
            total=sum([len(dl) for dl in dataloaders.values()])
        )
        model.eval()
        for dataset_name, dataset in datasets.items():
            for protein_id in dataset.cluster_df["protein_id"]:
                emb = dataset.protein_embeddings[protein_id]
                ann = dataset.protein_annotations[protein_id]
                nan_mask = torch.isnan(ann)
                pred = model(emb)
                loss = criterion(pred[~nan_mask], ann[~nan_mask])
                records.append({"dataset": dataset_name, "protein_id": protein_id, "mse": loss})
                pbar.advance(overall_progress)
        """
        for dataset_name, dl in dataloaders.items():
            pbar.update(overall_progress, description=dataset_name)

            all_annotations = []
            all_preds = []
            for embeddings, annotations in dl:
                pbar.advance(overall_progress)
                preds = model(embeddings)
                nan_mask = torch.isnan(annotations)
                all_annotations.append(annotations)
                all_preds.append(preds)
            all_annotations = torch.cat(all_annotations)
            all_preds = torch.cat(all_preds)
            nan_mask = torch.isnan(all_annotations)
            all_annotations = all_annotations[~nan_mask]
            all_preds = all_preds[~nan_mask]
            loss = criterion(all_preds, all_annotations).item()

            records.append({"dataset": dataset_name, "mse": loss})
         """

    results_df = pl.from_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.write_csv(output_path)

    gc.collect()
    torch.cuda.empty_cache()
    wandb.finish()


if __name__ == "__main__":
    typer.run(main)

