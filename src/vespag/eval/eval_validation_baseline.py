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

from src.vespag.training.dataset import PerResidueDataset
from src.vespag.utils import MeanModel, get_device, get_precision
from src.vespag.utils import load_model as load_model_from_config
from src.vespag.utils import setup_logger

def main(
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
        for dataset_name, dataset in datasets.items():
            for protein_id in dataset.cluster_df["protein_id"]:
                emb = dataset.protein_embeddings[protein_id]
                ann = dataset.protein_annotations[protein_id]
                #pred = ann.flatten()[torch.randperm(ann.numel())].reshape(ann.shape)
                pred = ann[:,torch.randperm(ann.shape[1])]
                nan_mask = torch.isnan(ann) | torch.isnan(pred)
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
                preds = annotations[:,torch.randperm(annotations.shape[1])]
                all_annotations.append(annotations)
                all_preds.append(preds)
            all_annotations = torch.cat(all_annotations)
            all_preds = torch.cat(all_preds)
            nan_mask = torch.isnan(all_annotations) | torch.isnan(all_preds)
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

