from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Literal

import requests
import rich.progress as progress
import torch
import torch.multiprocessing as mp
from rich.logging import RichHandler

from vespag.models import FNN, MinimalCNN

from .type_hinting import Architecture, EmbeddingType

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

DEFAULT_MODEL_PARAMETERS = {
    "architecture": "fnn",
    "model_parameters": {"hidden_dims": [256], "dropout_rate": 0.2},
    "embedding_type": "esm2",
}

PROTEINGYM_CHANGED_FILENAMES = {
    "A0A140D2T1_ZIKV_Sourisseau_growth_2019": "A0A140D2T1_ZIKV_Sourisseau_2019.csv",
    "A4_HUMAN_Seuma_2021": "A4_HUMAN_Seuma_2022.csv",
    "A4D664_9INFA_Soh_CCL141_2019": "A4D664_9INFA_Soh_2019.csv",
    "CAPSD_AAV2S_Sinai_substitutions_2021": "CAPSD_AAV2S_Sinai_2021.csv",
    "CP2C9_HUMAN_Amorosi_abundance_2021": "CP2C9_HUMAN_Amorosi_2021_abundance.csv",
    "CP2C9_HUMAN_Amorosi_activity_2021": "CP2C9_HUMAN_Amorosi_2021_activity.csv",
    "DYR_ECOLI_Thompson_plusLon_2019": "DYR_ECOLI_Thompson_2019.csv",
    "GCN4_YEAST_Staller_induction_2018": "GCN4_YEAST_Staller_2018.csv",
    "B3VI55_LIPST_Klesmith_2015": "LGK_LIPST_Klesmith_2015.csv",
    "MTH3_HAEAE_Rockah-Shmuel_2015": "MTH3_HAEAE_RockahShmuel_2015.csv",
    "NRAM_I33A0_Jiang_standard_2016": "NRAM_I33A0_Jiang_2016.csv",
    "P53_HUMAN_Giacomelli_NULL_Etoposide_2018": "P53_HUMAN_Giacomelli_2018_Null_Etoposide.csv",
    "P53_HUMAN_Giacomelli_NULL_Nutlin_2018": "P53_HUMAN_Giacomelli_2018_Null_Nutlin.csv",
    "P53_HUMAN_Giacomelli_WT_Nutlin_2018": "P53_HUMAN_Giacomelli_2018_WT_Nutlin.csv",
    "R1AB_SARS2_Flynn_growth_2022": "R1AB_SARS2_Flynn_2022.csv",
    "RL401_YEAST_Mavor_2016": "RL40A_YEAST_Mavor_2016.csv",
    "RL401_YEAST_Roscoe_2013": "RL40A_YEAST_Roscoe_2013.csv",
    "RL401_YEAST_Roscoe_2014": "RL40A_YEAST_Roscoe_2014.csv",
    "SPIKE_SARS2_Starr_bind_2020": "SPIKE_SARS2_Starr_2020_binding.csv",
    "SPIKE_SARS2_Starr_expr_2020": "SPIKE_SARS2_Starr_2020_expression.csv",
    "SRC_HUMAN_Ahler_CD_2019": "SRC_HUMAN_Ahler_2019.csv",
    "TPOR_HUMAN_Bridgford_S505N_2020": "TPOR_HUMAN_Bridgford_2020.csv",
    "VKOR1_HUMAN_Chiasson_abundance_2020": "VKOR1_HUMAN_Chiasson_2020_abundance.csv",
    "VKOR1_HUMAN_Chiasson_activity_2020": "VKOR1_HUMAN_Chiasson_2020_activity.csv",
}


def save_async(obj, pool: mp.Pool, path: Path, mkdir: bool = True):
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    pool.apply_async(torch.save, (obj, path))


def load_model_from_config(
    architecture: str, model_parameters: dict, embedding_type: str
):
    if architecture == "fnn":
        model = FNN(
            hidden_layer_sizes=model_parameters["hidden_dims"],
            input_dim=get_embedding_dim(embedding_type),
            dropout_rate=model_parameters["dropout_rate"],
        )
    elif architecture == "cnn":
        model = MinimalCNN(
            input_dim=get_embedding_dim(embedding_type),
            n_channels=model_parameters["n_channels"],
            kernel_size=model_parameters["kernel_size"],
            padding=model_parameters["padding"],
            fnn_hidden_layers=model_parameters["fully_connected_layers"],
            cnn_dropout_rate=model_parameters["dropout"]["cnn"],
            fnn_dropout_rate=model_parameters["dropout"]["fnn"],
        )
    return model


def load_model(
    architecture: Architecture,
    model_parameters: dict,
    embedding_type: EmbeddingType,
    checkpoint_file: Path = None,
) -> torch.nn.Module:
    checkpoint_file = checkpoint_file or Path.cwd() / "model_weights/state_dict_v2.pt"
    model = load_model_from_config(architecture, model_parameters, embedding_type)
    model.load_state_dict(torch.load(checkpoint_file))
    return model


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )
    logger = logging.getLogger("rich")
    logger.setLevel(logging.INFO)
    return logger


def get_embedding_dim(embedding_type: EmbeddingType) -> int:
    if embedding_type == "prott5":
        return 1024
    elif embedding_type == "esm2":
        return 2560


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_precision() -> Literal["half", "float"]:
    if "cuda" in str(get_device()):
        return "half"
    else:
        return "float"


def download(
    url: str, path: Path, progress_description: str, remove_bar: bool = False
) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    with progress.Progress(
        progress.TextColumn("[progress.description]{task.description}"),
        progress.BarColumn(),
        progress.TaskProgressColumn(),
        progress.DownloadColumn(),
        progress.TransferSpeedColumn(),
    ) as pbar, open(path, "wb") as f:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        download_progress = pbar.add_task(progress_description, total=total_size)
        for data in response.iter_content(256):
            f.write(data)
            pbar.update(download_progress, advance=len(data))

        if remove_bar:
            pbar.remove_task(download_progress)


def unzip(
    zip_path: Path, out_path: Path, progress_description: str, remove_bar: bool = False
) -> None:
    out_path.mkdir(exist_ok=True, parents=True)
    with progress.Progress(
        *progress.Progress.get_default_columns()
    ) as pbar, zipfile.ZipFile(zip_path, "r") as zip:
        extraction_progress = pbar.add_task(
            progress_description, total=len(zip.infolist())
        )
        for member in zip.infolist():
            zip.extract(member, out_path)
            pbar.advance(extraction_progress)

        if remove_bar:
            pbar.remove_task(extraction_progress)
