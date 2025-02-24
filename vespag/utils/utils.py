from __future__ import annotations

import logging
import math
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import requests
import rich.progress as progress
import sklearn.preprocessing
import torch
import torch.multiprocessing as mp
from rich.logging import RichHandler

from vespag.models import FNN, MinimalCNN

from .type_hinting import Architecture, EmbeddingType

GEMME_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
VESPA_ALPHABET = "ALGVSREDTIPKFQNYMHWC"
AMINO_ACIDS = sorted(GEMME_ALPHABET)

DEFAULT_MODEL_PARAMETERS = {
    "architecture": Architecture.fnn,
    "model_parameters": {"hidden_dims": [256], "dropout_rate": 0.2},
    "embedding_type": EmbeddingType.esm2,
}

MODEL_VERSION = "v2"


def save_async(obj, pool: mp.Pool, path: Path, mkdir: bool = True):
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    pool.apply_async(torch.save, (obj, path))


def load_model_from_config(architecture: str, model_parameters: dict, embedding_type: str):
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
    checkpoint_file: Path | None = None,
) -> torch.nn.Module:
    checkpoint_file = checkpoint_file or Path.cwd() / f"model_weights/{MODEL_VERSION}/{embedding_type.value}.pt"
    model = load_model_from_config(architecture.value, model_parameters, embedding_type)
    model.load_state_dict(torch.load(checkpoint_file))
    return model


def setup_logger() -> logging.Logger:
    logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
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


def download(url: str, path: Path, progress_description: str, remove_bar: bool = False) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    with (
        progress.Progress(
            progress.TextColumn("[progress.description]{task.description}"),
            progress.BarColumn(),
            progress.TaskProgressColumn(),
            progress.DownloadColumn(),
            progress.TransferSpeedColumn(),
        ) as pbar,
        open(path, "wb") as f,
    ):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        download_progress = pbar.add_task(progress_description, total=total_size)
        for data in response.iter_content(256):
            f.write(data)
            pbar.update(download_progress, advance=len(data))

        if remove_bar:
            pbar.remove_task(download_progress)


def unzip(zip_path: Path, out_path: Path, progress_description: str, remove_bar: bool = False) -> None:
    out_path.mkdir(exist_ok=True, parents=True)
    with progress.Progress(*progress.Progress.get_default_columns()) as pbar, zipfile.ZipFile(zip_path, "r") as zip:
        extraction_progress = pbar.add_task(progress_description, total=len(zip.infolist()))
        for member in zip.infolist():
            zip.extract(member, out_path)
            pbar.advance(extraction_progress)

        if remove_bar:
            pbar.remove_task(extraction_progress)


def read_gemme_table(txt_file: Path) -> np.ndarray:
    df = pd.read_csv(txt_file, sep=" ").fillna(0)
    return df.to_numpy()


# TODO make this more elegant, e.g. through .npz file
# write test that Spearman on PG stays the same
raw_score_cdf = np.loadtxt("data/score_transformation/vespag_scores.csv", delimiter=",")
sorted_gemme_scores = np.loadtxt("data/score_transformation/sorted_gemme_scores.csv", delimiter=",")


def transform_scores(scores: np.typing.ArrayLike[float]) -> list[float]:
    """Transform VespaG score distribution by mapping it to a known distribution of GEMME scores through its quantile"""
    # TODO vectorize, this is quick and dirty
    transformed_scores = []
    for score in scores:
        quantile = (raw_score_cdf <= score).mean()
        transformed_scores.append(np.interp(quantile, np.linspace(0, 1, len(sorted_gemme_scores)), sorted_gemme_scores))
    return transformed_scores


class ScoreNormalizer:
    def __init__(self, type: Literal["sigmoid", "minmax"]) -> None:
        self.type = type
        if type == "minmax":
            self.scaler = sklearn.preprocessing.MinMaxScaler()
        else:
            self.scaler = None

    def fit(self, all_scores: np.ndarray | Iterable[float]) -> None:
        if type(all_scores) != np.ndarray:
            all_scores = np.array(all_scores)
        if self.type == "minmax":
            self.scaler.fit(all_scores.reshape(-1, 1))
        else:
            pass

    def normalize_score(self, score: float) -> float:
        """Normalize VespaG score to range."""
        return self.normalize_scores([score])[0]

    def normalize_scores(self, scores: np.ndarray | Sequence[float]) -> list[float]:
        """Normalize VespaG scores to range."""
        if self.type == "sigmoid":
            return [1 / (1 + math.exp(-score)) for score in scores]
        elif self.type == "minmax":
            if type(scores) != np.ndarray:
                scores = np.array(scores)
            return list(self.scaler.transform(scores.reshape(-1, 1)).reshape(-1))
