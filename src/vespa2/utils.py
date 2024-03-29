from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.multiprocessing as mp
from rich.logging import RichHandler

from src.vespa2.models import FNN, MinimalCNN

EmbeddingType = Literal["prott5", "esm2"]
PrecisionType = Literal["half", "float"]


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


def save_async(obj, pool: mp.Pool, path: Path, mkdir: bool = True):
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    pool.apply_async(torch.save, (obj, path))


def load_model_from_config(architecture: str, model_parameters: dict, embedding_type: str):
    if architecture == "fnn":
        model = FNN(
            hidden_layer_sizes=model_parameters["hidden_dims"],
            input_dim=get_embedding_dim(embedding_type),
            dropout_rate=model_parameters["dropout_rate"]
        )
    elif architecture == "cnn":
        model = MinimalCNN(
            input_dim=get_embedding_dim(embedding_type),
            n_channels=model_parameters["n_channels"],
            kernel_size=model_parameters["kernel_size"],
            padding=model_parameters["padding"],
            fnn_hidden_layers=model_parameters["fully_connected_layers"],
            cnn_dropout_rate=model_parameters["dropout"]["cnn"],
            fnn_dropout_rate=model_parameters["dropout"]["fnn"]
        )
    else:
        model = None
        # TODO

    return model


@dataclass
class SAV:
    position: int
    from_aa: str
    to_aa: str

    @classmethod
    def from_sav_string(cls, sav_string: str, one_indexed: bool = False, offset: int = 0) -> SAV:
        from_aa, to_aa = sav_string[0], sav_string[-1]
        position = int(sav_string[1:-1]) - offset
        if one_indexed:
            position -= 1
        return SAV(position, from_aa, to_aa)

    def __str__(self) -> str:
        return f"{self.from_aa}{self.position}{self.to_aa}"

    def __hash__(self):
        return hash(str(self))


@dataclass
class Mutation:
    savs: list[SAV]

    @classmethod
    def from_mutation_string(cls, mutation_string: str, one_indexed: bool = False, offset: int = 0) -> Mutation:
        return Mutation([SAV.from_sav_string(sav_string, one_indexed=one_indexed, offset=offset) for sav_string in
                         mutation_string.split(':')])

    def __str__(self) -> str:
        return ':'.join([str(sav) for sav in self.savs])

    def __hash__(self):
        return hash(str(self))

    def __iter__(self):
        yield from self.savs


class MeanModel(torch.nn.Module):
    def __init__(self, *models: torch.nn.Module):
        super(MeanModel, self).__init__()
        self.models = list(models)

    def forward(self, x):
        return sum([model(x) for model in self.models]) / len(self.models)
