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


def load_model(config_key: str, params: dict, checkpoint_dir: Path, embedding_type: str) -> torch.nn.Module:
    architecture = params[config_key]["architecture"]
    model_parameters = params[config_key]["model_parameters"]
    model = load_model_from_config(architecture, model_parameters, embedding_type)

    with open(checkpoint_dir / "wandb_run_id.txt", "r") as f:
        wandb_run_id = f.read()
    checkpoint_files = list(checkpoint_dir.iterdir())
    latest_checkpoint = sorted(checkpoint_files, key=lambda dir: int(dir.stem.split('-')[-1]))[-1]
    model.load_state_dict(torch.load(latest_checkpoint_file))

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
