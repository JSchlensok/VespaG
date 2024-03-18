from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union

config_item = Union[str, int, list[int], float, Path]
config_dict = dict[str, config_item]

# TODO what do we actually need here

class LossFunction(str, Enum):
    mse = "mse"
    bce = "bce"
    ce = "ce"
    mae = "mae"


class Optimizer(str, Enum):
    adam = "adam"
    adamw = "adamw"


class Architecture(str, Enum):
    fnn = "fnn"
    cnn = "cnn"
    combined = "combined"
    mean = "mean"


class ActivationFunction(str, Enum):
    none = None
    relu = "relu"
    leaky_relu = "leaky_relu"
    sigmoid = "sigmoid"


class EmbeddingType(str, Enum):
    prott5 = "prott5"
    ankh = "ankh"

class Score(str, Enum):
    effect = "effect"
    conservation = "conservation"


class BaseConfig:
    def __init__(self):
        pass

    def to_dict(self):
        return self.__dict__


@dataclass
class RunConfig(BaseConfig):
    name: str

@dataclass
class DataConfig(BaseConfig):
    embedding_file: Path
    gemme_directory: Path
    whitelist_fasta: Path
    gemme_id_regex: str = None
    additional_features: dict[str, Union[str, None, bool]] = None
    split: bool = False
    max_len: int = 999999


@dataclass
class ModelConfig(BaseConfig):
    architecture: str
    embedding_dim: int
    activation: ActivationFunction
    output_activation: ActivationFunction
    dropout: float

    def to_dict(self) -> config_dict:
        return self.__dict__


@dataclass
class FNNConfig(ModelConfig):
    hidden_layers: list[int]


@dataclass
class ConvolutionConfig(BaseConfig):
    n_channels: int
    kernel_size: int
    padding: int


@dataclass
class MinimalCNNConfig(ModelConfig):
    convolution: ConvolutionConfig
    hidden_layers: list[int]
    cnn_dropout: float = 0.2

    def to_dict(self) -> dict[str, Union[config_item, config_dict]]:
        return {
            "architecture": self.architecture,
            "embedding_dim": self.embedding_dim,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "dropout": self.dropout,
            "convolution": self.convolution.to_dict(),
            "hidden_layers": self.hidden_layers,
            "cnn_dropout": self.cnn_dropout
        }


@dataclass
class CombinedCNNConfig(ModelConfig):
    convolution: ConvolutionConfig
    cnn_hidden_layers: list[int]
    fnn_hidden_layers: list[int]
    shared_hidden_layers: list[int]
    cnn_dropout: float
    fnn_dropout: float

    def to_dict(self) -> dict[str, Union[config_item, config_dict]]:
        return {
            "architecture": self.architecture,
            "embedding_dim": self.embedding_dim,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "dropout": self.dropout,
            "convolution": self.convolution.to_dict(),
            "cnn_hidden_layers": self.cnn_hidden_layers,
            "fnn_hidden_layers": self.fnn_hidden_layers,
            "shared_hidden_layers": self.shared_hidden_layers,
            "cnn_dropout": self.cnn_dropout,
            "fnn_dropout": self.fnn_dropout
        }

@dataclass
class TrainConfig(BaseConfig):
    score: Score = "effect"
    optimizer: Optimizer = "adam"
    learning_rate: float = 0.001
    batch_size: int = 64
    loss_function: LossFunction = "mse"
    max_epochs: int = 100
    early_stopping: bool = False
    use_wandb: bool = False

@dataclass
class TrainingRunConfig(BaseConfig):
    run_config: RunConfig
    data_config: DataConfig
    model_config: ModelConfig
    train_config: TrainConfig

    def to_dict(self) -> dict[str, Union[config_dict, dict[str, Union[config_item, config_dict]]]]:
        return {
            "run": self.run_config.to_dict(),
            "data": self.data_config.to_dict(),
            "model": self.model_config.to_dict(),
            "training": self.train_config.to_dict()
        }
