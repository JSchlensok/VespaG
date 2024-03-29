from collections import defaultdict
from pathlib import Path
from typing import Union

import torch
import wandb
from torchtyping import TensorType

from src.vespa2.models import FNN, MinimalCNN, CombinedCNN
from src.vespa2.runner.mutations import SAV, Mutation
from src.vespa2.runner.type_hinting import ActivationFunction, LossFunction, Optimizer, Architecture, EmbeddingType

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"



# TODO set device based on if CUDA is available
def mask_non_mutations(gemme_prediction: TensorType["length", 20], wildtype_sequence) -> TensorType["length", 20]:
    """
    Simply set the predicted effect of the wildtype amino acid at each position (i.e. all non-mutations) to 0
    """
    gemme_prediction[
        torch.arange(len(wildtype_sequence)),
        torch.tensor([
            AMINO_ACIDS.index(aa)
            for aa in wildtype_sequence
        ])
    ] = 0.

    return gemme_prediction

def read_mutation_file(mutation_file: Path, one_indexed: bool=False) -> dict[str, list[SAV]]:
    mutations_per_protein = defaultdict(list)
    for line in mutation_file.open().readlines():
        protein_id, sav_string = line.split('_')
        mutations_per_protein[protein_id].append(SAV.from_sav_string(sav_string, one_indexed))

    return mutations_per_protein

def compute_mutation_score(y: TensorType["length", 20], mutation: Union[Mutation, SAV], alphabet: str=AMINO_ACIDS) -> float:
    if isinstance(mutation, Mutation):
        return sum([y[sav.position][alphabet.index(sav.to_aa)].item() for sav in mutation])
    else:
        return y[mutation.position][alphabet.index(mutation.to_aa)].item()


def load_activation_function(name: ActivationFunction) -> Union[torch.nn.Module, None]:
    if name is None:
        return None
    else:
        match name.lower():
            case "relu":
                return torch.nn.ReLU
            case "leaky_relu" | "leakyrelu":
                return torch.nn.LeakyReLU
            case "sigmoid":
                return torch.nn.Sigmoid


def load_loss_function(name: LossFunction):
    match name.lower():
        case "mse":
            return torch.nn.MSELoss
        case "bce":
            return torch.nn.BCELoss
        case "ce":
            return torch.nn.CrossEntropyLoss
        case "mae":
            return torch.nn.L1Loss


def load_optimizer(name: Optimizer) -> torch.optim.Optimizer:
    match name.lower():
        case "adam":
            return torch.optim.Adam
        case "adamw":
            return torch.optim.AdamW

def initialize_model(model_config, training_config, additional_feature_length: int = 0):
    input_dim = model_config["embedding_dim"] + additional_feature_length
    match model_config["architecture"]:
        case "fnn":
            model = FNN(
                hidden_layer_sizes=model_config["hidden_layers"],
                input_dim=input_dim,
                output_dim=20 if training_config["score"] == "effect" else 1,
                activation_function=load_activation_function(model_config["activation"]),
                output_activation_function=load_activation_function(model_config["output_activation"]),
                dropout_rate=model_config["dropout"]
            )
        case "cnn":
            model = MinimalCNN(
                input_dim=input_dim,
                output_dim=20 if training_config["score"] == "effect" else 1,
                kernel_size=model_config["convolution"]["kernel_size"],
                padding=model_config["convolution"]["padding"],
                n_channels=model_config["convolution"]["n_channels"],
                fnn_hidden_layers=model_config["hidden_layers"],
                activation_function=load_activation_function(model_config["activation"]),
                output_activation_function=load_activation_function(model_config["output_activation"]),
                fnn_dropout_rate=model_config["dropout"],
                cnn_dropout_rate=model_config["cnn_dropout"]
            )
        case "combined":
            model = CombinedCNN(
                input_dim=input_dim,
                output_dim=20 if training_config["score"] == "effect" else 1,
                n_channels=model_config["convolution"]["n_channels"],
                kernel_size=model_config["convolution"]["kernel_size"],
                padding=model_config["convolution"]["padding"],
                cnn_hidden_layers=model_config["cnn_hidden_layers"],
                fnn_hidden_layers=model_config["fnn_hidden_layers"],
                shared_hidden_layers=model_config["shared_hidden_layers"],
                activation_function=load_activation_function(model_config["activation"]),
                output_activation_function=load_activation_function(model_config["output_activation"]),
                shared_dropout_rate=model_config["dropout"],
                fnn_dropout_rate=model_config["fnn_dropout"],
                cnn_dropout_rate=model_config["cnn_dropout"]
            )

    return model


PATTERNS_TO_REPLACE = {
    "fnn": {"model.": "", "nn.": "net."},
    "cnn": {"model.": ""},
    "linreg": {"model.": ""},
    "combined": {"model.": ""}
}

BEST_RUNS = {
    "prott5": {"cnn": "bo1kzx2k", "fnn": "pk6g1da3"},
    "esm2": {"cnn": "g32cnixm", "fnn": "ilubgeqs"}
}

class ModelLoader:
    def __init__(self, wandb_user: str="jschlensok", project: str="vespa2"):
        self.wandb_user = wandb_user
        self.project = project

    @staticmethod
    def _replace_patterns_in_keys(state_dict, patterns: dict[str, str]):
        for pattern, substitution in patterns.items():
            state_dict = {key.replace(pattern, substitution): val for key, val in state_dict.items()}
        return state_dict

    def load_best(self, architecture: Architecture, embedding_type: EmbeddingType):
        run_id = BEST_RUNS[embedding_type][architecture]
        return self.load_from_run_id(run_id)

    def load_from_run_id(self, run_id: str) -> torch.nn.Module:
        api = wandb.Api()
        run = api.run(f"{self.wandb_user}/{self.project}/{run_id}")
        model_config = run.config["model"]
        model = initialize_model(run.config["model"], run.config["training"])

        artifact = api.artifact(f"{self.wandb_user}/{self.project}/model-{run_id}:v0")
        artifact_dir = artifact.download()
        checkpoint_path = next(Path(artifact_dir).glob("*.ckpt"))
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        raw_state_dict = checkpoint["state_dict"]
        state_dict = ModelLoader._replace_patterns_in_keys(raw_state_dict, PATTERNS_TO_REPLACE[model_config["architecture"]])

        model.load_state_dict(state_dict)

        return model


class MeanModel(torch.nn.Module):
    def __init__(self, *models: torch.nn.Module):
        super(MeanModel, self).__init__()
        self.models = list(models)

    def forward(self, x):
        return sum([model(x) for model in self.models]) / len(self.models)
