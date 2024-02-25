from typing import List

import torch
from jaxtyping import Float

from src.vespa2.models.utils import construct_fnn


class FNN(torch.nn.Module):
    """
    Fully-connected neural network with arbitrary hidden layers and activation functions

    Attributes:
        input_dim: Size of the input vectors (e.g. 1024 for ProtT5 embeddings, 1536 for ANKH embeddings). Default: 1024
        output_dim: Size of the output vector (e.g. 1 for conservation prediction, 20 for GEMME scores). Default: 20
        activation_function: Activation function to use for the hidden layers. Default: LeakyReLU
        output_activation_function: Activation function to use for the output layer, e.g. None for linear regression,
            Sigmoid for logistic regression. Default: None
        dropout_rate: Dropout rate to apply after every layer, if desired. Default: None

    Examples:
        gemme_prott5_linear_regression = FNN([], 1024, 20, None, None)
        gemme_conservation_linear_regression = FNN([], 1024, 1, None, torch.nn.Sigmoid)
        gemme_prott5_fnn_2_layers = FNN([256, 64], 1024, 20, torch.nn.LeakyReLU, None)
    """

    def __init__(
            self,
            hidden_layer_sizes: List[int],
            input_dim: int = 1024,
            output_dim: int = 20,
            activation_function: torch.nn.Module = torch.nn.LeakyReLU,
            output_activation_function: torch.nn.Module = None,
            dropout_rate: float = None,
    ):

        super(FNN, self).__init__()
        self.net = construct_fnn(
            hidden_layer_sizes,
            input_dim,
            output_dim,
            activation_function,
            output_activation_function,
            dropout_rate,
        )
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight.data, a=1e-2)
                torch.nn.init.zeros_(layer.bias.data)

    def forward(
            self, X: Float[torch.Tensor, "batch_size length input_dim"]
    ) -> Float[torch.Tensor, "batch_size length output_dim"]:
        return self.net(X).squeeze(-1)
