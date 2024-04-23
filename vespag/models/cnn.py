import torch
import torch.nn.functional as F

from .utils import construct_fnn

"""
batch_size x L x 1536
- transform ->
batch_size x 1536 x L x 1
"""

class MinimalCNN(torch.nn.Module):
    """
    Just a 1D convolution followed by two dense layers, akin to biotrainer's offering
    """
    def __init__(self,
                 input_dim: int = 1024,
                 output_dim: int = 20,
                 n_channels: int = 256,
                 kernel_size=7,
                 padding=3,
                 fnn_hidden_layers: list[int] = [256, 64],
                 activation_function: torch.nn.Module = torch.nn.LeakyReLU,
                 output_activation_function: torch.nn.Module = None,
                 cnn_dropout_rate: float = None,
                 fnn_dropout_rate: float = None
    ):
        super(MinimalCNN, self).__init__()
        conv_layers = [
            torch.nn.Conv1d(input_dim, n_channels, kernel_size=kernel_size, padding=padding),
            activation_function()
        ]

        if cnn_dropout_rate:
            conv_layers.append(torch.nn.Dropout(cnn_dropout_rate))

        self.conv = torch.nn.Sequential(*conv_layers)

        self.fnn = construct_fnn(fnn_hidden_layers, n_channels, output_dim, activation_function, output_activation_function, fnn_dropout_rate)

    def forward(self, X):
        X = X.movedim(-1, -2)
        X = self.conv(X)
        X = X.movedim(-1, -2)
        X = self.fnn(X)
        return X.squeeze(-1)


class CombinedCNN(torch.nn.Module):
    # TODO parametrize (CNN parameters, FNN parameters, shared FNN parameters)
    """
    Parallel FNN and CNN whose outputs are concatenated and again fed through dense layers
    """
    def __init__(self,
                 input_dim: int = 1024,
                 output_dim: int = 20,
                 n_channels: int = 256,
                 kernel_size=7,
                 padding=3,
                 cnn_hidden_layers: list[int] = [64],
                 fnn_hidden_layers: list[int] = [256, 64],
                 shared_hidden_layers: list[int] = [64],
                 activation_function: torch.nn.Module = torch.nn.LeakyReLU,
                 output_activation_function: torch.nn.Module = None,
                 shared_dropout_rate: float = None,
                 cnn_dropout_rate: float = None,
                 fnn_dropout_rate: float = None
                 ):
        super(CombinedCNN, self).__init__()
        self.conv = MinimalCNN(
            input_dim=input_dim,
            output_dim=cnn_hidden_layers[-1],
            n_channels=n_channels,
            kernel_size=kernel_size,
            padding=padding,
            fnn_hidden_layers=cnn_hidden_layers[:-1],
            activation_function=activation_function,
            output_activation_function=activation_function,
            cnn_dropout_rate=cnn_dropout_rate,
            fnn_dropout_rate=fnn_dropout_rate,
        )
        self.fnn = construct_fnn(
            hidden_layer_sizes=fnn_hidden_layers[:-1],
            input_dim=input_dim,
            output_dim=fnn_hidden_layers[-1],
            activation_function=activation_function,
            output_activation_function=activation_function,
            dropout_rate=fnn_dropout_rate
        )
        self.combined = construct_fnn(
            hidden_layer_sizes=shared_hidden_layers,
            input_dim=cnn_hidden_layers[-1] + fnn_hidden_layers[-1],
            output_dim=output_dim,
            activation_function=activation_function,
            output_activation_function=output_activation_function,
            dropout_rate=shared_dropout_rate
        )

    def forward(self, X):
        X_combined = torch.cat([self.conv(X), self.fnn(X)], dim=-1)
        pred = self.combined(X_combined)
        return pred.squeeze(-1)
