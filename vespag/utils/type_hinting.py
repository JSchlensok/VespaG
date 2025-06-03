import typing
from typing import Literal

import click

Architecture = Literal["fnn", "cnn", "combined", "mean"]
EmbeddingType = Literal["esm2", "prott5"]
Precision = Literal["half", "float"]


# https://github.com/fastapi/typer/pull/429#issuecomment-2780948253
ClickArchitectureType = click.Choice(typing.get_args(Architecture))
ClickEmbeddingType = click.Choice(typing.get_args(EmbeddingType))
ClickPrecisionType = click.Choice(typing.get_args(Precision))