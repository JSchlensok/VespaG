from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union

class PrecisionType(str, Enum):
    half = "half"
    float = "float"


class Architecture(str, Enum):
    fnn = "fnn"
    cnn = "cnn"
    combined = "combined"
    mean = "mean"


class EmbeddingType(str, Enum):
    esm2 = "esm2"
    prott5 = "prott5"
