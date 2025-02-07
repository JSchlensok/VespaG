from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typeguard import typeguard_ignore


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


# https://docs.kidger.site/jaxtyping/faq/#is-jaxtyping-compatible-with-static-type-checkers-like-mypypyrightpytype
if TYPE_CHECKING:

    def typeguard_ignore() -> Callable[[Any], Any]:
        return typeguard.typeguard_ignore
else:

    def typeguard_ignore() -> Callable[[Any], Any]:
        return lambda func: func
