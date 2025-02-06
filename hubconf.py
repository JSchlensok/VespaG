from vespag.utils import DEFAULT_MODEL_PARAMETERS, load_model
from vespag.utils.type_hinting import EmbeddingType

dependencies = ["torch"]


def v2(embedding_type: EmbeddingType):
    params = DEFAULT_MODEL_PARAMETERS
    params["embedding_type"] = embedding_type
    return load_model(**params)
