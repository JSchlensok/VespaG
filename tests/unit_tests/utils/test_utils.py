import math

import pytest
import torch
import torch.testing
from scipy import stats

from vespag.utils.mutations import mask_non_mutations
from vespag.utils.utils import GEMME_ALPHABET, ScoreNormalizer, transform_scores


@pytest.mark.parametrize("embedding_type", ["esm2", "prott5"])
def test_transform_scores(dms_vespag_scores, dms_experimental_scores, embedding_type) -> None:
    """Test whether score transformation leads to a change in correlation with experimental scores."""
    transformed_vespag_scores = transform_scores(dms_vespag_scores[embedding_type], embedding_type=embedding_type)
    spearman_raw = stats.spearmanr(dms_experimental_scores, dms_vespag_scores[embedding_type]).statistic
    spearman_transformed = stats.spearmanr(dms_experimental_scores, transformed_vespag_scores).statistic
    assert math.isclose(spearman_raw, spearman_transformed, abs_tol=0.001)

@pytest.mark.parametrize("embedding_type", ["esm2", "prott5"])
def test_normalize_scores(dms_vespag_scores, dms_experimental_scores, embedding_type) -> None:
    """Test whether score normalization leads to a change in correlation with experimental scores."""
    normalizer = ScoreNormalizer("minmax")
    normalizer.fit(dms_vespag_scores[embedding_type])
    normalized_vespag_scores = normalizer.normalize_scores(dms_vespag_scores[embedding_type])
    spearman_raw = stats.spearmanr(dms_experimental_scores, dms_vespag_scores[embedding_type]).statistic
    spearman_normalized = stats.spearmanr(dms_experimental_scores, normalized_vespag_scores).statistic
    assert math.isclose(spearman_raw, spearman_normalized)

def test_mask_non_mutations() -> None:
    """Test whether non-mutations are correctly masked."""
    pred = torch.rand(3, 20)
    wildtype_sequence = "SEQ"
    pred = mask_non_mutations(pred, wildtype_sequence)
    torch.testing.assert_close(pred[list(range(len(wildtype_sequence))), [GEMME_ALPHABET.index(aa) for aa in wildtype_sequence]], torch.zeros(3))


def test_mask_unknown_residues() -> None:
    """Test whether wildtype residue scores are masked."""
    pred = torch.rand(4, 20)
    wildtype_sequence = "SEQX"
    pred = mask_non_mutations(pred, wildtype_sequence)
    torch.testing.assert_close(pred, mask_non_mutations(pred, wildtype_sequence))