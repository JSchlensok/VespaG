import math
import unittest

import numpy as np
from scipy import stats

from vespag.utils.utils import ScoreNormalizer, transform_scores


class TestUtils(unittest.TestCase):
    def setUp(self):
        # 100 randomly sampled mutations from ProteinGym217
        self.raw_vespag_scores = np.loadtxt("tests/test_data/scores/raw_vespag_scores.csv", delimiter=",")
        self.dms_scores = np.loadtxt("tests/test_data/scores/dms_scores.csv", delimiter=",")

    def test_transform_scores(self) -> None:
        """Test whether score transformation leads to a change in correlation with experimental scores."""
        transformed_vespag_scores = transform_scores(self.raw_vespag_scores)
        spearman_raw = stats.spearmanr(self.dms_scores, self.raw_vespag_scores).statistic
        spearman_transformed = stats.spearmanr(self.dms_scores, transformed_vespag_scores).statistic
        assert math.isclose(spearman_raw, spearman_transformed, abs_tol=0.001)

    def test_normalize_scores(self) -> None:
        """Test whether score normalization leads to a change in correlation with experimental scores."""
        normalizer = ScoreNormalizer("minmax")
        normalizer.fit(self.raw_vespag_scores)
        normalized_vespag_scores = normalizer.normalize_scores(self.raw_vespag_scores)
        spearman_raw = stats.spearmanr(self.dms_scores, self.raw_vespag_scores).statistic
        spearman_normalized = stats.spearmanr(self.dms_scores, normalized_vespag_scores).statistic
        assert math.isclose(spearman_raw, spearman_normalized)
