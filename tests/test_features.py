import numpy as np
import pandas as pd
import unittest
from src.features import rolling_ratio, trailing_percentile


class FeatureTests(unittest.TestCase):
    def test_kday_ratio_is_ratio_of_sums(self):
        n = pd.Series([1.0, 9.0]); d = pd.Series([10.0, 90.0])
        self.assertAlmostEqual(rolling_ratio(n, d, 2).iloc[-1], 0.1)

    def test_percentile_does_not_use_future(self):
        base = pd.Series([3.0, 1.0, 2.0, 100.0])
        original = trailing_percentile(base, 3)
        changed = trailing_percentile(pd.Series([3.0, 1.0, 2.0, -100.0]), 3)
        self.assertEqual(original.iloc[2], changed.iloc[2])
        self.assertTrue(np.isnan(original.iloc[1]))
