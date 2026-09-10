import pandas as pd
import unittest
from src.outcomes import build_outcomes


class OutcomeTests(unittest.TestCase):
    def test_manual_o1_to_close_horizons(self):
        idx = pd.bdate_range("2024-01-01", periods=7)
        out = build_outcomes(pd.Series([10, 20, 30, 40, 50, 60, 70], idx), pd.Series([11, 22, 33, 44, 55, 66, 77], idx), (1, 3, 5))
        self.assertAlmostEqual(out.iloc[0]["O1_C1"], 22 / 20 - 1)
        self.assertAlmostEqual(out.iloc[0]["O1_C3"], 44 / 20 - 1)
        self.assertAlmostEqual(out.iloc[0]["O1_C5"], 66 / 20 - 1)
