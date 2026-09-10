import pandas as pd
import unittest
from src.outcomes import build_outcomes


class DateAlignmentTests(unittest.TestCase):
    def test_signal_maps_to_next_actual_trading_day(self):
        idx = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])
        out = build_outcomes(pd.Series([10, 20, 30], idx), pd.Series([11, 22, 33], idx), (1,))
        self.assertEqual(out.loc["2024-01-05", "entry_date"], pd.Timestamp("2024-01-08"))
        self.assertAlmostEqual(out.loc["2024-01-05", "O1_C1"], 0.1)
