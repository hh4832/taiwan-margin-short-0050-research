import pandas as pd
import unittest
from unittest.mock import patch
import numpy as np
from src.config import FINLAB_FIELDS
from src.outcomes import build_outcomes
from src.price_validation import raw_vs_adjusted_split_validation, split_artifact_observation_counts, validate_split_window


class OutcomeTests(unittest.TestCase):
    def test_outcome_fields_are_adjusted(self):
        self.assertEqual(FINLAB_FIELDS["open"], "etl:adj_open")
        self.assertEqual(FINLAB_FIELDS["close"], "etl:adj_close")

    def test_manual_o1_to_close_horizons(self):
        idx = pd.bdate_range("2024-01-01", periods=7)
        out = build_outcomes(pd.Series([10, 20, 30, 40, 50, 60, 70], idx), pd.Series([11, 22, 33, 44, 55, 66, 77], idx), (1, 3, 5))
        self.assertAlmostEqual(out.iloc[0]["O1_C1"], 22 / 20 - 1)
        self.assertAlmostEqual(out.iloc[0]["O1_C3"], 44 / 20 - 1)
        self.assertAlmostEqual(out.iloc[0]["O1_C5"], 66 / 20 - 1)

    def test_known_split_fixture_has_continuous_adjusted_outcomes(self):
        idx = pd.bdate_range("2025-05-20", "2025-06-20")
        adjusted_open = pd.Series(np.linspace(360, 368, len(idx)), idx)
        adjusted_close = adjusted_open + 0.5
        outcomes = build_outcomes(adjusted_open, adjusted_close, (1, 2, 3, 5, 10, 20))
        diagnostic = validate_split_window(adjusted_open, adjusted_close, outcomes)
        self.assertTrue(diagnostic["status"].isin(["pass", "diagnostic", "not_testable"]).all())
        self.assertGreater(outcomes.loc["2025-06-03", "O1_C10"], -0.50)
        self.assertGreater(outcomes.loc["2025-05-21", "O1_C20"], -0.50)

    def test_raw_vs_adjusted_diagnostic_detects_split_artifact(self):
        idx = pd.to_datetime(["2025-06-10", "2025-06-18"])
        raw = pd.Series([188.65, 47.57], idx)
        adjusted = pd.Series([363.500896, 366.641667], idx)
        result = raw_vs_adjusted_split_validation(raw, adjusted)
        self.assertAlmostEqual(result.iloc[0]["return"], 47.57 / 188.65 - 1)
        self.assertAlmostEqual(result.iloc[1]["return"], 366.641667 / 363.500896 - 1)

    def test_mapping_guard_fails_on_raw_fields(self):
        idx = pd.bdate_range("2025-05-20", "2025-06-20")
        prices = pd.Series(np.linspace(360, 368, len(idx)), idx)
        outcomes = build_outcomes(prices, prices, (1, 2, 3, 5, 10, 20))
        with patch.dict(FINLAB_FIELDS, {"open": "price:開盤價", "close": "price:收盤價"}):
            with self.assertRaises(RuntimeError):
                validate_split_window(prices, prices, outcomes)

    def test_split_artifact_observations_are_counted_by_horizon(self):
        idx = pd.bdate_range("2025-05-20", "2025-06-20")
        adjusted_open = pd.Series(np.linspace(360, 368, len(idx)), idx)
        adjusted_close = adjusted_open + .5
        raw_open = adjusted_open.copy()
        raw_close = adjusted_close.copy()
        raw_open.loc["2025-06-18":] /= 4
        raw_close.loc["2025-06-18":] /= 4
        impact = split_artifact_observation_counts(raw_open, raw_close, adjusted_open, adjusted_close)
        self.assertGreater(impact["split_artifact_observations"].sum(), 0)
        c1 = impact.loc[impact["outcome_horizon"].eq("O1_C1"), "split_artifact_observations"].iloc[0]
        self.assertEqual(c1, 0)
