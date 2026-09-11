import numpy as np
import pandas as pd
import unittest
from types import SimpleNamespace

from src.features import build_level_features, level_feature_diagnostics, position_market_value, pr_group, rolling_ratio, trailing_percentile


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

    def test_sparse_observations_use_full_valid_window_without_filling(self):
        base = pd.Series([3.0, np.nan, 2.0, 1.0, np.nan, 4.0])
        result = trailing_percentile(base, 3)
        self.assertTrue(pd.isna(result.iloc[1]))
        self.assertTrue(pd.isna(result.iloc[4]))
        self.assertEqual(result.first_valid_index(), 3)
        changed_future = base.copy(); changed_future.iloc[-1] = -100
        self.assertEqual(result.iloc[3], trailing_percentile(changed_future, 3).iloc[3])

    def test_position_value_aligns_dates_without_forward_fill(self):
        balance = pd.DataFrame({"2330": [1.0, 2.0]}, index=pd.to_datetime(["2024-01-01", "2024-01-03"]))
        close = pd.DataFrame({"2330": [100.0, 300.0]}, index=pd.to_datetime(["2024-01-01", "2024-01-03"]))
        result = position_market_value(balance, close, {"2330"})
        self.assertEqual(result.index.tolist(), balance.index.tolist())
        self.assertNotIn(pd.Timestamp("2024-01-02"), result.index)
        self.assertEqual(result.tolist(), [100_000.0, 600_000.0])

    def test_level_predictor_has_both_tails_on_valid_synthetic_data(self):
        idx = pd.bdate_range("2024-01-01", periods=60)
        values = pd.Series(np.tile(np.arange(20, 0, -1), 3), index=idx, dtype=float)
        base = {name: values.copy() for name in (
            "margin_position_level", "margin_credit_level", "approx_margin_maintenance",
            "short_position_level", "short_position_level_adjusted", "short_margin_ratio",
        )}
        features = build_level_features(base, SimpleNamespace(rolling_windows=(20,)))
        feature = "margin_position_level__w20_percentile"
        self.assertGreater(features[feature].notna().sum(), 0)
        self.assertGreater(features[feature].le(5).sum(), 0)
        self.assertGreater(features[feature].ge(95).sum(), 0)

        base.update({
            "margin_position_market_value": values * 2,
            "short_position_market_value": values,
            "short_position_market_value_adjusted": values,
            "margin_credit_amount": values * 3,
            "market_cap": values * 10,
        })
        outcomes = pd.DataFrame({"O1_C1": .01}, index=idx)
        diagnostics = level_feature_diagnostics(base, features, outcomes).set_index("feature")
        self.assertEqual(diagnostics.loc[feature, "raw_non_null_n"], 60)
        self.assertGreater(diagnostics.loc[feature, "outcome_common_date_n"], 0)
        self.assertEqual(diagnostics.loc[feature, "status"], "ok")

    def test_descriptive_bin_boundaries_are_non_overlapping(self):
        values = pd.Series([0, 5, 20, 40, 60, 80, 95, 100], dtype=float)
        self.assertEqual(list(pr_group(values).astype(str)), [
            "PR0-5", "PR5-20", "PR20-40", "PR40-60", "PR60-80",
            "PR80-95", "PR95-100", "PR95-100",
        ])
