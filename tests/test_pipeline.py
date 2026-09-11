import sys
import types
import unittest
from unittest.mock import patch

import pandas as pd

from src.pipeline import run


class PipelineTests(unittest.TestCase):
    def test_descriptive_results_are_not_added_to_fdr_universe(self):
        idx = pd.bdate_range("2024-01-01", periods=2)
        percentile = "margin_position_level__w126_percentile"
        features = pd.DataFrame({percentile: [1.0, 99.0]}, index=idx)
        features.attrs["catalog"] = {percentile: {"family": "margin_position_level", "k": 1, "window": 126}}
        outcomes = pd.DataFrame({"O1_C1": [.01, -.01]}, index=idx)
        primary = pd.DataFrame([{
            "predictor": percentile, "family": "margin_position_level", "k": 1,
            "rolling_window": 126, "pr_group": "PR0-5", "outcome_horizon": "O1_C1",
            "N": 1, "mean_return": .01, "median_return": .01, "win_rate": 1.0,
            "std": float("nan"), "q25": .01, "q75": .01,
            "effect_vs_unconditional": .01, "raw_p_value": .5,
        }])
        descriptive = pd.DataFrame([{"marker": "descriptive-only"}])
        calls = []
        fake_fdr = types.ModuleType("src.fdr")

        def apply(frame):
            calls.append(frame.copy())
            out = frame.copy(); out["evidence_level"] = "No Evidence"
            return out

        fake_fdr.apply_fdr = apply
        fake_fdr.fdr_diagnostics = lambda frame: pd.DataFrame()
        stage = {
            "datasets": {
                "open": pd.DataFrame({"0050": [10, 11]}, index=idx),
                "close": pd.DataFrame({"0050": [10, 11]}, index=idx),
                "margin_balance": pd.DataFrame({"2330": [1, 1]}, index=idx),
                "market_value": pd.DataFrame({"2330": [1, 1]}, index=idx),
            },
            "coverage": pd.DataFrame(), "reconciliation": pd.DataFrame(),
            "universe": pd.DataFrame(), "primary_symbols": {"2330"},
            "suspension": pd.DataFrame(),
            "suspension_validation": pd.DataFrame([{"SUSPENSION_COVERAGE_LIMITATION": False}]),
            "suspension_invalid_events": pd.DataFrame(), "annual_differences": [pd.DataFrame()],
            "base_series": {},
        }
        empty = pd.DataFrame()
        with (
            patch.dict(sys.modules, {"src.fdr": fake_fdr}),
            patch("src.pipeline.run_stage_zero", return_value=stage),
            patch("src.pipeline.build_features", return_value=features),
            patch("src.pipeline.build_outcomes", return_value=outcomes),
            patch("src.pipeline.level_feature_diagnostics", return_value=empty),
            patch("src.pipeline.run_primary_tests", side_effect=[primary.copy(), primary.copy()]),
            patch("src.pipeline.run_pr_bin_descriptive", return_value=descriptive),
            patch("src.pipeline.run_controlled_tests", return_value=empty),
            patch("src.pipeline.annual_results", return_value=empty),
            patch("src.pipeline.annual_robustness_summary", return_value=empty),
            patch("src.pipeline.neighborhood_consistency", return_value=empty),
            patch("src.pipeline.short_cover_variant_diagnostics", return_value=(empty, empty)),
            patch("src.pipeline.classify_pr_shapes", return_value=empty),
            patch("src.pipeline.regime_results", return_value=empty),
            patch("src.pipeline._build_base_series", return_value=({}, empty)),
            patch("src.pipeline.build_level_features", return_value=features),
            patch("src.pipeline.assess_signals", return_value=primary.assign(evidence_level="No Evidence")),
        ):
            result = run(export=False)
        self.assertEqual(len(calls), 1)
        pd.testing.assert_frame_equal(calls[0], primary)
        self.assertEqual(result["pr_bins"].iloc[0]["marker"], "descriptive-only")
        self.assertEqual(set(result["universe_sensitivity"]["universe"]), {"primary_common_equity", "all_available_margin_securities"})


if __name__ == "__main__":
    unittest.main()
