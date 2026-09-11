import unittest
from unittest.mock import patch
import importlib.util
import warnings

import numpy as np
import pandas as pd

from src.fdr import fdr_diagnostics
from src.statistics import (
    annual_results,
    annual_robustness_summary,
    classify_pr_shapes,
    neighborhood_consistency,
    run_controlled_tests,
    run_pr_bin_descriptive,
    short_cover_variant_diagnostics,
    controlled_regression,
    controlled_regression_diagnostics,
)


class FakeInfluence:
    def __init__(self, leverage):
        self.hat_matrix_diag = np.asarray(leverage, dtype=float)


class FakeRobustFit:
    def __init__(self, beta=.2, se=.05, p=.01, covariance=None):
        self.params = np.array([0.0, beta, 0.1])
        self.bse = np.array([0.01, se, 0.02])
        self.pvalues = np.array([1.0, p, .5])
        self._covariance = np.eye(3) if covariance is None else covariance

    def cov_params(self):
        return self._covariance


class FakeBaseFit:
    def __init__(self, leverage, robust=None):
        self.params = pd.Series({"const": 0.0, "signal": .2, "prior": .1})
        self._leverage = leverage
        self._robust = robust or FakeRobustFit()
        self.robust_called = False

    def get_influence(self):
        return FakeInfluence(self._leverage)

    def get_robustcov_results(self, cov_type):
        self.robust_called = True
        if cov_type != "HC3":
            raise AssertionError("controlled regression must preserve HC3")
        return self._robust


def balanced_inputs(n=80):
    signal = pd.Series(np.r_[np.zeros(n // 2), np.ones(n - n // 2)])
    prior = pd.Series(np.linspace(-1, 1, n))
    future = .01 + .02 * signal + .03 * prior
    return signal, future, prior


class StatisticsTests(unittest.TestCase):
    def test_normal_hc3_regression_has_finite_outputs(self):
        signal, future, prior = balanced_inputs()
        fake = FakeBaseFit(np.repeat(3 / len(signal), len(signal)))
        with patch("src.statistics._fit_base_ols", return_value=fake):
            result = controlled_regression(signal, future, prior)
        self.assertEqual(result["status"], "estimated")
        self.assertTrue(result["is_interpretable"])
        self.assertTrue(fake.robust_called)
        self.assertTrue(np.isfinite([result["signal_beta"], result["signal_se"], result["signal_p_value"]]).all())

    @unittest.skipUnless(importlib.util.find_spec("statsmodels"), "statsmodels is not installed in this local runtime")
    def test_real_statsmodels_hc3_integration(self):
        signal, _, prior = balanced_inputs(100)
        rng = np.random.default_rng(7)
        future = .01 + .02 * signal + .03 * prior + pd.Series(rng.normal(0, .01, len(signal)))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = controlled_regression(signal, future, prior)
        self.assertEqual(result["status"], "estimated")
        self.assertTrue(np.isfinite([result["signal_beta"], result["signal_se"], result["signal_p_value"]]).all())

    def test_controlled_insufficient_sample_status_is_preserved(self):
        result = controlled_regression(pd.Series([0.0, 1.0]), pd.Series([.1, .2]), pd.Series([0.0, .1]))
        self.assertEqual(result["status"], "insufficient_sample")

    def test_insufficient_tail_and_control_groups(self):
        prior = pd.Series(np.linspace(-1, 1, 60))
        future = pd.Series(np.linspace(0, .1, 60))
        tail_small = controlled_regression(pd.Series([1.0] * 2 + [0.0] * 58), future, prior)
        control_small = controlled_regression(pd.Series([1.0] * 58 + [0.0] * 2), future, prior)
        self.assertEqual(tail_small["status"], "insufficient_group_sample")
        self.assertEqual(control_small["status"], "insufficient_group_sample")
        self.assertEqual((tail_small["signal_n"], tail_small["control_n"]), (2, 58))
        self.assertTrue(pd.isna(tail_small["signal_p_value"]))

    def test_variation_and_rank_checks_precede_hc3(self):
        _, future, prior = balanced_inputs()
        no_signal = controlled_regression(pd.Series(np.zeros(80)), future, prior)
        no_prior = controlled_regression(pd.Series(np.tile([0.0, 1.0], 40)), future, pd.Series(np.ones(80)))
        signal = pd.Series(np.tile([0.0, 1.0], 40))
        rank_deficient = controlled_regression(signal, future, signal)
        almost_collinear = controlled_regression(signal, future, signal + np.linspace(0, 1e-14, 80))
        self.assertEqual(no_signal["status"], "no_signal_variation")
        self.assertEqual(no_prior["status"], "no_prior_variation")
        self.assertEqual(rank_deficient["status"], "rank_deficient")
        self.assertEqual(almost_collinear["status"], "rank_deficient")

    def test_high_leverage_is_stopped_before_hc3_without_runtime_warning(self):
        signal, future, prior = balanced_inputs()
        leverage = np.repeat(3 / len(signal), len(signal)); leverage[-1] = 1.0
        fake = FakeBaseFit(leverage)
        with warnings.catch_warnings(record=True) as caught, patch("src.statistics._fit_base_ols", return_value=fake):
            warnings.simplefilter("always")
            result = controlled_regression(signal, future, prior)
        self.assertEqual(result["status"], "high_leverage_unstable")
        self.assertEqual(result["interpretation"], "無法判定")
        self.assertFalse(fake.robust_called)
        self.assertTrue(pd.isna(result["signal_p_value"]))
        self.assertEqual(result["high_leverage_n"], 1)
        self.assertFalse(any(issubclass(item.category, RuntimeWarning) for item in caught))

    def test_hc3_nonfinite_is_retained_as_unestimable(self):
        signal, future, prior = balanced_inputs()
        robust = FakeRobustFit(se=np.inf, p=np.nan, covariance=np.full((3, 3), np.inf))
        fake = FakeBaseFit(np.repeat(3 / len(signal), len(signal)), robust)
        with patch("src.statistics._fit_base_ols", return_value=fake):
            result = controlled_regression(signal, future, prior)
        self.assertEqual(result["status"], "hc3_nonfinite")
        self.assertTrue(pd.isna(result["signal_p_value"]))

    def test_missing_data_reduces_effective_n_before_diagnostics(self):
        signal, future, prior = balanced_inputs(50)
        future.iloc[:15] = np.nan
        result = controlled_regression(signal, future, prior, min_total_n=40, min_tail_n=1, min_control_n=1)
        self.assertEqual(result["N"], 35)
        self.assertEqual(result["status"], "insufficient_sample")

    def test_controlled_diagnostics_summarize_failures_by_cell(self):
        results = pd.DataFrame([
            {"family": "a", "pr_group": "PR0-5", "prior_k": 1, "outcome_horizon": "O1_C1", "status": "estimated"},
            {"family": "a", "pr_group": "PR0-5", "prior_k": 1, "outcome_horizon": "O1_C1", "status": "high_leverage_unstable"},
        ])
        diagnostics = controlled_regression_diagnostics(results)
        overall = diagnostics[diagnostics["scope"].eq("overall")].iloc[0]
        self.assertEqual(overall["total_models"], 2)
        self.assertEqual(overall["estimated_models"], 1)
        self.assertEqual(overall["high_leverage_unstable_models"], 1)

    def test_fdr_diagnostics_counts_only_inferential_rows(self):
        inferential = pd.DataFrame({
            "family": ["a", "a", "b"], "raw_p_value": [.01, np.nan, .2],
            "evidence_level": ["Level A", "No Evidence", "No Evidence"],
        })
        diagnostics = fdr_diagnostics(inferential)
        total = diagnostics.query("metric == 'number_of_tests_total'").iloc[0]["count"]
        self.assertEqual(total, 2)
        self.assertNotIn("pr_group", diagnostics.columns)

    def test_controlled_low_and_high_tails_are_separate_and_missing_stays_missing(self):
        idx = pd.bdate_range("2024-01-01", periods=30)
        feature = "margin_position_level__w126_percentile"
        features = pd.DataFrame({feature: [1.0, 99.0, np.nan] + [50.0] * 27}, index=idx)
        features.attrs["catalog"] = {feature: {"family": "margin_position_level", "k": 1, "window": 126}}
        outcomes = pd.DataFrame({"O1_C1": np.arange(30, dtype=float)}, index=idx)
        calls = []

        def capture(signal, future, prior, **kwargs):
            calls.append(signal.copy())
            return {"N": 27, "signal_beta": 0.0, "signal_p_value": 1.0, "status": "estimated"}

        with patch("src.statistics.controlled_regression", side_effect=capture):
            result = run_controlled_tests(features, outcomes, pd.Series(np.arange(1, 31), index=idx))
        self.assertEqual(set(result["pr_group"]), {"PR0-5", "PR95-100"})
        low, high = calls[0], calls[4]
        self.assertEqual(low.iloc[0], 1.0); self.assertEqual(low.iloc[1], 0.0)
        self.assertEqual(high.iloc[0], 0.0); self.assertEqual(high.iloc[1], 1.0)
        self.assertTrue(pd.isna(low.iloc[2]) and pd.isna(high.iloc[2]))

    def test_descriptive_has_seven_bins_and_no_inferential_p_values(self):
        idx = pd.bdate_range("2024-01-01", periods=8)
        feature = "short_margin_ratio__w126_percentile"
        features = pd.DataFrame({feature: [0, 5, 20, 40, 60, 80, 95, 100]}, index=idx)
        features.attrs["catalog"] = {feature: {"family": "short_margin_ratio", "k": 1, "window": 126}}
        outcomes = pd.DataFrame({"O1_C1": np.arange(8) / 100}, index=idx)
        result = run_pr_bin_descriptive(features, outcomes)
        self.assertEqual(result["pr_group"].nunique(), 7)
        self.assertNotIn("raw_p_value", result.columns)
        self.assertEqual(result["N"].sum(), 8)

    def test_annual_candidates_are_all_a_b_not_raw_p_top_30(self):
        idx = pd.to_datetime(["2023-01-02", "2024-01-02"])
        features = pd.DataFrame({"a": [1, 1], "b": [99, 99], "c": [1, 1]}, index=idx)
        outcomes = pd.DataFrame({"O1_C1": [.1, -.1]}, index=idx)
        results = pd.DataFrame([
            {"predictor": "a", "family": "fa", "k": 1, "rolling_window": 126, "pr_group": "PR0-5", "outcome_horizon": "O1_C1", "raw_p_value": .9, "evidence_level": "Level A"},
            {"predictor": "b", "family": "fb", "k": 1, "rolling_window": 126, "pr_group": "PR95-100", "outcome_horizon": "O1_C1", "raw_p_value": .8, "evidence_level": "Level B"},
            {"predictor": "c", "family": "fc", "k": 1, "rolling_window": 126, "pr_group": "PR0-5", "outcome_horizon": "O1_C1", "raw_p_value": .001, "evidence_level": "Level C"},
        ])
        annual = annual_results(results, features, outcomes)
        self.assertEqual(set(annual["predictor"]), {"a", "b"})
        summary = annual_robustness_summary(annual)
        self.assertEqual(set(summary["predictor"]), {"a", "b"})
        self.assertIn("largest_year_sample_share", summary)

    def test_neighborhood_has_refined_dimensions(self):
        results = pd.DataFrame([
            {"family": "f", "pr_group": "PR0-5", "k": k, "rolling_window": w, "outcome_horizon": h, "effect_vs_unconditional": effect}
            for k, w, h, effect in [(1, 126, "O1_C1", 1), (3, 126, "O1_C1", 1), (1, 252, "O1_C5", -1)]
        ])
        result = neighborhood_consistency(results)
        self.assertTrue({"overall", "fixed_horizon", "fixed_k", "fixed_window"}.issubset(set(result["consistency_type"])))
        self.assertTrue({"n_cells", "positive_cells", "negative_cells", "consistency_score"}.issubset(result.columns))

    def test_short_cover_variant_conflict_is_detected(self):
        base = {"family": "short_cover", "k": 1, "rolling_window": 126, "pr_group": "PR95-100", "outcome_horizon": "O1_C1", "evidence_level": "Level A"}
        rows = [
            base | {"predictor": "short_cover_raw__k1__raw__w126_percentile", "effect_vs_unconditional": .01},
            base | {"predictor": "short_cover_raw__k1__volume_ratio__w126_percentile", "effect_vs_unconditional": -.01},
            base | {"predictor": "short_cover_adjusted__k1__raw__w126_percentile", "effect_vs_unconditional": .02},
            base | {"predictor": "short_cover_adjusted__k1__volume_ratio__w126_percentile", "effect_vs_unconditional": .01},
        ]
        detail, consistency = short_cover_variant_diagnostics(pd.DataFrame(rows))
        self.assertEqual(set(detail["variant"]), {"raw_sign", "normalized_sign", "adjusted_raw_sign", "adjusted_normalized_sign"})
        self.assertTrue(consistency.iloc[0]["conflict_flag"])

    def test_maintenance_shape_classification(self):
        means = [3, 2, 1, 0, 1, 2, 3]
        bins = ["PR0-5", "PR5-20", "PR20-40", "PR40-60", "PR60-80", "PR80-95", "PR95-100"]
        frame = pd.DataFrame({
            "predictor": "approx_margin_maintenance__w126_percentile", "family": "approx_margin_maintenance",
            "k": 1, "rolling_window": 126, "outcome_horizon": "O1_C1", "pr_group": bins,
            "mean_return": means,
        })
        self.assertEqual(classify_pr_shapes(frame).iloc[0]["shape_classification"], "U_shape")


if __name__ == "__main__":
    unittest.main()
