import json
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.margin_composition import (
    BASELINE_COMMIT,
    FDR_SCOPE,
    TURNOVER_BASE_COMMIT,
    MarginCompositionConfig,
    _classify_absorption,
    _composition_group,
    _fit_nested_models,
    _low_turnover_clusters,
    build_margin_composition_features,
    load_frozen_turnover,
    margin_composition_kday,
    run_composition_primary,
    run_low_turnover_composition,
    run_margin_composition_study,
    select_composition_absorption_candidates,
)


def datasets(periods=20):
    index = pd.bdate_range("2020-01-01", periods=periods)
    buy = pd.DataFrame({
        "上市融資交易金額": np.linspace(10, 50, periods),
        "上櫃融資交易金額": np.linspace(2, 4, periods),
    }, index=index)
    sell = pd.DataFrame({
        "上市融資交易金額": np.linspace(40, 5, periods),
        "上櫃融資交易金額": np.linspace(4, 2, periods),
    }, index=index)
    return {"aggregate_buy": buy, "aggregate_sell": sell}


def full_datasets(periods=20):
    result = datasets(periods)
    index = result["aggregate_buy"].index
    result.update({
        "open": pd.DataFrame({"0050": np.linspace(99, 110, periods)}, index=index),
        "close": pd.DataFrame({"0050": np.linspace(100, 111, periods)}, index=index),
        "margin_buy": pd.DataFrame({"2330": np.linspace(1, 2, periods)}, index=index),
        "margin_sell": pd.DataFrame({"2330": np.linspace(2, 1, periods)}, index=index),
        "margin_balance": pd.DataFrame({"2330": 1.}, index=index),
        "market_volume": pd.DataFrame({"TAIEX": 1000., "OTC": 500.}, index=index),
        "market_amount": pd.DataFrame({"TAIEX": 10000., "OTC": 5000.}, index=index),
        "market_value": pd.DataFrame({"2330": 100.}, index=index),
    })
    return result


def frozen_dirs(root: Path):
    baseline = root / "baseline"
    turnover = root / "turnover"
    baseline.mkdir(); turnover.mkdir()
    (baseline / "run_info.txt").write_text(f"git_commit={BASELINE_COMMIT}\n", encoding="utf-8")
    for name in ("fdr_results.csv", "controlled_results.csv", "annual_robustness_summary.csv", "neighborhood_consistency.csv"):
        pd.DataFrame({"family": []}).to_csv(baseline / name, index=False)
    (turnover / "run_info_turnover.txt").write_text(
        f"git_commit={TURNOVER_BASE_COMMIT}\nbaseline_commit={BASELINE_COMMIT}\n", encoding="utf-8"
    )
    for name in (
        "turnover_primary_results.csv", "turnover_fdr_results.csv", "turnover_annual_results.csv",
        "turnover_annual_robustness_summary.csv", "turnover_absorption_results.csv",
    ):
        columns = {"status": [], "original_family": [], "original_predictor": [], "turnover_variant": []} if name == "turnover_absorption_results.csv" else {"x": []}
        pd.DataFrame(columns).to_csv(turnover / name, index=False)
    return baseline, turnover


class FakeInfluence:
    def __init__(self, n):
        self.hat_matrix_diag = np.repeat(.05, n)


class FakeRobust:
    def __init__(self, size, signal=.2):
        self.params = np.arange(size, dtype=float) / 10
        self.params[1] = signal
        self.pvalues = np.repeat(.01, size)
        self._size = size

    def cov_params(self):
        return np.eye(self._size)


class FakeFit:
    def __init__(self, n, size, signal=.2):
        self.n, self.size, self.signal = n, size, signal

    def get_influence(self):
        return FakeInfluence(self.n)

    def get_robustcov_results(self, cov_type):
        if cov_type != "HC3":
            raise AssertionError("HC3 required")
        return FakeRobust(self.size, self.signal)


class MarginCompositionTests(unittest.TestCase):
    def test_ratio_uses_sum_then_divide(self):
        buy = pd.Series([1.0, 9.0, 1.0])
        sell = pd.Series([9.0, 1.0, 1.0])
        result = margin_composition_kday(buy, sell, 2)
        self.assertAlmostEqual(result.loc[1, "buy_share"], .5)

    def test_imbalance_identity(self):
        raw = margin_composition_kday(pd.Series([2., 4.]), pd.Series([6., 2.]), 1)
        np.testing.assert_allclose(raw["imbalance"], 2 * raw["buy_share"] - 1)

    def test_zero_denominator_is_nan(self):
        raw = margin_composition_kday(pd.Series([0.]), pd.Series([0.]), 1)
        self.assertTrue(pd.isna(raw.loc[0, "buy_share"]))

    def test_negative_denominator_is_nan(self):
        raw = margin_composition_kday(pd.Series([-2.]), pd.Series([1.]), 1)
        self.assertTrue(pd.isna(raw.loc[0, "imbalance"]))

    def test_no_fill_for_missing_amount(self):
        raw = margin_composition_kday(pd.Series([1., np.nan, 2.]), pd.Series([1., 1., 1.]), 1)
        self.assertTrue(pd.isna(raw.loc[1, "buy_share"]))

    def test_kday_requires_full_window(self):
        raw = margin_composition_kday(pd.Series([1., 2., 3.]), pd.Series([1., 2., 3.]), 3)
        self.assertTrue(raw["buy_share"].iloc[:2].isna().all())

    def test_kday_is_not_mean_of_daily_ratios(self):
        buy, sell = pd.Series([1., 9.]), pd.Series([1., 1.])
        result = margin_composition_kday(buy, sell, 2)["buy_share"].iloc[-1]
        daily_mean = (buy / (buy + sell)).mean()
        self.assertNotAlmostEqual(result, daily_mean)

    def test_trailing_percentile_does_not_use_future(self):
        config = MarginCompositionConfig(k_values=(1,), rolling_windows=(3,))
        first = datasets()
        changed = datasets()
        changed["aggregate_buy"].iloc[-1, :] = -1_000_000
        a, _, _ = build_margin_composition_features(first, config)
        b, _, _ = build_margin_composition_features(changed, config)
        pd.testing.assert_series_equal(a.iloc[:-1, 0], b.iloc[:-1, 0])

    def test_features_only_contain_buy_share(self):
        config = MarginCompositionConfig(k_values=(1,), rolling_windows=(3,))
        features, _, _ = build_margin_composition_features(datasets(), config)
        self.assertEqual(list(features), ["margin_composition__buy_share__k1__w3_percentile"])

    def test_default_feature_count_is_16(self):
        features, catalog, _ = build_margin_composition_features(datasets(900))
        self.assertEqual(features.shape[1], 16)
        self.assertEqual(len(catalog), 16)

    def test_equivalence_validation_passes(self):
        config = MarginCompositionConfig(k_values=(1, 3), rolling_windows=(3, 5))
        _, _, validation = build_margin_composition_features(datasets(), config)
        self.assertTrue(validation["formula_valid"].all())

    def test_equivalence_has_required_metrics(self):
        config = MarginCompositionConfig(k_values=(1,), rolling_windows=(3,))
        _, _, result = build_margin_composition_features(datasets(), config)
        self.assertTrue({"pearson", "spearman", "pr_bin_equivalence", "max_abs_formula_diff"}.issubset(result))

    def test_composition_groups_have_fixed_boundaries(self):
        result = _composition_group(pd.Series([0., 19.99, 20., 79.99, 80., 100.]))
        self.assertEqual(result.tolist(), [
            "sell_dominant_PR0-20", "sell_dominant_PR0-20",
            "neutral_PR20-80", "neutral_PR20-80",
            "buy_dominant_PR80-100", "buy_dominant_PR80-100",
        ])

    def test_primary_scope_excludes_imbalance(self):
        config = MarginCompositionConfig(k_values=(1,), rolling_windows=(3,), outcome_horizons=(1,))
        features, _, _ = build_margin_composition_features(datasets(), config)
        outcomes = pd.DataFrame({"O1_C1": np.linspace(-.01, .02, len(features))}, index=features.index)
        with patch("src.margin_composition.apply_fdr", side_effect=lambda x: x.assign(
            family_fdr_q_value=np.nan, global_fdr_q_value=np.nan, evidence_level="No Evidence"
        )):
            primary, fdr, _ = run_composition_primary(features, outcomes)
        self.assertEqual(len(primary), 2)
        self.assertEqual(set(fdr["fdr_scope"]), {FDR_SCOPE})
        self.assertFalse(fdr["predictor"].str.contains("imbalance").any())

    def test_default_inferential_universe_is_192(self):
        features, _, _ = build_margin_composition_features(datasets(900))
        outcomes = pd.DataFrame({f"O1_C{h}": np.linspace(-.01, .02, 900) for h in (1,2,3,5,10,20)}, index=features.index)
        with patch("src.margin_composition.apply_fdr", side_effect=lambda x: x.assign(
            family_fdr_q_value=np.nan, global_fdr_q_value=np.nan, evidence_level="No Evidence"
        )):
            primary, _, _ = run_composition_primary(features, outcomes)
        self.assertEqual(len(primary), 192)

    def test_nan_p_is_not_testable(self):
        features = pd.DataFrame({"x_percentile": [1.]}, index=pd.date_range("2020-01-01", periods=1))
        features.attrs["catalog"] = {"x_percentile": {"family": "margin_composition", "k": 1, "window": 3}}
        outcomes = pd.DataFrame({"O1_C1": [np.nan]}, index=features.index)
        with patch("src.margin_composition.apply_fdr", side_effect=lambda x: x.assign(
            family_fdr_q_value=np.nan, global_fdr_q_value=np.nan, evidence_level="No Evidence"
        )):
            _, fdr, _ = run_composition_primary(features, outcomes)
        self.assertEqual(set(fdr["evidence_level"]), {"Not Testable"})

    def test_low_turnover_outputs_three_groups_by_three_horizons(self):
        index = pd.bdate_range("2020-01-01", periods=12)
        turnover = pd.Series([1.] * 9 + [50.] * 3, index=index)
        composition = pd.Series([10., 50., 90.] * 4, index=index)
        outcomes = pd.DataFrame({f"O1_C{h}": np.linspace(-.1, .1, 12) for h in (5,10,20)}, index=index)
        pooled, annual, summary, contribution, clusters = run_low_turnover_composition(turnover, composition, outcomes)
        self.assertEqual(len(pooled), 9)
        self.assertTrue({"q05", "q95", "downside_below_minus_10pct"}.issubset(pooled))
        self.assertFalse(annual.empty)
        self.assertFalse(summary.empty)
        self.assertFalse(contribution.empty)
        self.assertFalse(clusters.empty)

    def test_clusters_use_trading_calendar_adjacency(self):
        index = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])
        joined = pd.DataFrame({"composition_pr": [10., 20., 30.], "O1_C5": 0., "O1_C10": 0., "O1_C20": 0.}, index=index)
        clusters = _low_turnover_clusters(joined, pd.Series([True, True, False], index=index))
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters.loc[0, "trading_days"], 2)

    def test_cluster_uses_first_signal_outcomes(self):
        index = pd.bdate_range("2024-01-01", periods=3)
        joined = pd.DataFrame({"composition_pr": [10., 20., 30.], "O1_C5": [1.,2.,3.], "O1_C10": [4.,5.,6.], "O1_C20": [7.,8.,9.]}, index=index)
        clusters = _low_turnover_clusters(joined, pd.Series([True, True, False], index=index))
        self.assertEqual(clusters.loc[0, "first_signal_O1_C20"], 7.)

    def test_year_contributions_reconcile_to_pooled_mean(self):
        index = pd.bdate_range("2020-12-20", periods=20)
        turnover = pd.Series(1., index=index)
        composition = pd.Series(50., index=index)
        outcomes = pd.DataFrame({"O1_C5": .01, "O1_C10": .02, "O1_C20": np.linspace(-.1, .1, 20)}, index=index)
        _, _, _, contribution, _ = run_low_turnover_composition(turnover, composition, outcomes)
        pooled_mean = outcomes["O1_C20"].mean()
        self.assertAlmostEqual(contribution["weighted_contribution_to_total_mean"].sum(), pooled_mean)

    def test_turnover_artifacts_validate_both_commits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run_info_turnover.txt").write_text(
                f"git_commit={TURNOVER_BASE_COMMIT}\nbaseline_commit={BASELINE_COMMIT}\n", encoding="utf-8"
            )
            for name in (
                "turnover_primary_results.csv", "turnover_fdr_results.csv", "turnover_annual_results.csv",
                "turnover_annual_robustness_summary.csv", "turnover_absorption_results.csv",
            ):
                pd.DataFrame({"x": []}).to_csv(root / name, index=False)
            self.assertEqual(load_frozen_turnover(root)["run_info"]["git_commit"], TURNOVER_BASE_COMMIT)

    def test_turnover_commit_mismatch_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run_info_turnover.txt").write_text(f"git_commit=wrong\nbaseline_commit={BASELINE_COMMIT}\n", encoding="utf-8")
            for name in (
                "turnover_primary_results.csv", "turnover_fdr_results.csv", "turnover_annual_results.csv",
                "turnover_annual_robustness_summary.csv", "turnover_absorption_results.csv",
            ):
                pd.DataFrame({"x": []}).to_csv(root / name, index=False)
            with self.assertRaises(ValueError):
                load_frozen_turnover(root)

    def test_absorption_candidate_filter_is_strict(self):
        rows = pd.DataFrame({
            "status": ["estimated", "estimated", "failed"],
            "original_family": ["margin_buy", "short_sell", "margin_sell"],
            "original_predictor": ["a__amount_ratio__x", "b__amount_ratio__x", "c__amount_ratio__x"],
            "turnover_variant": ["amount_ratio", "amount_ratio", "amount_ratio"],
        })
        self.assertEqual(len(select_composition_absorption_candidates(rows)), 1)

    def test_nested_models_share_complete_cases_and_hc3(self):
        n = 60
        index = pd.RangeIndex(n)
        signal = pd.Series([0.] * 30 + [1.] * 30, index=index)
        turnover = pd.Series(np.linspace(0, 1, n), index=index)
        composition = pd.Series(np.sin(np.linspace(0, 4, n)), index=index)
        composition.iloc[0] = np.nan
        future = pd.Series(np.linspace(-.02, .03, n), index=index)
        prior = pd.Series(np.cos(np.linspace(0, 4, n)), index=index)
        def fake_fit(y, design):
            return FakeFit(len(y), design.shape[1], .2 if design.shape[1] == 4 else .1)
        with patch("src.margin_composition._fit_base_ols", side_effect=fake_fit):
            result = _fit_nested_models(signal, turnover, composition, future, prior, MarginCompositionConfig())
        self.assertEqual(result["N"], 59)
        self.assertEqual(result["status"], "estimated")
        self.assertTrue(result["hc3_finite_b"] and result["hc3_finite_c"])

    def test_rank_failure_is_unstable_collinearity(self):
        n = 60
        signal = pd.Series([0.] * 30 + [1.] * 30)
        same = pd.Series(np.linspace(0, 1, n))
        result = _fit_nested_models(signal, same, same, same, same, MarginCompositionConfig())
        self.assertEqual(result["status"], "unstable_collinearity")

    def test_absorption_classification_never_calls_instability_economic_absorption(self):
        attenuation, label = _classify_absorption({"status": "unstable_collinearity"}, MarginCompositionConfig())
        self.assertTrue(pd.isna(attenuation))
        self.assertEqual(label, "unstable_collinearity")

    def test_notebook_is_valid_json(self):
        path = Path("notebooks/margin_composition_incremental_colab.ipynb")
        if path.exists():
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["nbformat"], 4)

    def test_incremental_runner_does_not_call_full_pipeline(self):
        from src.margin_composition import run_margin_composition_study
        source = inspect.getsource(run_margin_composition_study)
        self.assertNotIn("pipeline.run", source)
        self.assertNotIn("run_margin_turnover_study", source)

    def test_incremental_export_has_complete_separate_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline, turnover = frozen_dirs(root)
            empty = pd.DataFrame()
            config = MarginCompositionConfig(
                k_values=(1,), rolling_windows=(3,), outcome_horizons=(1, 5, 10, 20),
                fixed_turnover_k=1, fixed_turnover_window=3,
                output_root=root / "outputs_composition",
            )
            with patch("src.margin_composition.run_composition_primary", return_value=(empty, empty, empty)), \
                 patch("src.margin_composition.run_pr_bin_descriptive", return_value=empty), \
                 patch("src.margin_composition.run_controlled_tests", return_value=empty), \
                 patch("src.margin_composition.annual_results", return_value=empty), \
                 patch("src.margin_composition.annual_robustness_summary", return_value=empty):
                result = run_margin_composition_study(
                    baseline, turnover, config=config, datasets=full_datasets(), export=True
                )
            expected = {
                "run_info_composition.txt", "composition_feature_catalog.csv",
                "composition_equivalence_validation.csv", "composition_primary_results.csv",
                "composition_fdr_results.csv", "composition_fdr_diagnostics.csv",
                "composition_pr_bin_results.csv", "composition_controlled_results.csv",
                "composition_annual_results.csv", "composition_annual_robustness_summary.csv",
                "composition_neighborhood_consistency.csv", "low_turnover_composition_results.csv",
                "low_turnover_composition_annual_results.csv", "low_turnover_composition_annual_summary.csv",
                "low_turnover_year_contribution.csv", "low_turnover_event_clusters.csv",
                "composition_absorption_results.csv", "composition_summary.md",
            }
            self.assertTrue(expected.issubset({path.name for path in result["run_dir"].iterdir()}))
            info = (result["run_dir"] / "run_info_composition.txt").read_text(encoding="utf-8")
            self.assertIn("baseline_full_research_rerun=False", info)
            self.assertIn("turnover_full_research_rerun=False", info)
            self.assertIn(f"fdr_scope={FDR_SCOPE}", info)


if __name__ == "__main__":
    unittest.main()
