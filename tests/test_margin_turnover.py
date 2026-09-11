import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.features import pr_group, trailing_percentile
from src.margin_turnover import (
    BASELINE_COMMIT,
    FDR_SCOPE,
    MarginTurnoverConfig,
    _after_turnover_regression,
    _turnover_base_series,
    build_margin_turnover_features,
    load_frozen_baseline,
    margin_turnover_variant_series,
    run_margin_turnover_study,
    run_turnover_absorption,
    run_turnover_primary,
    select_retained_margin_signals,
    turnover_neighborhood_consistency,
)


def synthetic_datasets(periods=12):
    idx = pd.bdate_range("2024-01-01", periods=periods)
    margin_buy = pd.DataFrame(
        {"2330": np.arange(1, periods + 1), "2317": 2.0, "0050": 1000.0}, index=idx
    )
    margin_sell = pd.DataFrame(
        {"2330": 3.0, "2317": np.arange(2, periods + 2), "0050": 1000.0}, index=idx
    )
    aggregate_buy = pd.DataFrame(
        {"上市融資交易金額": np.arange(10, periods + 10), "上櫃融資交易金額": 5.0}, index=idx
    )
    aggregate_sell = pd.DataFrame(
        {"上市融資交易金額": np.arange(20, periods + 20), "上櫃融資交易金額": 5.0}, index=idx
    )
    market_volume = pd.DataFrame({"TAIEX": 1_000_000.0, "OTC": 500_000.0}, index=idx)
    market_amount = pd.DataFrame({"TAIEX": 10_000.0, "OTC": 5_000.0}, index=idx)
    close = pd.DataFrame({"0050": np.linspace(100, 112, periods)}, index=idx)
    open_ = pd.DataFrame({"0050": np.linspace(99, 111, periods)}, index=idx)
    margin_balance = pd.DataFrame({"2330": 1.0, "2317": 1.0, "0050": 1.0}, index=idx)
    market_value = pd.DataFrame({"2330": 1.0, "2317": 1.0, "0050": 1.0}, index=idx)
    return {
        "open": open_, "close": close,
        "margin_buy": margin_buy, "margin_sell": margin_sell,
        "margin_balance": margin_balance,
        "aggregate_buy": aggregate_buy, "aggregate_sell": aggregate_sell,
        "market_volume": market_volume, "market_amount": market_amount,
        "market_value": market_value,
    }


def make_baseline_dir(root: Path, commit=BASELINE_COMMIT, fdr=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_info.txt").write_text(f"git_commit={commit}\n", encoding="utf-8")
    (fdr if fdr is not None else pd.DataFrame({"family": []})).to_csv(root / "fdr_results.csv", index=False)
    for name in (
        "controlled_results.csv", "annual_robustness_summary.csv", "neighborhood_consistency.csv"
    ):
        pd.DataFrame({"placeholder": []}).to_csv(root / name, index=False)


class FakeInfluence:
    def __init__(self, leverage):
        self.hat_matrix_diag = np.asarray(leverage, dtype=float)


class FakeRobust:
    params = np.array([0.0, 0.20, 0.10, 0.05])
    pvalues = np.array([1.0, 0.01, 0.02, 0.03])

    def cov_params(self):
        return np.eye(4)


class FakeFit:
    def __init__(self, n):
        self.n = n
        self.robust_called = False

    def get_influence(self):
        return FakeInfluence(np.repeat(4 / self.n, self.n))

    def get_robustcov_results(self, cov_type):
        if cov_type != "HC3":
            raise AssertionError("HC3 must be preserved")
        self.robust_called = True
        return FakeRobust()


class MarginTurnoverTests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_datasets()
        self.symbols = {"2330", "2317"}

    def test_raw_turnover_is_primary_universe_buy_plus_sell(self):
        base = _turnover_base_series(self.data, self.symbols)
        expected = (
            self.data["margin_buy"][["2330", "2317"]].sum(axis=1)
            + self.data["margin_sell"][["2330", "2317"]].sum(axis=1)
        )
        pd.testing.assert_series_equal(base["turnover_lots"], expected)
        self.assertLess(base["turnover_lots"].iloc[0], 2000)

    def test_volume_ratio_uses_ratio_of_rolling_sums(self):
        base = _turnover_base_series(self.data, self.symbols)
        result = margin_turnover_variant_series(base, 3)["volume_ratio"]
        expected = base["turnover_shares"].iloc[:3].sum() / base["market_volume"].iloc[:3].sum()
        self.assertAlmostEqual(result.iloc[2], expected)

    def test_amount_ratio_uses_ratio_of_rolling_sums(self):
        base = _turnover_base_series(self.data, self.symbols)
        result = margin_turnover_variant_series(base, 3)["amount_ratio"]
        expected = base["turnover_amount"].iloc[:3].sum() / base["market_amount"].iloc[:3].sum()
        self.assertAlmostEqual(result.iloc[2], expected)

    def test_all_fixed_k_values_are_built_correctly(self):
        base = _turnover_base_series(self.data, self.symbols)
        for k in (1, 3, 5, 10):
            raw = margin_turnover_variant_series(base, k)["raw_lots"]
            self.assertAlmostEqual(raw.iloc[k - 1], base["turnover_lots"].iloc[:k].sum())
            self.assertTrue(raw.iloc[: k - 1].isna().all())

    def test_turnover_percentile_does_not_use_future(self):
        base = pd.Series([3.0, 1.0, 2.0, 100.0])
        changed = base.copy(); changed.iloc[-1] = -100.0
        self.assertEqual(trailing_percentile(base, 3).iloc[2], trailing_percentile(changed, 3).iloc[2])

    def test_missing_date_is_not_forward_filled(self):
        self.data["margin_buy"].loc[self.data["margin_buy"].index[3], ["2330", "2317"]] = np.nan
        self.data["margin_sell"].loc[self.data["margin_sell"].index[3], ["2330", "2317"]] = np.nan
        config = MarginTurnoverConfig(k_values=(1,), rolling_windows=(3,), outcome_horizons=(1,))
        features = build_margin_turnover_features(self.data, self.symbols, config)
        column = "margin_turnover__raw_lots__k1__w3_percentile"
        self.assertTrue(pd.isna(features.loc[features.index[3], column]))

    def test_fixed_pr_bins_are_non_overlapping(self):
        values = pd.Series([0, 5, 20, 40, 60, 80, 95, 100], dtype=float)
        self.assertEqual(list(pr_group(values).astype(str)), [
            "PR0-5", "PR5-20", "PR20-40", "PR40-60", "PR60-80", "PR80-95", "PR95-100", "PR95-100"
        ])

    def test_turnover_fdr_scope_contains_only_turnover_predictors(self):
        config = MarginTurnoverConfig(k_values=(1,), rolling_windows=(3,), outcome_horizons=(1,))
        features = build_margin_turnover_features(self.data, self.symbols, config)
        outcomes = pd.DataFrame({"O1_C1": np.linspace(-.01, .02, len(features))}, index=features.index)

        def fake_fdr(frame):
            out = frame.copy(); out["evidence_level"] = "No Evidence"
            out["family_fdr_q_value"] = np.nan; out["global_fdr_q_value"] = np.nan
            return out

        with patch("src.margin_turnover.apply_fdr", side_effect=fake_fdr):
            _, result, diagnostics = run_turnover_primary(features, outcomes)
        self.assertEqual(set(result["family"]), {"margin_turnover"})
        self.assertEqual(set(result["fdr_scope"]), {FDR_SCOPE})
        self.assertEqual(set(diagnostics["fdr_scope"]), {FDR_SCOPE})

    def test_default_incremental_universe_has_576_cells(self):
        features = build_margin_turnover_features(self.data, self.symbols)
        outcomes = pd.DataFrame(
            {f"O1_C{h}": np.linspace(-.01, .02, len(features)) for h in (1, 2, 3, 5, 10, 20)},
            index=features.index,
        )

        def fake_fdr(frame):
            out = frame.copy(); out["evidence_level"] = "No Evidence"
            out["family_fdr_q_value"] = np.nan; out["global_fdr_q_value"] = np.nan
            return out

        with patch("src.margin_turnover.apply_fdr", side_effect=fake_fdr):
            primary, fdr, _ = run_turnover_primary(features, outcomes)
        self.assertEqual(features.shape[1], 48)
        self.assertEqual(len(primary), 576)
        self.assertEqual(len(fdr), 576)

    def test_neighborhood_keeps_normalization_variants_separate(self):
        results = pd.DataFrame([
            {"family": "margin_turnover", "variant": "amount_ratio", "pr_group": "PR95-100", "k": 1, "rolling_window": 126, "outcome_horizon": "O1_C5", "effect_vs_unconditional": .01},
            {"family": "margin_turnover", "variant": "volume_ratio", "pr_group": "PR95-100", "k": 1, "rolling_window": 126, "outcome_horizon": "O1_C5", "effect_vs_unconditional": -.01},
        ])
        summary = turnover_neighborhood_consistency(results)
        overall = summary[summary["consistency_type"].eq("overall")]
        self.assertEqual(set(overall["variant"]), {"amount_ratio", "volume_ratio"})
        self.assertEqual(overall["n_cells"].tolist(), [1, 1])

    def test_baseline_frame_is_read_only_during_selection(self):
        baseline = pd.DataFrame({
            "family": ["margin_buy"], "research_decision": ["保留"], "predictor": ["x"]
        })
        before = baseline.copy(deep=True)
        select_retained_margin_signals(baseline)
        assert_frame_equal(baseline, before)

    def test_retained_candidate_selection_and_fallback(self):
        direct = pd.DataFrame({
            "family": ["margin_buy", "margin_sell", "short_sell"],
            "research_decision": ["保留", "修改後再測", "保留"],
        })
        self.assertEqual(select_retained_margin_signals(direct).index.tolist(), [0])
        fallback = pd.DataFrame({
            "family": ["margin_sell", "margin_buy"], "evidence_level": ["Level A", "Level A"],
            "robustness_status": ["robust", "mixed"], "live_eligibility": [True, True],
        })
        self.assertEqual(select_retained_margin_signals(fallback).index.tolist(), [0])

    def test_after_turnover_regression_uses_hc3_and_returns_coefficients(self):
        n = 50
        signal = pd.Series([0.0] * 25 + [1.0] * 25)
        turnover = pd.Series(np.linspace(0, 1, n) ** 2)
        prior = pd.Series(np.sin(np.linspace(0, 4, n)))
        future = pd.Series(np.linspace(-.02, .03, n))
        fake = FakeFit(n)
        with patch("src.margin_turnover._fit_base_ols", return_value=fake):
            result = _after_turnover_regression(signal, turnover, future, prior, MarginTurnoverConfig())
        self.assertEqual(result["status"], "estimated")
        self.assertAlmostEqual(result["beta_after_turnover"], .20)
        self.assertAlmostEqual(result["turnover_beta"], .10)
        self.assertTrue(fake.robust_called)

    def test_absorption_before_after_and_attenuation_are_recorded(self):
        idx = self.data["close"].index
        candidate = pd.DataFrame([{
            "predictor": "margin_buy__k1__amount_ratio__w2_percentile", "family": "margin_buy",
            "research_decision": "保留", "pr_group": "PR95-100", "k": 1,
            "rolling_window": 2, "outcome_horizon": "O1_C1", "N": 30,
            "win_rate": .6, "effect_vs_unconditional": .01, "evidence_level": "Level A",
        }])
        turnover = pd.DataFrame({
            "margin_turnover__amount_ratio__k1__w2_percentile": np.linspace(1, 100, len(idx)),
            "margin_turnover__volume_ratio__k1__w2_percentile": np.linspace(1, 100, len(idx)),
        }, index=idx)
        original = pd.DataFrame({candidate.iloc[0]["predictor"]: np.linspace(1, 100, len(idx))}, index=idx)
        outcomes = pd.DataFrame({"O1_C1": .01}, index=idx)
        before = {"signal_beta": .10, "signal_p_value": .01, "status": "estimated"}
        after = {"N": 40, "beta_after_turnover": .05, "p_after_turnover": .02,
                 "turnover_beta": .03, "turnover_p_value": .04, "condition_number": 10.0,
                 "status": "estimated"}
        config = MarginTurnoverConfig(k_values=(1,), rolling_windows=(2,), outcome_horizons=(1,))
        with patch("src.margin_turnover._baseline_original_features", return_value=original), \
             patch("src.margin_turnover.controlled_regression", return_value=before), \
             patch("src.margin_turnover._after_turnover_regression", return_value=after):
            result = run_turnover_absorption(candidate, self.data, turnover, outcomes, self.data["close"]["0050"], config)
        self.assertTrue(np.allclose(result["attenuation_ratio"], .5))
        self.assertEqual(set(result["absorption_classification"]), {"partially_absorbed"})
        self.assertEqual(set(result["baseline_N"]), {30})

    def test_absorption_before_and_after_use_same_complete_cases(self):
        idx = self.data["close"].index
        predictor = "margin_buy__k1__amount_ratio__w2_percentile"
        candidate = pd.DataFrame([{
            "predictor": predictor, "family": "margin_buy", "research_decision": "保留",
            "pr_group": "PR95-100", "k": 1, "rolling_window": 2,
            "outcome_horizon": "O1_C1",
        }])
        original = pd.DataFrame({predictor: np.linspace(1, 100, len(idx))}, index=idx)
        turnover = pd.DataFrame({
            "margin_turnover__amount_ratio__k1__w2_percentile": np.linspace(1, 100, len(idx)),
            "margin_turnover__volume_ratio__k1__w2_percentile": np.linspace(1, 100, len(idx)),
        }, index=idx)
        turnover.iloc[0] = np.nan
        captured = []

        def before(signal, future, prior, **kwargs):
            captured.append((signal.copy(), future.copy(), prior.copy()))
            return {"signal_beta": .1, "signal_p_value": .01, "status": "estimated"}

        after = {"N": len(idx) - 1, "beta_after_turnover": .08, "p_after_turnover": .02,
                 "turnover_beta": .01, "turnover_p_value": .2, "condition_number": 2.0,
                 "status": "estimated"}
        with patch("src.margin_turnover._baseline_original_features", return_value=original), \
             patch("src.margin_turnover.controlled_regression", side_effect=before), \
             patch("src.margin_turnover._after_turnover_regression", return_value=after):
            run_turnover_absorption(
                candidate, self.data, turnover, pd.DataFrame({"O1_C1": .01}, index=idx),
                self.data["close"]["0050"], MarginTurnoverConfig(k_values=(1,)),
            )
        self.assertTrue(all(series.iloc[0] != series.iloc[0] for call in captured for series in call))

    def test_near_zero_beta_has_undefined_attenuation(self):
        idx = self.data["close"].index
        candidate = pd.DataFrame([{
            "predictor": "margin_buy__k1__amount_ratio__w2_percentile", "family": "margin_buy",
            "research_decision": "保留", "pr_group": "PR95-100", "k": 1,
            "rolling_window": 2, "outcome_horizon": "O1_C1",
        }])
        turnover = pd.DataFrame({
            "margin_turnover__amount_ratio__k1__w2_percentile": 50.0,
            "margin_turnover__volume_ratio__k1__w2_percentile": 50.0,
        }, index=idx)
        original = pd.DataFrame({candidate.iloc[0]["predictor"]: np.linspace(1, 100, len(idx))}, index=idx)
        after = {"N": 40, "beta_after_turnover": .01, "p_after_turnover": .5,
                 "turnover_beta": .0, "turnover_p_value": .5, "condition_number": 2.0,
                 "status": "estimated"}
        with patch("src.margin_turnover._baseline_original_features", return_value=original), \
             patch("src.margin_turnover.controlled_regression", return_value={"signal_beta": 1e-15, "signal_p_value": .5, "status": "estimated"}), \
             patch("src.margin_turnover._after_turnover_regression", return_value=after):
            result = run_turnover_absorption(candidate, self.data, turnover, pd.DataFrame({"O1_C1": .01}, index=idx), self.data["close"]["0050"])
        self.assertTrue(result["attenuation_ratio"].isna().all())

    def test_multicollinearity_is_explicit(self):
        n = 50
        signal = pd.Series([0.0] * 25 + [1.0] * 25)
        result = _after_turnover_regression(
            signal, signal, pd.Series(np.linspace(0, 1, n)),
            pd.Series(np.linspace(-1, 1, n)), MarginTurnoverConfig(),
        )
        self.assertEqual(result["status"], "rank_deficient")

    def test_baseline_commit_mismatch_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_baseline_dir(root, commit="wrong")
            with self.assertRaisesRegex(ValueError, "Baseline commit mismatch"):
                load_frozen_baseline(root)

    def test_incremental_runner_never_calls_full_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_baseline_dir(root)
            empty = pd.DataFrame()
            config = MarginTurnoverConfig(k_values=(1,), rolling_windows=(2,), outcome_horizons=(1,))
            with patch("src.pipeline.run", side_effect=AssertionError("full pipeline called")) as full_run, \
                 patch("src.margin_turnover.run_turnover_primary", return_value=(empty, empty, empty)), \
                 patch("src.margin_turnover.run_pr_bin_descriptive", return_value=empty), \
                 patch("src.margin_turnover.run_controlled_tests", return_value=empty), \
                 patch("src.margin_turnover.annual_results", return_value=empty), \
                 patch("src.margin_turnover.annual_robustness_summary", return_value=empty), \
                 patch("src.margin_turnover.neighborhood_consistency", return_value=empty), \
                 patch("src.margin_turnover.run_turnover_absorption", return_value=empty):
                result = run_margin_turnover_study(root, config=config, datasets=self.data, export=False)
            full_run.assert_not_called()
            self.assertIn("features", result)

    def test_incremental_export_uses_separate_complete_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_dir = root / "baseline"
            make_baseline_dir(baseline_dir)
            empty = pd.DataFrame()
            config = MarginTurnoverConfig(
                k_values=(1,), rolling_windows=(2,), outcome_horizons=(1,),
                output_root=root / "outputs_turnover",
            )
            with patch("src.margin_turnover.run_turnover_primary", return_value=(empty, empty, empty)), \
                 patch("src.margin_turnover.run_pr_bin_descriptive", return_value=empty), \
                 patch("src.margin_turnover.run_controlled_tests", return_value=empty), \
                 patch("src.margin_turnover.annual_results", return_value=empty), \
                 patch("src.margin_turnover.annual_robustness_summary", return_value=empty), \
                 patch("src.margin_turnover.run_turnover_absorption", return_value=empty):
                result = run_margin_turnover_study(
                    baseline_dir, config=config, datasets=self.data, export=True
                )
            expected = {
                "run_info_turnover.txt", "turnover_feature_catalog.csv",
                "turnover_primary_results.csv", "turnover_fdr_results.csv",
                "turnover_fdr_diagnostics.csv", "turnover_pr_bin_results.csv",
                "turnover_controlled_results.csv", "turnover_absorption_results.csv",
                "turnover_annual_results.csv", "turnover_annual_robustness_summary.csv",
                "turnover_neighborhood_consistency.csv", "turnover_summary.md",
            }
            self.assertTrue(expected.issubset({path.name for path in result["run_dir"].iterdir()}))
            run_info = (result["run_dir"] / "run_info_turnover.txt").read_text(encoding="utf-8")
            self.assertIn("baseline_full_research_rerun=False", run_info)
            self.assertIn(f"fdr_scope={FDR_SCOPE}", run_info)


if __name__ == "__main__":
    unittest.main()
