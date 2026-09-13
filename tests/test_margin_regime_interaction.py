import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.margin_regime_interaction import (
    COMPOSITION_COMMIT,
    FDR_SCOPE,
    MarginRegimeInteractionConfig,
    apply_interaction_fdr,
    binary_regime,
    fit_regime_interaction,
    load_frozen_composition,
    prior_5d_return,
    regime_event_clusters,
    regime_primary_results,
    regime_signal_distribution,
    run_margin_regime_interaction_study,
    select_primary_signal_specs,
)
from src.margin_turnover import BASELINE_COMMIT
from src.margin_composition import TURNOVER_BASE_COMMIT


PREDICTOR = "margin_buy__amount_ratio__k1__w10_percentile"


def baseline_fdr():
    rows = []
    for family in ("margin_buy", "margin_sell"):
        predictor = f"{family}__amount_ratio__k1__w10_percentile"
        for outcome in ("O1_C1", "O1_C2", "O1_C3", "O1_C5", "O1_C10", "O1_C20"):
            rows.append({
                "predictor": predictor, "family": family, "k": 1,
                "rolling_window": 10, "pr_group": "PR95-100",
                "outcome_horizon": outcome,
            })
    rows.append({
        "predictor": "short_sell__amount_ratio__k1__w10_percentile",
        "family": "short_sell", "k": 1, "rolling_window": 10,
        "pr_group": "PR95-100", "outcome_horizon": "O1_C1",
    })
    return pd.DataFrame(rows)


def synthetic_data(periods=240):
    index = pd.bdate_range("2019-01-01", periods=periods)
    wave = np.sin(np.arange(periods) / 8)
    close = 100 + np.cumsum(.1 + wave)
    buy = 100 + 25 * (wave + 1)
    sell = 100 + 25 * (1 - wave)
    return {
        "open": pd.DataFrame({"0050": close - .1}, index=index),
        "close": pd.DataFrame({"0050": close}, index=index),
        "margin_buy": pd.DataFrame({"2330": buy}, index=index),
        "margin_sell": pd.DataFrame({"2330": sell}, index=index),
        "margin_balance": pd.DataFrame({"2330": 1000.}, index=index),
        "aggregate_buy": pd.DataFrame({
            "上市融資交易金額": buy * 100., "上櫃融資交易金額": buy * 10.,
        }, index=index),
        "aggregate_sell": pd.DataFrame({
            "上市融資交易金額": sell * 100., "上櫃融資交易金額": sell * 10.,
        }, index=index),
        "market_volume": pd.DataFrame({"TAIEX": 1e6, "OTC": 5e5}, index=index),
        "market_amount": pd.DataFrame({"TAIEX": 1e8, "OTC": 5e7}, index=index),
        "market_value": pd.DataFrame({"2330": 1e9}, index=index),
    }


def frozen_dirs(root: Path):
    baseline = root / "baseline"
    turnover = root / "turnover"
    composition = root / "composition"
    baseline.mkdir(); turnover.mkdir(); composition.mkdir()
    (baseline / "run_info.txt").write_text(f"git_commit={BASELINE_COMMIT}\n", encoding="utf-8")
    baseline_fdr().to_csv(baseline / "fdr_results.csv", index=False)
    for name in ("controlled_results.csv", "annual_robustness_summary.csv", "neighborhood_consistency.csv"):
        pd.DataFrame({"x": []}).to_csv(baseline / name, index=False)
    (turnover / "run_info_turnover.txt").write_text(
        f"git_commit={TURNOVER_BASE_COMMIT}\nbaseline_commit={BASELINE_COMMIT}\n", encoding="utf-8"
    )
    for name in (
        "turnover_primary_results.csv", "turnover_fdr_results.csv", "turnover_annual_results.csv",
        "turnover_annual_robustness_summary.csv", "turnover_absorption_results.csv",
    ):
        pd.DataFrame({"x": []}).to_csv(turnover / name, index=False)
    (composition / "run_info_composition.txt").write_text(
        f"git_commit={COMPOSITION_COMMIT}\nturnover_base_commit={TURNOVER_BASE_COMMIT}\n"
        f"baseline_commit={BASELINE_COMMIT}\nprice_source_open=etl:adj_open\n"
        "price_source_close=etl:adj_close\n", encoding="utf-8"
    )
    for name in (
        "composition_primary_results.csv", "composition_fdr_results.csv",
        "composition_annual_results.csv", "composition_annual_robustness_summary.csv",
        "composition_absorption_results.csv",
    ):
        pd.DataFrame({"x": []}).to_csv(composition / name, index=False)
    return baseline, turnover, composition


class MarginRegimeInteractionTests(unittest.TestCase):
    def test_prior_return_uses_t_and_t_minus_5_only(self):
        close = pd.Series(np.arange(100., 112.))
        original = prior_5d_return(close)
        changed = close.copy(); changed.iloc[-1] = 9999
        revised = prior_5d_return(changed)
        pd.testing.assert_series_equal(original.iloc[:-1], revised.iloc[:-1])
        self.assertAlmostEqual(original.iloc[5], 105 / 100 - 1)

    def test_binary_regime_boundary(self):
        result = binary_regime(pd.Series([.01, 0., -.01, np.nan]))
        self.assertEqual(result.iloc[:3].tolist(), ["Up", "Down", "Down"])
        self.assertTrue(pd.isna(result.iloc[3]))

    def test_only_high_amount_ratio_buy_sell_specs_enter(self):
        source = baseline_fdr()
        source.loc[len(source)] = [
            "margin_buy__raw__k1__w10_percentile", "margin_buy", 1, 10,
            "PR95-100", "O1_C1",
        ]
        source.loc[len(source)] = [PREDICTOR, "margin_buy", 1, 10, "PR0-5", "O1_C1"]
        result = select_primary_signal_specs(source)
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result.family), {"margin_buy", "margin_sell"})
        self.assertTrue(result.predictor.str.contains("__amount_ratio__", regex=False).all())

    def test_occurrence_percentages_use_signal_days_only(self):
        index = pd.bdate_range("2024-01-01", periods=5)
        signals = pd.DataFrame({PREDICTOR: [1., 1., 0., 1., np.nan]}, index=index)
        regime = pd.Series(["Up", "Down", "Up", "Up", "Down"], index=index, dtype="string")
        specs = pd.DataFrame([{"predictor": PREDICTOR, "family": "margin_buy", "k": 1,
                               "rolling_window": 10, "pr_group": "PR95-100"}])
        result = regime_signal_distribution(signals, regime, specs).iloc[0]
        self.assertEqual((result["Up N"], result["Down N"]), (2, 1))
        self.assertAlmostEqual(result["Up %"], 2 / 3)

    def test_descriptive_effect_uses_same_regime_control(self):
        index = pd.bdate_range("2024-01-01", periods=8)
        signals = pd.DataFrame({PREDICTOR: [1, 0, 1, 0, 1, 0, 1, 0]}, index=index, dtype=float)
        regime = pd.Series(["Up"] * 4 + ["Down"] * 4, index=index, dtype="string")
        outcomes = pd.DataFrame({"O1_C1": [3, 1, 3, 1, 10, 8, 10, 8]}, index=index)
        specs = pd.DataFrame([{"predictor": PREDICTOR, "family": "margin_buy", "k": 1,
                               "rolling_window": 10, "pr_group": "PR95-100"}])
        result = regime_primary_results(signals, regime, outcomes, specs)
        self.assertEqual(set(result.state), {"A", "B", "C", "D"})
        self.assertTrue((result.loc[result.signal_state.eq(1), "effect_vs_same_regime_non_signal"] == 2).all())

    def test_interaction_recovers_up_and_down_effects(self):
        rng = np.random.default_rng(4)
        n = 400
        signal = pd.Series(np.tile([0., 1.], n // 2))
        down = pd.Series(np.repeat([0., 1.], n // 2))
        future = 0.01 + .02 * signal - .01 * down + .04 * signal * down
        future += pd.Series(rng.normal(0, .002, n))
        result = fit_regime_interaction(
            signal, down, future, MarginRegimeInteractionConfig(min_group_n=20)
        )
        self.assertEqual(result["status"], "estimated")
        self.assertAlmostEqual(result["beta_up"], .02, places=2)
        self.assertAlmostEqual(result["beta_down"], .06, places=2)
        self.assertAlmostEqual(result["interaction_beta"], .04, places=2)

    def test_sparse_cell_is_not_testable(self):
        signal = pd.Series([0.] * 30 + [1.] * 30)
        down = pd.Series([0.] * 60)
        result = fit_regime_interaction(
            signal, down, pd.Series(np.arange(60.)), MarginRegimeInteractionConfig(min_group_n=2)
        )
        self.assertEqual(result["status"], "insufficient_group_sample")
        self.assertTrue(pd.isna(result["interaction_p"]))

    def test_turnover_control_is_in_same_interaction_model(self):
        rng = np.random.default_rng(9)
        n = 400
        signal = pd.Series(np.tile([0., 1.], n // 2))
        down = pd.Series(np.repeat([0., 1.], n // 2))
        turnover = pd.Series(rng.uniform(0, 1, n))
        future = .02 * signal + .03 * signal * down + .01 * turnover
        future += pd.Series(rng.normal(0, .003, n))
        result = fit_regime_interaction(
            signal, down, future, MarginRegimeInteractionConfig(min_group_n=20), turnover
        )
        self.assertEqual(result["status"], "estimated")
        self.assertIn("turnover_beta", result)
        self.assertAlmostEqual(result["interaction_beta"], .03, places=2)

    def test_fdr_uses_interaction_only_and_marks_nan(self):
        rows = pd.DataFrame({
            "family": ["margin_buy", "margin_sell", "margin_buy"],
            "model": ["binary_primary", "binary_primary", "secondary_continuous_prior"],
            "interaction_p": [.001, np.nan, 1e-10], "p_up": [.9, .001, 1.],
        })
        fdr, diagnostics = apply_interaction_fdr(rows)
        self.assertEqual(len(fdr), 2)
        self.assertEqual(fdr.loc[1, "evidence_level"], "Not Testable")
        total = diagnostics.loc[diagnostics.metric.eq("number_of_tests_total"), "value"].iloc[0]
        self.assertEqual(total, 2)
        self.assertEqual(set(fdr.fdr_scope), {FDR_SCOPE})

    def test_clusters_use_trading_adjacency_and_break_on_regime(self):
        index = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"])
        signals = pd.DataFrame({PREDICTOR: [1., 1., 1., 0.]}, index=index)
        regime = pd.Series(["Up", "Up", "Down", "Down"], index=index, dtype="string")
        outcomes = pd.DataFrame({"O1_C1": [1., 2., 3., 4.]}, index=index)
        specs = pd.DataFrame([{"predictor": PREDICTOR, "family": "margin_buy", "k": 1,
                               "rolling_window": 10, "pr_group": "PR95-100"}])
        result = regime_event_clusters(signals, regime, outcomes, specs)
        self.assertEqual(result.cluster_length.tolist(), [2, 1])
        self.assertEqual(result.first_signal_return.tolist(), [1., 3.])
        self.assertTrue((result.daily_N == 3).all())
        self.assertTrue((result.cluster_N == 2).all())

    def test_composition_chain_validation_stops_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, composition = frozen_dirs(Path(tmp))
            self.assertEqual(load_frozen_composition(composition)["run_info"]["git_commit"], COMPOSITION_COMMIT)
            info = composition / "run_info_composition.txt"
            info.write_text(info.read_text().replace(COMPOSITION_COMMIT, "wrong"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "git_commit mismatch"):
                load_frozen_composition(composition)

    def test_runner_is_incremental_and_exports_required_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline, turnover, composition = frozen_dirs(root)
            config = MarginRegimeInteractionConfig(
                min_group_n=2, output_root=root / "outputs_regime_interaction"
            )
            result = run_margin_regime_interaction_study(
                baseline, turnover, composition, config=config,
                datasets=synthetic_data(), export=True,
            )
            expected = {
                "run_info_regime_interaction.txt", "regime_signal_distribution.csv",
                "regime_primary_results.csv", "regime_interaction_results.csv",
                "regime_interaction_fdr_results.csv", "regime_interaction_fdr_diagnostics.csv",
                "regime_annual_results.csv", "regime_annual_robustness_summary.csv",
                "regime_event_clusters.csv", "regime_turnover_controlled_results.csv",
                "regime_summary.md",
            }
            self.assertTrue(expected.issubset({p.name for p in result["run_dir"].iterdir()}))
            info = (result["run_dir"] / "run_info_regime_interaction.txt").read_text()
            for value in (BASELINE_COMMIT, TURNOVER_BASE_COMMIT, COMPOSITION_COMMIT,
                          "price_source_open=etl:adj_open", "price_source_close=etl:adj_close"):
                self.assertIn(value, info)
            source = inspect.getsource(run_margin_regime_interaction_study)
            self.assertNotIn("pipeline.run", source)
            self.assertNotIn("run_margin_turnover_study", source)
            self.assertNotIn("run_margin_composition_study", source)


if __name__ == "__main__":
    unittest.main()
