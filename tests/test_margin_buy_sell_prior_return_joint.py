import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.margin_buy_sell_prior_return_joint import (
    ANNUAL_COLUMNS,
    ANNUAL_SUMMARY_COLUMNS,
    COLAB_REPO_DIR,
    FORMAL_BASELINE_RUN_DIR,
    FORMAL_COMPOSITION_RUN_DIR,
    FORMAL_DRIVE_OUTPUT_ROOT,
    FORMAL_REGIME_RUN_DIR,
    FORMAL_TURNOVER_RUN_DIR,
    INTERACTION_FDR_SCOPE,
    MAIN_EFFECT_FDR_SCOPE,
    REGIME_COMMIT,
    JointFlowPriorReturnConfig,
    _create_run_directory,
    annual_joint_results,
    apply_joint_fdr,
    fit_joint_model,
    joint_event_clusters,
    nested_model_comparison,
    prior_5d_return,
    run_margin_buy_sell_prior_return_joint_study,
    run_joint_models,
    select_joint_signal_pairs,
    validate_joint_input_runs,
)
from src.margin_composition import TURNOVER_BASE_COMMIT
from src.margin_regime_interaction import COMPOSITION_COMMIT
from src.margin_turnover import BASELINE_COMMIT


def baseline_fdr(k_values=(1,), windows=(10,)):
    rows = []
    for k in k_values:
        for window in windows:
            for family in ("margin_buy", "margin_sell"):
                for outcome in ("O1_C1", "O1_C2", "O1_C3", "O1_C5", "O1_C10", "O1_C20"):
                    rows.append({
                        "predictor": f"{family}__amount_ratio__k{k}__w{window}_percentile",
                        "family": family, "k": k, "rolling_window": window,
                        "pr_group": "PR95-100", "outcome_horizon": outcome,
                    })
    return pd.DataFrame(rows)


def frozen_dirs(root: Path):
    baseline, turnover = root / "baseline", root / "turnover"
    composition, regime = root / "composition", root / "regime"
    for directory in (baseline, turnover, composition, regime):
        directory.mkdir()
    adjusted = (
        "repository=taiwan-margin-short-0050-research\n"
        "price_source_open=etl:adj_open\nprice_source_close=etl:adj_close\n"
        "outcome_price_adjusted=True\n"
    )
    (baseline / "run_info.txt").write_text(
        adjusted + f"git_commit={BASELINE_COMMIT}\n", encoding="utf-8"
    )
    baseline_fdr().to_csv(baseline / "fdr_results.csv", index=False)
    for name in ("controlled_results.csv", "annual_robustness_summary.csv", "neighborhood_consistency.csv"):
        pd.DataFrame({"x": []}).to_csv(baseline / name, index=False)
    (turnover / "run_info_turnover.txt").write_text(
        adjusted + f"git_commit={TURNOVER_BASE_COMMIT}\nbaseline_commit={BASELINE_COMMIT}\n",
        encoding="utf-8",
    )
    for name in ("turnover_primary_results.csv", "turnover_fdr_results.csv",
                 "turnover_annual_results.csv", "turnover_annual_robustness_summary.csv",
                 "turnover_absorption_results.csv"):
        pd.DataFrame({"x": []}).to_csv(turnover / name, index=False)
    (composition / "run_info_composition.txt").write_text(
        adjusted + f"git_commit={COMPOSITION_COMMIT}\nbaseline_commit={BASELINE_COMMIT}\n"
        f"turnover_base_commit={TURNOVER_BASE_COMMIT}\n", encoding="utf-8",
    )
    for name in ("composition_primary_results.csv", "composition_fdr_results.csv",
                 "composition_annual_results.csv", "composition_annual_robustness_summary.csv",
                 "composition_absorption_results.csv"):
        pd.DataFrame({"x": []}).to_csv(composition / name, index=False)
    (regime / "run_info_regime_interaction.txt").write_text(
        adjusted + f"git_commit={REGIME_COMMIT}\nbaseline_commit={BASELINE_COMMIT}\n"
        f"turnover_commit={TURNOVER_BASE_COMMIT}\ncomposition_commit={COMPOSITION_COMMIT}\n",
        encoding="utf-8",
    )
    for name in ("regime_interaction_fdr_results.csv", "regime_secondary_continuous_results.csv",
                 "regime_turnover_controlled_results.csv"):
        pd.DataFrame({"x": []}).to_csv(regime / name, index=False)
    return baseline, turnover, composition, regime


def synthetic_data(periods=280):
    index = pd.bdate_range("2018-01-01", periods=periods)
    wave = np.sin(np.arange(periods) / 7)
    close = 100 + np.cumsum(.05 + wave / 20)
    buy = 100 + 50 * (1 + np.sin(np.arange(periods) / 11))
    sell = 100 + 50 * (1 + np.cos(np.arange(periods) / 13))
    return {
        "open": pd.DataFrame({"0050": close - .05}, index=index),
        "close": pd.DataFrame({"0050": close}, index=index),
        "margin_buy": pd.DataFrame({"2330": buy}, index=index),
        "margin_sell": pd.DataFrame({"2330": sell}, index=index),
        "margin_balance": pd.DataFrame({"2330": 1000.0}, index=index),
        "aggregate_buy": pd.DataFrame({"上市融資交易金額": buy * 100, "上櫃融資交易金額": buy * 10}, index=index),
        "aggregate_sell": pd.DataFrame({"上市融資交易金額": sell * 100, "上櫃融資交易金額": sell * 10}, index=index),
        "market_volume": pd.DataFrame({"TAIEX": 1e6, "OTC": 5e5}, index=index),
        "market_amount": pd.DataFrame({"TAIEX": 1e8, "OTC": 5e7}, index=index),
        "market_value": pd.DataFrame({"2330": 1e9}, index=index),
    }


def model_data(n=800):
    rng = np.random.default_rng(21)
    index = pd.bdate_range("2018-01-01", periods=n)
    buy = pd.Series(rng.binomial(1, .35, n), index=index, dtype=float)
    sell = pd.Series(rng.binomial(1, .35, n), index=index, dtype=float)
    prior = pd.Series(rng.normal(0, .03, n), index=index)
    noise = pd.Series(rng.normal(0, .002, n), index=index)
    future = .01 * buy + .02 * sell + .1 * prior + .3 * buy * prior - .4 * sell * prior + noise
    return buy, sell, prior, future


class JointFlowPriorReturnTests(unittest.TestCase):
    def test_frozen_input_paths_and_dependency_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            directories = frozen_dirs(Path(tmp))
            result = validate_joint_input_runs(*directories)
            self.assertEqual(result["regime"]["run_info"]["git_commit"], REGIME_COMMIT)
            self.assertEqual(set(result["diagnostics"].status), {"PASS"})

    def test_dependency_commit_mismatch_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            directories = frozen_dirs(Path(tmp))
            info = directories[3] / "run_info_regime_interaction.txt"
            info.write_text(info.read_text().replace(REGIME_COMMIT, "wrong"))
            with self.assertRaisesRegex(ValueError, "git_commit mismatch"):
                validate_joint_input_runs(*directories)

    def test_adjusted_price_mismatch_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            directories = frozen_dirs(Path(tmp))
            info = directories[2] / "run_info_composition.txt"
            info.write_text(info.read_text().replace("etl:adj_close", "price:收盤價"))
            with self.assertRaisesRegex(ValueError, "price_source_close mismatch"):
                validate_joint_input_runs(*directories)

    def test_same_k_window_matching_and_unmatched_rejection(self):
        config = JointFlowPriorReturnConfig(k_values=(1,), rolling_windows=(10,))
        pairs = select_joint_signal_pairs(baseline_fdr(), config)
        self.assertEqual(pairs[["k", "rolling_window"]].iloc[0].tolist(), [1, 10])
        with self.assertRaisesRegex(ValueError, "must match"):
            select_joint_signal_pairs(
                baseline_fdr().query("family == 'margin_buy'"), config
            )

    def test_prior_return_has_no_lookahead(self):
        close = pd.Series(np.arange(100.0, 112.0))
        original = prior_5d_return(close)
        changed = close.copy(); changed.iloc[-1] = 9999
        revised = prior_5d_return(changed)
        pd.testing.assert_series_equal(original.iloc[:-1], revised.iloc[:-1])
        self.assertAlmostEqual(original.iloc[5], .05)

    def test_joint_interactions_are_constructed_and_recovered(self):
        buy, sell, prior, future = model_data()
        fit, marginal, collinearity = fit_joint_model(
            buy, sell, prior, future,
            JointFlowPriorReturnConfig(min_joint_state_n=10, max_vif=50),
        )
        self.assertEqual(fit["status"], "estimated")
        self.assertAlmostEqual(fit["buy_prior_beta"], .3, places=1)
        self.assertAlmostEqual(fit["sell_prior_beta"], -.4, places=1)
        self.assertEqual(set(marginal.prior_level), {"p25", "p50", "p75"})
        self.assertEqual(set(collinearity.term), {"buy", "sell", "prior", "buy_prior", "sell_prior"})

    def test_marginal_effect_output_uses_named_p25_p50_p75_columns(self):
        buy, sell, prior, future = model_data()
        signals = pd.DataFrame({"buy": buy, "sell": sell})
        outcomes = pd.DataFrame({"O1_C1": future})
        pairs = pd.DataFrame([{"k": 1, "rolling_window": 10, "pr_group": "PR95-100",
                               "buy_signal": "buy", "sell_signal": "sell"}])
        _, marginal, _ = run_joint_models(
            signals, prior, outcomes, pairs,
            JointFlowPriorReturnConfig(k_values=(1,), rolling_windows=(10,),
                                       outcome_horizons=(1,), min_joint_state_n=10, max_vif=50),
        )
        for family in ("buy", "sell"):
            for level in ("p25", "p50", "p75"):
                self.assertIn(f"{family}_effect_{level}", marginal)
                self.assertIn(f"{family}_effect_{level}_ci_low", marginal)
                self.assertIn(f"{family}_effect_{level}_ci_high", marginal)

    def test_nested_models_a_b_c_and_attenuation(self):
        buy, sell, prior, future = model_data()
        index = buy.index
        signals = pd.DataFrame({"buy": buy, "sell": sell})
        outcomes = pd.DataFrame({"O1_C1": future}, index=index)
        pairs = pd.DataFrame([{"k": 1, "rolling_window": 10, "pr_group": "PR95-100",
                               "buy_signal": "buy", "sell_signal": "sell"}])
        result = nested_model_comparison(
            signals, prior, outcomes, pairs,
            JointFlowPriorReturnConfig(k_values=(1,), rolling_windows=(10,),
                                       outcome_horizons=(1,), min_joint_state_n=10, max_vif=50),
        )
        self.assertEqual(set(result.before_model), {"A", "B"})
        self.assertEqual(set(result.after_model), {"C"})
        self.assertTrue(result.attenuation_ratio.notna().all())

    def test_fdr_universes_are_separate(self):
        primary = pd.DataFrame([{
            "k": 1, "rolling_window": 10, "outcome": "O1_C1", "status": "estimated",
            "buy_beta": .01, "buy_p": .01, "sell_beta": .02, "sell_p": .2,
            "buy_prior_beta": .3, "buy_prior_p": .001,
            "sell_prior_beta": -.4, "sell_prior_p": .04,
        }])
        interaction, _, main, _ = apply_joint_fdr(primary)
        self.assertEqual(set(interaction.fdr_scope), {INTERACTION_FDR_SCOPE})
        self.assertEqual(set(main.fdr_scope), {MAIN_EFFECT_FDR_SCOPE})
        self.assertEqual(set(interaction.hypothesis), {"buy_x_prior", "sell_x_prior"})
        self.assertEqual(set(main.hypothesis), {"margin_buy", "margin_sell"})

    def test_default_primary_universes_have_192_tests_each(self):
        rows = []
        for k in (1, 3, 5, 10):
            for window in (126, 252, 504, 756):
                for outcome in ("O1_C1", "O1_C2", "O1_C3", "O1_C5", "O1_C10", "O1_C20"):
                    rows.append({
                        "k": k, "rolling_window": window, "outcome": outcome,
                        "status": "estimated", "buy_beta": .01, "buy_p": .5,
                        "sell_beta": .01, "sell_p": .5,
                        "buy_prior_beta": .01, "buy_prior_p": .5,
                        "sell_prior_beta": -.01, "sell_prior_p": .5,
                    })
        interaction, _, main, _ = apply_joint_fdr(pd.DataFrame(rows))
        self.assertEqual(len(rows), 96)
        self.assertEqual(len(interaction), 192)
        self.assertEqual(len(main), 192)

    def test_empty_level_a_b_annual_has_formal_schema(self):
        annual, summary = annual_joint_results(
            pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()
        )
        self.assertEqual(tuple(annual.columns), ANNUAL_COLUMNS)
        self.assertEqual(tuple(summary.columns), ANNUAL_SUMMARY_COLUMNS)

    def test_insufficient_joint_state_is_not_estimated(self):
        buy, sell, prior, future = model_data(100)
        sell[:] = 0
        fit, marginal, _ = fit_joint_model(
            buy, sell, prior, future, JointFlowPriorReturnConfig(min_joint_state_n=5)
        )
        self.assertEqual(fit["status"], "insufficient_group_sample")
        self.assertTrue(marginal.empty)

    def test_collinearity_is_flagged(self):
        buy, sell, prior, future = model_data(500)
        fit, _, _ = fit_joint_model(
            buy, sell, prior, future,
            JointFlowPriorReturnConfig(min_joint_state_n=5, max_vif=1.01),
        )
        self.assertEqual(fit["status"], "unstable_collinearity")

    def test_event_clustering_uses_trading_row_adjacency(self):
        index = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"])
        signals = pd.DataFrame({"buy": [1., 1., 0., 1.], "sell": [0., 1., 1., 0.]}, index=index)
        prior = pd.Series([.1, .2, -.1, .1], index=index)
        pairs = pd.DataFrame([{"k": 1, "rolling_window": 10, "buy_signal": "buy", "sell_signal": "sell"}])
        result = joint_event_clusters(signals, prior, pairs)
        buy = result[result.family.eq("margin_buy")]
        self.assertEqual(buy.cluster_length.tolist(), [2, 1])
        self.assertTrue((buy.daily_signal_N == 3).all())
        self.assertTrue((buy.independent_cluster_N == 2).all())

    def test_drive_output_naming(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = JointFlowPriorReturnConfig(output_root=Path(tmp))
            run_dir = _create_run_directory(config)
            self.assertTrue(run_dir.name.endswith("_margin_buy_sell_prior_return_joint"))

    def test_colab_contains_exact_paths_and_runner(self):
        notebook = json.loads(Path("notebooks/margin_buy_sell_prior_return_joint_colab.ipynb").read_text())
        source = "\n".join(line for cell in notebook["cells"] for line in cell.get("source", []))
        for path in (COLAB_REPO_DIR, FORMAL_BASELINE_RUN_DIR, FORMAL_TURNOVER_RUN_DIR,
                     FORMAL_COMPOSITION_RUN_DIR, FORMAL_REGIME_RUN_DIR, FORMAL_DRIVE_OUTPUT_ROOT):
            self.assertIn(str(path), source)
        self.assertIn("Dependency validation: PASS", source)
        self.assertIn("Adjusted price validation: PASS", source)

    def test_runner_exports_required_files_and_run_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directories = frozen_dirs(root)
            config = JointFlowPriorReturnConfig(
                k_values=(1,), rolling_windows=(10,), outcome_horizons=(1, 2, 3, 5, 10, 20),
                min_joint_state_n=2, max_vif=100,
                output_root=root / "output",
            )
            result = run_margin_buy_sell_prior_return_joint_study(
                *directories, config=config, datasets=synthetic_data(), export=True
            )
            expected = {
                "run_info_joint_flow_prior_return.txt", "joint_input_validation.csv",
                "joint_signal_overlap.csv", "joint_state_descriptive.csv", "joint_primary_results.csv",
                "joint_interaction_fdr_results.csv", "joint_interaction_fdr_diagnostics.csv",
                "joint_main_effect_fdr_results.csv", "joint_main_effect_fdr_diagnostics.csv",
                "joint_nested_model_comparison.csv", "joint_marginal_effects.csv",
                "joint_collinearity_diagnostics.csv", "joint_turnover_controlled_results.csv",
                "joint_event_clusters.csv", "joint_annual_results.csv",
                "joint_annual_robustness_summary.csv", "joint_summary.md",
            }
            self.assertTrue(expected.issubset({path.name for path in result["run_dir"].iterdir()}))
            info = (result["run_dir"] / "run_info_joint_flow_prior_return.txt").read_text()
            for key in ("repository=", "branch=", "git_commit=", "baseline_run_dir=",
                        "regime_commit=", "prior_return_definition=", "interaction_fdr_scope=",
                        "main_effect_fdr_scope=", "timezone=Asia/Taipei", "run_timestamp="):
                self.assertIn(key, info)


if __name__ == "__main__":
    unittest.main()
