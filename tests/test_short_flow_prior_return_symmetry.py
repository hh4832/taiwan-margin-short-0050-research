import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.margin_buy_sell_prior_return_joint import INTERACTION_FDR_SCOPE as MARGIN_INTERACTION_SCOPE
from src.margin_buy_sell_prior_return_joint import MAIN_EFFECT_FDR_SCOPE as MARGIN_MAIN_SCOPE
from src.short_flow_prior_return_symmetry import (
    BASELINE_FDR_SCOPE,
    COLAB_REPO_DIR,
    FORMAL_BASELINE_RUN_DIR,
    FORMAL_DRIVE_OUTPUT_ROOT,
    FORMAL_MARGIN_JOINT_ROOT,
    INTERACTION_FDR_SCOPE,
    MAIN_EFFECT_FDR_SCOPE,
    MARGIN_JOINT_COMMIT,
    PRIMARY_VARIANT,
    ShortFlowSymmetryConfig,
    _create_run_directory,
    _fit_short_model,
    _signal_frame,
    annual_short_robustness,
    build_short_flow_features,
    prior_5d_return,
    run_margin_short_symmetry,
    run_short_cluster_validation,
    run_short_crossover_analysis,
    run_short_flow_baseline,
    run_short_flow_prior_return_symmetry_study,
    run_short_prior_interaction,
    run_short_turnover_robustness,
    short_flow_definitions,
    short_overlap_diagnostics,
    validate_short_flow_inputs,
)
from src.margin_turnover import BASELINE_COMMIT


def frozen_inputs(root: Path):
    baseline, margin = root / "baseline", root / "margin_joint"
    baseline.mkdir(); margin.mkdir()
    adjusted = (
        "repository=taiwan-margin-short-0050-research\n"
        "price_source_open=etl:adj_open\nprice_source_close=etl:adj_close\n"
        "outcome_price_adjusted=True\n"
    )
    (baseline / "run_info.txt").write_text(adjusted + f"git_commit={BASELINE_COMMIT}\n")
    pd.DataFrame({
        "family": ["margin_cash_repayment"], "evidence_level": ["No Evidence"],
        "outcome_horizon": ["O1_C1"],
    }).to_csv(baseline / "fdr_results.csv", index=False)
    for name in ("reconciliation_summary.csv", "outcome_price_diagnostics.csv"):
        pd.DataFrame({"status": ["PASS"]}).to_csv(baseline / name, index=False)
    (margin / "run_info_joint_flow_prior_return.txt").write_text(
        adjusted + f"git_commit={MARGIN_JOINT_COMMIT}\nbaseline_commit={BASELINE_COMMIT}\n"
        "prior_return_definition=adjusted_close[t] / adjusted_close[t-5] - 1\n"
        f"interaction_fdr_scope={MARGIN_INTERACTION_SCOPE}\n"
        f"main_effect_fdr_scope={MARGIN_MAIN_SCOPE}\n"
        "k=(1, 3, 5, 10)\nrolling_windows=(126, 252, 504, 756)\n"
        "outcome_horizons=(1, 2, 3, 5, 10, 20)\n"
    )
    pd.DataFrame({"status": ["PASS"]}).to_csv(margin / "joint_input_validation.csv", index=False)
    pd.DataFrame({"status": ["PASS"]}).to_csv(margin / "outcome_price_diagnostics.csv", index=False)
    pd.DataFrame({
        "hypothesis": ["buy_x_prior", "sell_x_prior"],
        "evidence_level": ["No Evidence", "No Evidence"], "beta": [.1, -.1],
    }).to_csv(margin / "joint_interaction_fdr_results.csv", index=False)
    pd.DataFrame({
        "hypothesis": ["margin_buy", "margin_sell"],
        "evidence_level": ["No Evidence", "No Evidence"], "beta": [.1, -.1],
    }).to_csv(margin / "joint_main_effect_fdr_results.csv", index=False)
    for name in ("joint_turnover_controlled_results.csv", "joint_event_clusters.csv",
                 "joint_marginal_effects.csv", "joint_nested_model_comparison.csv",
                 "joint_annual_robustness_summary.csv"):
        pd.DataFrame({"x": []}).to_csv(margin / name, index=False)
    return baseline, margin


def synthetic_datasets(periods=320):
    index = pd.bdate_range("2018-01-01", periods=periods)
    rng = np.random.default_rng(9)
    close = 100 + np.cumsum(rng.normal(.05, .3, periods))
    frames = {
        "short_sell": pd.DataFrame({"2330": rng.integers(10, 100, periods)}, index=index),
        "short_cover": pd.DataFrame({"2330": rng.integers(10, 100, periods)}, index=index),
        "short_stock_repayment": pd.DataFrame({"2330": rng.integers(2, 50, periods)}, index=index),
        "short_balance": pd.DataFrame({"2330": 1000.0}, index=index),
        "market_volume": pd.DataFrame({"TAIEX": 1e7, "OTC": 5e6}, index=index),
        "open": pd.DataFrame({"0050": close - .1}, index=index),
        "close": pd.DataFrame({"0050": close}, index=index),
    }
    suspension = pd.DataFrame(columns=["symbol", "停券起日(最後回補日)", "停券迄日"])
    suspension.attrs["quarantined_rows"] = 0
    frames["short_suspension"] = suspension
    return frames


def manual_features(n=900):
    rng = np.random.default_rng(4)
    index = pd.bdate_range("2018-01-01", periods=n)
    values, catalog = {}, {}
    for flow in ("short_sell", "short_cover", "short_repayment"):
        name = f"{flow}_raw__k1__volume_ratio__w10_percentile"
        values[name] = pd.Series(np.where(rng.binomial(1, .25, n), 100.0, 50.0), index=index)
        catalog[name] = {"family": flow, "analysis_variant": "raw", "k": 1,
                         "window": 10, "normalization": PRIMARY_VARIANT,
                         "primary": True, "retrospective_sensitivity": False}
    features = pd.DataFrame(values)
    features.attrs["catalog"] = catalog
    return features


def model_data(n=1000):
    rng = np.random.default_rng(8)
    index = pd.bdate_range("2018-01-01", periods=n)
    flows = pd.DataFrame({flow: rng.binomial(1, .3, n).astype(float)
                          for flow in ("short_sell", "short_cover", "short_repayment")}, index=index)
    prior = pd.Series(rng.normal(0, .03, n), index=index)
    future = (.01 * flows.short_sell + .02 * flows.short_cover - .01 * flows.short_repayment
              + .1 * prior + .3 * flows.short_sell * prior
              - .4 * flows.short_cover * prior + .2 * flows.short_repayment * prior
              + rng.normal(0, .003, n))
    return flows, prior, future


class ShortFlowPriorReturnTests(unittest.TestCase):
    def test_adjusted_price_and_frozen_commits_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = validate_short_flow_inputs(*frozen_inputs(Path(tmp)))
            self.assertEqual(result["margin_joint_info"]["git_commit"], MARGIN_JOINT_COMMIT)
            self.assertTrue(result["diagnostics"].status.str.startswith("PASS").all())

    def test_commit_or_adjusted_source_mismatch_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline, margin = frozen_inputs(Path(tmp))
            info = margin / "run_info_joint_flow_prior_return.txt"
            info.write_text(info.read_text().replace(MARGIN_JOINT_COMMIT, "wrong"))
            with self.assertRaisesRegex(ValueError, "git_commit mismatch"):
                validate_short_flow_inputs(baseline, margin)
        with tempfile.TemporaryDirectory() as tmp:
            baseline, margin = frozen_inputs(Path(tmp))
            info = baseline / "run_info.txt"
            info.write_text(info.read_text().replace("etl:adj_open", "price:開盤價"))
            with self.assertRaisesRegex(ValueError, "price_source_open mismatch"):
                validate_short_flow_inputs(baseline, margin)

    def test_three_short_definitions_are_distinct_and_not_invented_amounts(self):
        definitions = short_flow_definitions()
        self.assertIn("融券賣出", definitions["short_sell"])
        self.assertIn("融券買進", definitions["short_cover"])
        self.assertIn("融券現券償還", definitions["short_repayment"])
        self.assertNotEqual(definitions["short_cover"], definitions["short_repayment"])
        self.assertEqual(PRIMARY_VARIANT, "volume_ratio")

    def test_feature_grid_alignment_and_adjusted_copy(self):
        config = ShortFlowSymmetryConfig(k_values=(1,), rolling_windows=(10,), outcome_horizons=(1,))
        features, catalog, controls = build_short_flow_features(synthetic_datasets(), config)
        self.assertEqual(len(catalog), 6)
        self.assertEqual(set(catalog.analysis_variant), {"raw", "adjusted"})
        self.assertEqual(catalog[catalog.primary].family.nunique(), 3)
        self.assertIn((1, 10), controls)
        self.assertEqual(set(_signal_frame(features, 1, 10)), {"short_sell", "short_cover", "short_repayment"})

    def test_short_flow_volume_ratios_use_distinct_source_fields(self):
        datasets = synthetic_datasets(30)
        config = ShortFlowSymmetryConfig(k_values=(1,), rolling_windows=(10,), outcome_horizons=(1,))
        features, _, _ = build_short_flow_features(datasets, config)
        ratios = features.attrs["raw_ratio_series"]
        denominator = datasets["market_volume"][["TAIEX", "OTC"]].sum(axis=1)
        for flow, source in (("short_sell", "short_sell"), ("short_cover", "short_cover"),
                             ("short_repayment", "short_stock_repayment")):
            expected = datasets[source].sum(axis=1) * 1000 / denominator
            pd.testing.assert_series_equal(ratios[("raw", flow, 1)], expected)

    def test_prior_return_no_lookahead(self):
        close = pd.Series(np.arange(100.0, 112.0))
        original = prior_5d_return(close)
        changed = close.copy(); changed.iloc[-1] = 9999
        pd.testing.assert_series_equal(original.iloc[:-1], prior_5d_return(changed).iloc[:-1])

    def test_outcome_alignment_and_baseline_fdr(self):
        features = manual_features()
        index = features.index
        outcomes = pd.DataFrame({"O1_C1": np.linspace(-.01, .02, len(index))}, index=index)
        primary, fdr, diagnostics = run_short_flow_baseline(
            features, outcomes,
            ShortFlowSymmetryConfig(k_values=(1,), rolling_windows=(10,), outcome_horizons=(1,))
        )
        self.assertEqual(len(primary), 3)
        self.assertEqual(set(fdr.fdr_scope), {BASELINE_FDR_SCOPE})
        self.assertEqual(diagnostics.loc[0, "value"], 3)

    def test_joint_interaction_and_nested_models(self):
        flows, prior, future = model_data()
        fit, marginal, collinearity = _fit_short_model(
            flows, prior, future, ShortFlowSymmetryConfig(min_group_n=10, max_vif=50)
        )
        self.assertEqual(fit["status"], "estimated")
        self.assertAlmostEqual(fit["short_sell_prior_beta"], .3, places=1)
        self.assertAlmostEqual(fit["short_cover_prior_beta"], -.4, places=1)
        self.assertEqual(set(marginal.prior_level), {"p25", "p50", "p75"})
        self.assertIn("short_repayment_prior", set(collinearity.term))

    def test_interaction_and_main_fdr_are_separate(self):
        features = manual_features()
        flows = _signal_frame(features, 1, 10)
        rng = np.random.default_rng(10)
        prior = pd.Series(rng.normal(0, .03, len(features)), index=features.index)
        future = (.2 * flows.short_sell * prior - .2 * flows.short_cover * prior
                  + rng.normal(0, .01, len(features)))
        outcomes = pd.DataFrame({"O1_C1": future}, index=features.index)
        result = run_short_prior_interaction(
            features, prior, outcomes,
            ShortFlowSymmetryConfig(k_values=(1,), rolling_windows=(10,), outcome_horizons=(1,),
                                    min_group_n=10, max_vif=50),
        )
        self.assertEqual(set(result["interaction_fdr"].fdr_scope), {INTERACTION_FDR_SCOPE})
        self.assertEqual(set(result["main_effect_fdr"].fdr_scope), {MAIN_EFFECT_FDR_SCOPE})
        self.assertEqual(len(result["interaction_fdr"]), 3)
        self.assertEqual(len(result["nested_comparison"]), 6)

    def test_marginal_effect_formula(self):
        flows, prior, future = model_data()
        fit, marginal, _ = _fit_short_model(
            flows, prior, future, ShortFlowSymmetryConfig(min_group_n=10, max_vif=50)
        )
        row = marginal[(marginal.flow == "short_sell") & (marginal.prior_level == "p25")].iloc[0]
        expected = fit["short_sell_beta"] + fit["short_sell_prior_beta"] * row.prior_value
        self.assertAlmostEqual(row.effect, expected)
        self.assertLess(row.ci_low, row.ci_high)

    def test_turnover_robustness_compares_before_after(self):
        features = manual_features()
        rng = np.random.default_rng(11)
        prior = pd.Series(rng.normal(0, .03, len(features)), index=features.index)
        outcomes = pd.DataFrame({"O1_C1": rng.normal(0, .01, len(features))}, index=features.index)
        interaction = run_short_prior_interaction(
            features, prior, outcomes,
            ShortFlowSymmetryConfig(k_values=(1,), rolling_windows=(10,), outcome_horizons=(1,),
                                    min_group_n=10, max_vif=50),
        )
        controls = {(1, 10): pd.Series(rng.uniform(0, 1, len(features)), index=features.index)}
        result = run_short_turnover_robustness(
            features, prior, outcomes, interaction["joint_results"], controls,
            ShortFlowSymmetryConfig(k_values=(1,), rolling_windows=(10,), outcome_horizons=(1,),
                                    min_group_n=10, max_vif=50),
        )
        self.assertEqual(len(result), 6)
        self.assertTrue({"coefficient_before", "coefficient_after", "attenuation_ratio", "classification"}.issubset(result))

    def test_cluster_construction_and_hac_inference(self):
        features = manual_features()
        prior = pd.Series(np.linspace(-.05, .05, len(features)), index=features.index)
        outcomes = pd.DataFrame({"O1_C1": np.sin(np.arange(len(features))) / 100}, index=features.index)
        candidate = pd.DataFrame([{"flow": "short_sell", "k": 1, "rolling_window": 10,
                                   "outcome": "O1_C1", "evidence_level": "Level A"}])
        result = run_short_cluster_validation(
            features, prior, outcomes, candidate,
            ShortFlowSymmetryConfig(k_values=(1,), rolling_windows=(10,), outcome_horizons=(1,),
                                    min_group_n=10, max_vif=50),
        )
        summary = result[result.record_type.eq("cluster_summary")]
        hac = result[result.record_type.eq("HAC_inference")]
        self.assertEqual(len(summary), 3)
        self.assertEqual(len(hac), 1)
        self.assertIn("Newey-West HAC", hac.cluster_inference_method.iloc[0])

    def test_crossover_formula_near_zero_and_extrapolation(self):
        prior = pd.Series(np.linspace(-.05, .05, 1000))
        common = {"flow": "short_sell", "k": 1, "rolling_window": 126, "outcome": "O1_C1",
                  "evidence_level": "Level A", "main_variance": 1e-8,
                  "interaction_variance": 1e-6, "main_interaction_covariance": 0.0}
        stable = run_short_crossover_analysis(pd.DataFrame([common | {"main_beta": .001, "beta": .1}]), prior)
        self.assertAlmostEqual(stable.crossover_prior_return.iloc[0], -.01)
        near = run_short_crossover_analysis(pd.DataFrame([common | {"main_beta": .001, "beta": 1e-12}]), prior)
        self.assertEqual(near.crossover_status.iloc[0], "unstable_crossover")
        extra = run_short_crossover_analysis(pd.DataFrame([common | {"main_beta": .1, "beta": .1}]), prior)
        self.assertEqual(extra.crossover_status.iloc[0], "extrapolation_crossover")

    def test_signal_overlap_and_collinearity_warning(self):
        features = manual_features()
        overlap = short_overlap_diagnostics(
            features, ShortFlowSymmetryConfig(k_values=(1,), rolling_windows=(10,))
        )
        self.assertTrue({"SS_SC_both_high_N", "all_three_high_N", "phi_short_sell_short_cover"}.issubset(overlap))
        flows, prior, future = model_data(500)
        fit, _, _ = _fit_short_model(
            flows, prior, future, ShortFlowSymmetryConfig(min_group_n=5, max_vif=1.01)
        )
        self.assertEqual(fit["status"], "unstable_collinearity")

    def test_annual_insufficient_sample_is_unable_to_determine(self):
        features = manual_features(120)
        prior = pd.Series(np.linspace(-.02, .02, len(features)), index=features.index)
        outcomes = pd.DataFrame({"O1_C1": np.linspace(-.01, .01, len(features))}, index=features.index)
        candidates = pd.DataFrame([{"flow": "short_sell", "k": 1, "rolling_window": 10,
                                    "outcome": "O1_C1", "evidence_level": "Level A"}])
        result = annual_short_robustness(
            candidates, features, prior, outcomes,
            ShortFlowSymmetryConfig(k_values=(1,), rolling_windows=(10,), outcome_horizons=(1,), min_group_n=50),
        )
        self.assertTrue(result.status.eq("Unable to Determine").all())

    def test_symmetry_does_not_force_cash_repayment_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = validate_short_flow_inputs(*frozen_inputs(Path(tmp)))
            short = pd.DataFrame({"flow": ["short_sell", "short_cover", "short_repayment"],
                                  "evidence_level": ["No Evidence"] * 3, "beta": [1., -1., 1.]})
            clusters = pd.DataFrame({"record_type": ["cluster_summary"] * 3,
                                     "flow": ["short_sell", "short_cover", "short_repayment"],
                                     "cluster_daily_ratio": [.5, .5, .5]})
            turnover = pd.DataFrame({"flow": ["short_sell", "short_cover", "short_repayment"],
                                     "classification": ["survive", "attenuate", "not_testable"]})
            result = run_margin_short_symmetry(inputs, short, short, turnover, clusters)
            cash = result[result.Pair.str.startswith("Cash Repayment")].iloc[0]
            self.assertEqual(cash["Symmetry status"], "Unable to Determine")

    def test_drive_name_notebook_and_run_info_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _create_run_directory(ShortFlowSymmetryConfig(output_root=Path(tmp)))
            self.assertTrue(run_dir.name.endswith("_short_flow_prior_return_symmetry"))
        notebook = json.loads(Path("notebooks/run_short_flow_prior_return_symmetry.ipynb").read_text())
        source = "\n".join(line for cell in notebook["cells"] for line in cell.get("source", []))
        for path in (COLAB_REPO_DIR, FORMAL_BASELINE_RUN_DIR, FORMAL_MARGIN_JOINT_ROOT,
                     FORMAL_DRIVE_OUTPUT_ROOT):
            self.assertIn(str(path), source)
        self.assertIn("run_short_flow_prior_return_symmetry_study", source)

    def test_integrated_runner_exports_compact_required_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline, margin = frozen_inputs(root)
            config = ShortFlowSymmetryConfig(k_values=(1,), rolling_windows=(10,), outcome_horizons=(1,),
                                             min_group_n=2, max_vif=100, output_root=root / "outputs")
            datasets = synthetic_datasets()
            stage = {"datasets": datasets, "coverage": pd.DataFrame(), "reconciliation": pd.DataFrame()}
            result = run_short_flow_prior_return_symmetry_study(
                baseline, margin, config=config, export=True, stage_zero=stage
            )
            expected = {
                "short_flow_primary_results.csv", "short_flow_fdr_results.csv",
                "short_flow_fdr_diagnostics.csv", "short_flow_prior_interaction_results.csv",
                "short_flow_joint_model_results.csv", "short_flow_main_effect_fdr_results.csv",
                "short_flow_marginal_effects.csv", "short_flow_nested_comparison.csv",
                "short_flow_turnover_robustness.csv", "short_flow_cluster_robustness.csv",
                "short_flow_crossover.csv", "short_flow_overlap_diagnostics.csv",
                "short_flow_collinearity.csv", "margin_short_symmetry.csv",
                "short_flow_annual_robustness.csv", "short_flow_summary.md",
                "input_validation.csv", "run_info.txt",
            }
            self.assertEqual({p.name for p in result["run_dir"].iterdir()}, expected)
            info = (result["run_dir"] / "run_info.txt").read_text()
            for key in ("repository=", "branch=", "git_commit=", "baseline_run_dir=",
                        "margin_joint_commit=", "short_sell_definition=", "short_cover_definition=",
                        "short_repayment_definition=", "short_turnover_definition=",
                        "prior_return_definition=", "cluster_inference_method=", "timezone=Asia/Taipei",
                        "python_version=", "finlab_version="):
                self.assertIn(key, info)


if __name__ == "__main__":
    unittest.main()
