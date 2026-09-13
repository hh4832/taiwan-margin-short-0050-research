from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import stats

from .features import adjust_short_for_suspensions, rolling_ratio, trailing_percentile
from .margin_buy_sell_prior_return_joint import (
    INTERACTION_FDR_SCOPE as MARGIN_INTERACTION_SCOPE,
    MAIN_EFFECT_FDR_SCOPE as MARGIN_MAIN_SCOPE,
    _vif,
)
from .margin_regime_interaction import _bh_adjust, _hc3_ols, prior_5d_return
from .margin_turnover import BASELINE_COMMIT, _read_run_info, _sum_columns
from .outcomes import build_outcomes
from .pipeline import run_stage_zero
from .price_validation import validate_split_window
from .statistics import compare_group


MARGIN_JOINT_COMMIT = "4a6fb82795342a55d174dbfc9beb1ee80a93bb4c"
BASELINE_FDR_SCOPE = "short_flow_baseline_primary"
INTERACTION_FDR_SCOPE = "short_flow_prior_return_interactions"
MAIN_EFFECT_FDR_SCOPE = "short_flow_incremental_main_effects"
PRIMARY_VARIANT = "volume_ratio"
COLAB_REPO_DIR = Path("/content/taiwan-margin-short-0050-research")
FORMAL_BASELINE_RUN_DIR = Path(
    "/content/drive/MyDrive/Quant_Research/taiwan-margin-short-0050-research/"
    "20260912_220859_88d9f26_adjusted_price"
)
FORMAL_MARGIN_JOINT_ROOT = Path(
    "/content/drive/MyDrive/Quant_Research/taiwan-margin-short-0050-research/"
    "margin_buy_sell_prior_return_joint"
)
FORMAL_DRIVE_OUTPUT_ROOT = Path(
    "/content/drive/MyDrive/Quant_Research/taiwan-margin-short-0050-research/"
    "short_flow_prior_return_symmetry"
)
FLOW_KEYS = ("short_sell", "short_cover", "short_repayment")
FLOW_LABELS = {
    "short_sell": "Short Sell",
    "short_cover": "Short Cover",
    "short_repayment": "Short Repayment",
}
DATASET_KEYS = {
    "short_sell": "short_sell",
    "short_cover": "short_cover",
    "short_repayment": "short_stock_repayment",
}
OUTCOMES = tuple(f"O1_C{h}" for h in (1, 2, 3, 5, 10, 20))
INTERACTION_COLUMNS = {
    "short_sell": ("short_sell_beta", "short_sell_p", "short_sell_prior_beta", "short_sell_prior_p"),
    "short_cover": ("short_cover_beta", "short_cover_p", "short_cover_prior_beta", "short_cover_prior_p"),
    "short_repayment": ("short_repayment_beta", "short_repayment_p", "short_repayment_prior_beta", "short_repayment_prior_p"),
}


@dataclass(frozen=True)
class ShortFlowSymmetryConfig:
    repository: str = "taiwan-margin-short-0050-research"
    timezone: str = "Asia/Taipei"
    target_symbol: str = "0050"
    baseline_commit: str = BASELINE_COMMIT
    margin_joint_commit: str = MARGIN_JOINT_COMMIT
    k_values: tuple[int, ...] = (1, 3, 5, 10)
    rolling_windows: tuple[int, ...] = (126, 252, 504, 756)
    outcome_horizons: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    primary_tail: str = "PR95-100"
    min_group_n: int = 20
    max_vif: float = 10.0
    max_condition_number: float = 1e12
    leverage_tolerance: float = 1e-10
    crossover_denominator_epsilon: float = 1e-8
    robustness_min_years: int = 3
    robustness_min_direction_ratio: float = 0.60
    robustness_max_year_sample_share: float = 0.50
    output_root: Path = Path("outputs_short_flow_prior_return_symmetry")


def _git_value(args: list[str], default: str = "unknown") -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return default


def _require_values(layer: str, info: dict[str, str], expected: dict[str, str]) -> None:
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(
                f"{layer} {key} mismatch: expected {value!r}, got {info.get(key)!r}"
            )


def _validate_artifact_status(layer: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError(f"{layer} validation artifact is empty")
    if "status" in frame:
        failed = frame["status"].astype(str).str.lower().str.contains("fail|error")
        if failed.any():
            raise ValueError(f"{layer} validation contains failed rows")


def _read_csv_allow_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def validate_short_flow_inputs(
    baseline_run_dir: str | Path,
    margin_joint_run_dir: str | Path,
    config: ShortFlowSymmetryConfig | None = None,
) -> dict[str, object]:
    """Validate exact frozen baseline and margin-joint artifacts; never discover latest runs."""
    config = config or ShortFlowSymmetryConfig()
    baseline = Path(baseline_run_dir).expanduser()
    margin = Path(margin_joint_run_dir).expanduser()
    required = {
        "Baseline": (
            "run_info.txt", "reconciliation_summary.csv", "outcome_price_diagnostics.csv",
            "fdr_results.csv",
        ),
        "Margin joint": (
            "run_info_joint_flow_prior_return.txt", "joint_input_validation.csv",
            "outcome_price_diagnostics.csv", "joint_interaction_fdr_results.csv",
            "joint_main_effect_fdr_results.csv", "joint_turnover_controlled_results.csv",
            "joint_event_clusters.csv", "joint_marginal_effects.csv",
            "joint_nested_model_comparison.csv", "joint_annual_robustness_summary.csv",
        ),
    }
    for layer, root in (("Baseline", baseline), ("Margin joint", margin)):
        if not root.is_dir():
            raise FileNotFoundError(f"{layer} run directory does not exist: {root}")
        missing = [name for name in required[layer] if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{layer} run directory is missing required files: {missing}")
    baseline_info = _read_run_info(baseline / "run_info.txt")
    margin_info = _read_run_info(margin / "run_info_joint_flow_prior_return.txt")
    adjusted = {
        "repository": config.repository,
        "price_source_open": "etl:adj_open",
        "price_source_close": "etl:adj_close",
        "outcome_price_adjusted": "True",
    }
    _require_values("Baseline", baseline_info, adjusted | {"git_commit": config.baseline_commit})
    _require_values("Margin joint", margin_info, adjusted | {
        "git_commit": config.margin_joint_commit,
        "baseline_commit": config.baseline_commit,
        "prior_return_definition": "adjusted_close[t] / adjusted_close[t-5] - 1",
        "interaction_fdr_scope": MARGIN_INTERACTION_SCOPE,
        "main_effect_fdr_scope": MARGIN_MAIN_SCOPE,
        "k": "(1, 3, 5, 10)",
        "rolling_windows": "(126, 252, 504, 756)",
        "outcome_horizons": "(1, 2, 3, 5, 10, 20)",
    })
    baseline_reconciliation = pd.read_csv(baseline / "reconciliation_summary.csv")
    baseline_price = pd.read_csv(baseline / "outcome_price_diagnostics.csv")
    margin_validation = pd.read_csv(margin / "joint_input_validation.csv")
    margin_price = pd.read_csv(margin / "outcome_price_diagnostics.csv")
    for layer, artifact in (
        ("Baseline reconciliation", baseline_reconciliation),
        ("Baseline price", baseline_price),
        ("Margin joint input", margin_validation),
        ("Margin joint price", margin_price),
    ):
        _validate_artifact_status(layer, artifact)
    diagnostics = pd.DataFrame([
        {"layer": "Baseline", "run_dir": str(baseline), "commit": baseline_info["git_commit"],
         "price_source_open": baseline_info["price_source_open"],
         "price_source_close": baseline_info["price_source_close"],
         "outcome_price_adjusted": baseline_info["outcome_price_adjusted"], "status": "PASS"},
        {"layer": "Margin joint", "run_dir": str(margin), "commit": margin_info["git_commit"],
         "baseline_commit": margin_info["baseline_commit"],
         "price_source_open": margin_info["price_source_open"],
         "price_source_close": margin_info["price_source_close"],
         "outcome_price_adjusted": margin_info["outcome_price_adjusted"], "status": "PASS"},
        {"layer": "Short normalization", "run_dir": "not_applicable", "commit": "not_applicable",
         "status": "PASS_WITH_LIMITATION", "primary_variant": PRIMARY_VARIANT,
         "limitation": "Baseline has no native short amount_ratio; existing volume_ratio is used without inventing notional amounts."},
    ])
    return {
        "baseline_dir": baseline, "margin_joint_dir": margin,
        "baseline_info": baseline_info, "margin_joint_info": margin_info,
        "baseline_fdr": pd.read_csv(baseline / "fdr_results.csv"),
        "baseline_reconciliation": baseline_reconciliation,
        "baseline_price_diagnostics": baseline_price,
        "margin_input_validation": margin_validation,
        "margin_price_diagnostics": margin_price,
        "margin_interaction_fdr": pd.read_csv(margin / "joint_interaction_fdr_results.csv"),
        "margin_main_fdr": pd.read_csv(margin / "joint_main_effect_fdr_results.csv"),
        "margin_turnover": _read_csv_allow_empty(margin / "joint_turnover_controlled_results.csv"),
        "margin_clusters": _read_csv_allow_empty(margin / "joint_event_clusters.csv"),
        "margin_marginal": _read_csv_allow_empty(margin / "joint_marginal_effects.csv"),
        "margin_nested": _read_csv_allow_empty(margin / "joint_nested_model_comparison.csv"),
        "margin_annual": _read_csv_allow_empty(margin / "joint_annual_robustness_summary.csv"),
        "diagnostics": diagnostics,
    }


def short_flow_definitions() -> dict[str, str]:
    denominator = "TAIEX+OTC market traded shares; ratio of k-day sums"
    return {
        "short_sell": f"融券賣出 lots × 1000 / {denominator}",
        "short_cover": f"融券買進 lots × 1000 / {denominator}",
        "short_repayment": f"融券現券償還 lots × 1000 / {denominator}",
        "short_turnover": f"(融券賣出 lots + 融券買進 lots) × 1000 / {denominator}; repayment excluded",
    }


def build_short_flow_features(
    datasets: dict[str, pd.DataFrame], config: ShortFlowSymmetryConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[int, int], pd.Series]]:
    """Build baseline-consistent raw and suspension-adjusted volume-ratio percentiles."""
    config = config or ShortFlowSymmetryConfig()
    market_volume = _sum_columns(datasets["market_volume"], ("TAIEX", "OTC"))
    raw_frames = {key: datasets[key] for key in ("short_sell", "short_cover", "short_stock_repayment", "short_balance")}
    adjusted, _ = adjust_short_for_suspensions(raw_frames, datasets["short_suspension"])
    values: dict[str, pd.Series] = {}
    catalog: dict[str, dict] = {}
    raw_ratio_series: dict[tuple[str, str, int], pd.Series] = {}
    turnover_controls: dict[tuple[int, int], pd.Series] = {}
    for variant, frames in (("raw", raw_frames), ("adjusted", adjusted)):
        base: dict[str, pd.Series] = {}
        for flow in FLOW_KEYS:
            base[flow] = frames[DATASET_KEYS[flow]].sum(axis=1, min_count=1) * 1000.0
        for k in config.k_values:
            ratios = {flow: rolling_ratio(base[flow], market_volume, k) for flow in FLOW_KEYS}
            raw_ratio_series.update({(variant, flow, k): ratio for flow, ratio in ratios.items()})
            turnover = rolling_ratio(base["short_sell"] + base["short_cover"], market_volume, k)
            for window in config.rolling_windows:
                if variant == "raw":
                    turnover_controls[(k, window)] = trailing_percentile(turnover, window) / 100.0
                for flow, ratio in ratios.items():
                    name = f"{flow}_{variant}__k{k}__volume_ratio__w{window}_percentile"
                    values[name] = trailing_percentile(ratio, window)
                    catalog[name] = {
                        "family": flow, "analysis_variant": variant, "k": k,
                        "window": window, "normalization": PRIMARY_VARIANT,
                        "primary": variant == "raw", "retrospective_sensitivity": variant == "adjusted",
                    }
    frame = pd.DataFrame(values).sort_index()
    frame.attrs["catalog"] = catalog
    frame.attrs["raw_ratio_series"] = raw_ratio_series
    return frame, pd.DataFrame([{"predictor": key, **value} for key, value in catalog.items()]), turnover_controls


def _primary_signal_specs(features: pd.DataFrame) -> pd.DataFrame:
    catalog = pd.DataFrame([
        {"predictor": predictor, **meta} for predictor, meta in features.attrs.get("catalog", {}).items()
    ])
    primary = catalog[catalog.primary.astype(bool)].copy()
    counts = primary.groupby(["k", "window"]).family.nunique()
    bad = counts[counts.ne(3)]
    if len(bad):
        raise ValueError(f"Short flows are not aligned on identical k/window: {bad.to_dict()}")
    return primary.sort_values(["k", "window", "family"]).reset_index(drop=True)


def _apply_fdr(frame: pd.DataFrame, p_col: str, family_col: str, scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = frame.copy()
    out["family_fdr_q_value"] = np.nan
    for _, index in out.groupby(family_col).groups.items():
        use = out.loc[index].index[out.loc[index, p_col].notna()]
        if len(use):
            out.loc[use, "family_fdr_q_value"] = _bh_adjust(out.loc[use, p_col])
    out["global_fdr_q_value"] = np.nan
    use = out.index[out[p_col].notna()]
    if len(use):
        out.loc[use, "global_fdr_q_value"] = _bh_adjust(out.loc[use, p_col])
    out["evidence_level"] = "No Evidence"
    out.loc[out[p_col].lt(.05), "evidence_level"] = "Level C"
    out.loc[out.family_fdr_q_value.lt(.05), "evidence_level"] = "Level B"
    out.loc[out.global_fdr_q_value.lt(.05), "evidence_level"] = "Level A"
    out.loc[out[p_col].isna(), "evidence_level"] = "Not Testable"
    out["fdr_scope"] = scope
    levels = ("Level A", "Level B", "Level C", "No Evidence", "Not Testable")
    diagnostics = pd.DataFrame(
        [{"metric": "number_of_tests_total", "value": len(out)},
         {"metric": "number_of_tests_estimable", "value": int(out[p_col].notna().sum())},
         {"metric": "number_of_tests_not_testable", "value": int(out[p_col].isna().sum())}]
        + [{"metric": level, "value": int(out.evidence_level.eq(level).sum())} for level in levels]
    )
    diagnostics["fdr_scope"] = scope
    return out, diagnostics


def run_short_flow_baseline(
    features: pd.DataFrame, outcomes: pd.DataFrame, config: ShortFlowSymmetryConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = config or ShortFlowSymmetryConfig()
    specs = _primary_signal_specs(features)
    rows = []
    for spec in specs.itertuples():
        percentile = features[spec.predictor]
        active = percentile.ge(95).astype(float).where(percentile.notna())
        for outcome in [name for name in OUTCOMES if name in outcomes]:
            values = outcomes[outcome].where(percentile.notna())
            result = compare_group(values, active.eq(1))
            control_n = int((active.eq(0) & values.notna()).sum())
            signal_values = values[active.eq(1)].dropna()
            control_values = values[active.eq(0)].dropna()
            effect_vs_nongroup = (
                signal_values.mean() - control_values.mean()
                if len(signal_values) and len(control_values) else np.nan
            )
            if result["N"] < config.min_group_n or control_n < config.min_group_n:
                result["raw_p_value"] = np.nan
                status = "insufficient_group_sample"
            else:
                status = "estimated"
            rows.append({
                "predictor": spec.predictor, "family": spec.family,
                "normalization": PRIMARY_VARIANT, "pr_group": config.primary_tail,
                "k": int(spec.k), "rolling_window": int(spec.window),
                "outcome": outcome, "control_N": control_n, "status": status, **result,
                "effect_vs_nongroup": effect_vs_nongroup,
            })
    primary = pd.DataFrame(rows)
    fdr, diagnostics = _apply_fdr(primary, "raw_p_value", "family", BASELINE_FDR_SCOPE)
    return primary, fdr, diagnostics


def _nw_fit(y: pd.Series, design: pd.DataFrame, maxlags: int) -> dict[str, object]:
    """OLS with Newey-West covariance using fixed Bartlett weights."""
    x = design.to_numpy(float)
    target = y.to_numpy(float)
    inverse = np.linalg.inv(x.T @ x)
    beta = inverse @ x.T @ target
    residual = target - x @ beta
    xu = x * residual[:, None]
    meat = xu.T @ xu
    for lag in range(1, min(maxlags, len(target) - 1) + 1):
        weight = 1 - lag / (maxlags + 1)
        gamma = xu[lag:].T @ xu[:-lag]
        meat += weight * (gamma + gamma.T)
    covariance = inverse @ meat @ inverse
    diagonal = np.diag(covariance)
    se = np.sqrt(np.where(diagonal >= 0, diagonal, np.nan))
    t_values = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    dof = max(len(target) - design.shape[1], 1)
    pvalues = 2 * stats.t.sf(np.abs(t_values), dof)
    return {
        "params": pd.Series(beta, index=design.columns),
        "pvalues": pd.Series(pvalues, index=design.columns),
        "covariance": pd.DataFrame(covariance, index=design.columns, columns=design.columns),
        "df_resid": dof,
    }


def _fit_short_model(
    flows: pd.DataFrame, prior: pd.Series, future: pd.Series,
    config: ShortFlowSymmetryConfig, turnover: pd.Series | None = None,
    covariance: str = "HC3", hac_lags: int = 1,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    values = {"future": future, "prior": prior}
    values.update({flow: flows[flow] for flow in flows})
    if turnover is not None:
        values["short_turnover"] = turnover
    frame = pd.concat(values, axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    counts = {f"{flow}_signal_N": int(frame[flow].eq(1).sum()) for flow in flows}
    base = {"N_total": len(frame), **counts, "condition_number": np.nan,
            "max_vif": np.nan, "max_leverage": np.nan, "status": "not_testable"}
    for flow in FLOW_KEYS:
        base.update({f"{flow}_beta": np.nan, f"{flow}_p": np.nan,
                     f"{flow}_prior_beta": np.nan, f"{flow}_prior_p": np.nan,
                     f"{flow}_main_variance": np.nan, f"{flow}_interaction_variance": np.nan,
                     f"{flow}_main_interaction_covariance": np.nan})
    empty_marginal = pd.DataFrame(columns=["flow", "prior_level", "prior_value", "effect", "ci_low", "ci_high"])
    empty_collinearity = pd.DataFrame(columns=["term", "vif"])
    if len(frame) < 2 * config.min_group_n or any(
        frame[flow].eq(1).sum() < config.min_group_n
        or frame[flow].eq(0).sum() < config.min_group_n for flow in flows
    ):
        return base | {"status": "insufficient_group_sample"}, empty_marginal, empty_collinearity
    design = pd.DataFrame({"const": 1.0, "prior": frame.prior}, index=frame.index)
    for flow in flows:
        design[flow] = frame[flow]
        design[f"{flow}_prior"] = frame[flow] * frame.prior
    ordered = ["const", *flows, "prior", *(f"{flow}_prior" for flow in flows)]
    if turnover is not None:
        design["short_turnover"] = frame.short_turnover
        ordered.append("short_turnover")
    design = design[ordered]
    matrix = design.to_numpy(float)
    condition = float(np.linalg.cond(matrix))
    vifs = _vif(design)
    max_vif = max(vifs.values()) if vifs else np.nan
    collinearity = pd.DataFrame([{"term": term, "vif": value} for term, value in vifs.items()])
    if (np.linalg.matrix_rank(matrix) < design.shape[1] or not np.isfinite(condition)
            or condition >= config.max_condition_number or not np.isfinite(max_vif)
            or max_vif > config.max_vif):
        return base | {"condition_number": condition, "max_vif": max_vif,
                       "status": "unstable_collinearity"}, empty_marginal, collinearity
    if covariance == "HAC":
        fit = _nw_fit(frame.future, design, hac_lags)
        max_leverage = np.nan
    else:
        fit = _hc3_ols(frame.future, design)
        leverage = np.asarray(fit["leverage"], float)
        max_leverage = float(leverage.max())
        if not np.isfinite(leverage).all() or np.any(leverage >= 1 - config.leverage_tolerance):
            return base | {"condition_number": condition, "max_vif": max_vif,
                           "max_leverage": max_leverage,
                           "status": "high_leverage_unstable"}, empty_marginal, collinearity
    params, pvalues, cov = fit["params"], fit["pvalues"], fit["covariance"]
    if not np.isfinite(np.r_[params, pvalues, cov.to_numpy().ravel()]).all():
        return base | {"condition_number": condition, "max_vif": max_vif,
                       "max_leverage": max_leverage, "status": "covariance_nonfinite"}, empty_marginal, collinearity
    result = base | {"prior_beta": float(params.prior), "prior_p": float(pvalues.prior),
                     "condition_number": condition, "max_vif": max_vif,
                     "max_leverage": max_leverage, "status": "estimated",
                     "covariance_method": covariance, "hac_lags": hac_lags if covariance == "HAC" else 0}
    marginal_rows = []
    critical = stats.t.ppf(.975, fit["df_resid"])
    for flow in flows:
        interaction = f"{flow}_prior"
        result.update({
            f"{flow}_beta": float(params[flow]), f"{flow}_p": float(pvalues[flow]),
            f"{flow}_prior_beta": float(params[interaction]),
            f"{flow}_prior_p": float(pvalues[interaction]),
            f"{flow}_main_variance": float(cov.loc[flow, flow]),
            f"{flow}_interaction_variance": float(cov.loc[interaction, interaction]),
            f"{flow}_main_interaction_covariance": float(cov.loc[flow, interaction]),
        })
        for level, value in zip(("p25", "p50", "p75"), frame.prior.quantile([.25, .5, .75])):
            effect = params[flow] + value * params[interaction]
            variance = (cov.loc[flow, flow] + value ** 2 * cov.loc[interaction, interaction]
                        + 2 * value * cov.loc[flow, interaction])
            se = np.sqrt(variance) if variance >= 0 else np.nan
            marginal_rows.append({"flow": flow, "prior_level": level, "prior_value": value,
                                  "effect": effect, "ci_low": effect - critical * se,
                                  "ci_high": effect + critical * se})
    if turnover is not None:
        result.update(short_turnover_beta=float(params.short_turnover),
                      short_turnover_p=float(pvalues.short_turnover))
    return result, pd.DataFrame(marginal_rows), collinearity


def _signal_frame(features: pd.DataFrame, k: int, window: int, variant: str = "raw") -> pd.DataFrame:
    catalog = features.attrs["catalog"]
    columns = {}
    for predictor, meta in catalog.items():
        if meta["analysis_variant"] == variant and meta["k"] == k and meta["window"] == window:
            values = features[predictor]
            columns[meta["family"]] = values.ge(95).astype(float).where(values.notna())
    if set(columns) != set(FLOW_KEYS):
        raise ValueError(f"Missing aligned short signals for k={k}, window={window}: {sorted(columns)}")
    return pd.DataFrame(columns)


def run_short_prior_interaction(
    features: pd.DataFrame, prior: pd.Series, outcomes: pd.DataFrame,
    config: ShortFlowSymmetryConfig | None = None,
) -> dict[str, pd.DataFrame]:
    config = config or ShortFlowSymmetryConfig()
    joint_rows, marginal_rows, nested_rows, collinearity_rows = [], [], [], []
    for k in config.k_values:
        for window in config.rolling_windows:
            signals = _signal_frame(features, k, window)
            for outcome in [name for name in OUTCOMES if name in outcomes]:
                key = {"k": k, "rolling_window": window, "outcome": outcome}
                joint, marginal, col = _fit_short_model(signals, prior, outcomes[outcome], config)
                joint_rows.append(key | joint)
                if not marginal.empty:
                    marginal_rows.append(marginal.assign(**key, model="joint"))
                if not col.empty:
                    collinearity_rows.append(col.assign(**key, model="joint",
                                                        condition_number=joint["condition_number"],
                                                        status=joint["status"]))
                for flow in FLOW_KEYS:
                    nested, _, nested_col = _fit_short_model(
                        signals[[flow]], prior, outcomes[outcome], config
                    )
                    main_before, interaction_before = nested[f"{flow}_beta"], nested[f"{flow}_prior_beta"]
                    for term, before, after, p_before, p_after in (
                        ("main_effect", main_before, joint[f"{flow}_beta"], nested[f"{flow}_p"], joint[f"{flow}_p"]),
                        ("prior_interaction", interaction_before, joint[f"{flow}_prior_beta"],
                         nested[f"{flow}_prior_p"], joint[f"{flow}_prior_p"]),
                    ):
                        attenuation = (1 - abs(after) / abs(before) if np.isfinite(before)
                                       and np.isfinite(after) and abs(before) > 1e-12 else np.nan)
                        if nested["status"] != "estimated" or joint["status"] != "estimated":
                            classification = (
                                "unstable_collinearity"
                                if "unstable_collinearity" in (nested["status"], joint["status"])
                                else "not_testable"
                            )
                        elif np.sign(before) != np.sign(after):
                            classification = "sign_reversal"
                        elif attenuation > .70 or (p_before < .05 <= p_after):
                            classification = "largely_absorbed"
                        elif attenuation >= .30:
                            classification = "partially_absorbed"
                        else:
                            classification = "survives_joint_control"
                        nested_rows.append(key | {
                            "flow": flow, "term": term, "single_model": f"Model_{flow}",
                            "joint_model": "Joint", "coefficient_before": before,
                            "coefficient_after": after, "p_before": p_before, "p_after": p_after,
                            "attenuation_ratio": attenuation,
                            "sign_flip": bool(np.isfinite(before) and np.isfinite(after) and np.sign(before) != np.sign(after)),
                            "classification": classification,
                        })
                    if not nested_col.empty:
                        collinearity_rows.append(nested_col.assign(**key, model=f"single_{flow}",
                                                                 condition_number=nested["condition_number"],
                                                                 status=nested["status"]))
    joint_results = pd.DataFrame(joint_rows)
    interaction_rows, main_rows = [], []
    for row in joint_results.itertuples():
        key = {"k": row.k, "rolling_window": row.rolling_window,
               "outcome": row.outcome, "model_status": row.status}
        for flow in FLOW_KEYS:
            interaction_rows.append(key | {"flow": flow,
                "beta": getattr(row, f"{flow}_prior_beta"),
                "raw_p_value": getattr(row, f"{flow}_prior_p"),
                "main_beta": getattr(row, f"{flow}_beta"),
                "main_variance": getattr(row, f"{flow}_main_variance"),
                "interaction_variance": getattr(row, f"{flow}_interaction_variance"),
                "main_interaction_covariance": getattr(row, f"{flow}_main_interaction_covariance")})
            main_rows.append(key | {"flow": flow, "beta": getattr(row, f"{flow}_beta"),
                                    "raw_p_value": getattr(row, f"{flow}_p")})
    interaction_fdr, interaction_diag = _apply_fdr(
        pd.DataFrame(interaction_rows), "raw_p_value", "flow", INTERACTION_FDR_SCOPE
    )
    main_fdr, main_diag = _apply_fdr(
        pd.DataFrame(main_rows), "raw_p_value", "flow", MAIN_EFFECT_FDR_SCOPE
    )
    return {
        "joint_results": joint_results,
        "interaction_fdr": interaction_fdr, "interaction_fdr_diagnostics": interaction_diag,
        "main_effect_fdr": main_fdr, "main_effect_fdr_diagnostics": main_diag,
        "marginal_effects": pd.concat(marginal_rows, ignore_index=True) if marginal_rows else pd.DataFrame(),
        "nested_comparison": pd.DataFrame(nested_rows),
        "collinearity": pd.concat(collinearity_rows, ignore_index=True) if collinearity_rows else pd.DataFrame(),
    }


def run_short_turnover_robustness(
    features: pd.DataFrame, prior: pd.Series, outcomes: pd.DataFrame,
    primary: pd.DataFrame, turnover_controls: dict[tuple[int, int], pd.Series],
    config: ShortFlowSymmetryConfig | None = None,
) -> pd.DataFrame:
    config = config or ShortFlowSymmetryConfig()
    rows = []
    before_lookup = primary.set_index(["k", "rolling_window", "outcome"])
    for k in config.k_values:
        for window in config.rolling_windows:
            signals = _signal_frame(features, k, window)
            for outcome in [name for name in OUTCOMES if name in outcomes]:
                after, _, _ = _fit_short_model(
                    signals, prior, outcomes[outcome], config,
                    turnover_controls[(k, window)],
                )
                before = before_lookup.loc[(k, window, outcome)]
                for flow in FLOW_KEYS:
                    for term in (flow, f"{flow}_prior"):
                        beta_col, p_col = f"{term}_beta", f"{term}_p"
                        coefficient_before, coefficient_after = before[beta_col], after[beta_col]
                        p_before, p_after = before[p_col], after[p_col]
                        attenuation = (1 - abs(coefficient_after) / abs(coefficient_before)
                                       if np.isfinite(coefficient_before) and np.isfinite(coefficient_after)
                                       and abs(coefficient_before) > 1e-12 else np.nan)
                        if before.status != "estimated" or after["status"] != "estimated":
                            classification = "not_testable"
                        elif np.sign(coefficient_before) != np.sign(coefficient_after):
                            classification = "sign_reversal"
                        elif attenuation > .70 or (p_before < .05 <= p_after):
                            classification = "largely_absorbed"
                        elif attenuation >= .30:
                            classification = "attenuate"
                        else:
                            classification = "survive"
                        rows.append({
                            "flow": flow, "term": "main_effect" if term == flow else "prior_interaction",
                            "k": k, "rolling_window": window, "outcome": outcome,
                            "coefficient_before": coefficient_before, "coefficient_after": coefficient_after,
                            "p_before": p_before, "p_after": p_after,
                            "attenuation_ratio": attenuation,
                            "sign_flip": bool(np.isfinite(coefficient_before) and np.isfinite(coefficient_after)
                                              and np.sign(coefficient_before) != np.sign(coefficient_after)),
                            "classification": classification, "after_status": after["status"],
                        })
    return pd.DataFrame(rows)


def _cluster_summary(features: pd.DataFrame, config: ShortFlowSymmetryConfig) -> pd.DataFrame:
    rows = []
    for k in config.k_values:
        for window in config.rolling_windows:
            signals = _signal_frame(features, k, window)
            for flow in FLOW_KEYS:
                active = np.flatnonzero(signals[flow].eq(1).to_numpy())
                lengths: list[int] = []
                previous: int | None = None
                for position in active:
                    if previous is None or position != previous + 1:
                        lengths.append(1)
                    else:
                        lengths[-1] += 1
                    previous = int(position)
                daily_n, cluster_n = len(active), len(lengths)
                rows.append({
                    "record_type": "cluster_summary", "flow": flow, "k": k,
                    "rolling_window": window, "daily_signal_N": daily_n,
                    "cluster_N": cluster_n, "cluster_daily_ratio": cluster_n / daily_n if daily_n else np.nan,
                    "median_cluster_length": np.median(lengths) if lengths else np.nan,
                    "max_cluster_length": max(lengths) if lengths else np.nan,
                })
    return pd.DataFrame(rows)


def run_short_cluster_validation(
    features: pd.DataFrame, prior: pd.Series, outcomes: pd.DataFrame,
    interaction_fdr: pd.DataFrame, config: ShortFlowSymmetryConfig | None = None,
) -> pd.DataFrame:
    config = config or ShortFlowSymmetryConfig()
    summary = _cluster_summary(features, config)
    candidates = interaction_fdr[interaction_fdr.evidence_level.isin(["Level A", "Level B"])]
    rows = []
    for candidate in candidates.itertuples():
        signals = _signal_frame(features, int(candidate.k), int(candidate.rolling_window))
        horizon = int(str(candidate.outcome).removeprefix("O1_C"))
        fit, _, _ = _fit_short_model(
            signals, prior, outcomes[candidate.outcome], config,
            covariance="HAC", hac_lags=horizon,
        )
        rows.append({
            "record_type": "HAC_inference", "flow": candidate.flow,
            "k": candidate.k, "rolling_window": candidate.rolling_window,
            "outcome": candidate.outcome, "evidence_level": candidate.evidence_level,
            "daily_signal_N": fit.get(f"{candidate.flow}_signal_N"),
            "interaction_beta": fit.get(f"{candidate.flow}_prior_beta"),
            "cluster_robust_p": fit.get(f"{candidate.flow}_prior_p"),
            "cluster_inference_method": f"Newey-West HAC maxlags={horizon}",
            "status": fit["status"],
        })
    return pd.concat([summary, pd.DataFrame(rows)], ignore_index=True, sort=False)


def run_short_crossover_analysis(
    interaction_fdr: pd.DataFrame, prior: pd.Series,
    config: ShortFlowSymmetryConfig | None = None,
) -> pd.DataFrame:
    config = config or ShortFlowSymmetryConfig()
    columns = ["flow", "k", "rolling_window", "outcome", "evidence_level", "main_beta",
               "interaction_beta", "crossover_prior_return", "crossover_ci_low",
               "crossover_ci_high", "observed_prior_p1", "observed_prior_p99",
               "crossover_status", "k_consistency", "short_window_consistency",
               "horizon_consistency"]
    candidates = interaction_fdr[interaction_fdr.evidence_level.isin(["Level A", "Level B"])]
    rows = []
    clean_prior = prior.dropna()
    p1, p99 = clean_prior.quantile(.01), clean_prior.quantile(.99)
    for row in candidates.itertuples():
        denominator = row.beta
        if not np.isfinite(denominator) or abs(denominator) <= config.crossover_denominator_epsilon:
            crossover, low, high, status = np.nan, np.nan, np.nan, "unstable_crossover"
        else:
            crossover = -row.main_beta / denominator
            gradient = np.array([-1 / denominator, row.main_beta / denominator ** 2])
            covariance = np.array([[row.main_variance, row.main_interaction_covariance],
                                   [row.main_interaction_covariance, row.interaction_variance]])
            variance = float(gradient @ covariance @ gradient)
            se = np.sqrt(variance) if variance >= 0 and np.isfinite(variance) else np.nan
            low, high = crossover - 1.96 * se, crossover + 1.96 * se
            if crossover < p1 or crossover > p99:
                status = "extrapolation_crossover"
            elif not np.isfinite(low) or not np.isfinite(high) or low < p1 or high > p99:
                status = "unstable_crossover"
            else:
                status = "stable_crossover"
        rows.append({
            "flow": row.flow, "k": row.k, "rolling_window": row.rolling_window,
            "outcome": row.outcome, "evidence_level": row.evidence_level,
            "main_beta": row.main_beta, "interaction_beta": denominator,
            "crossover_prior_return": crossover, "crossover_ci_low": low,
            "crossover_ci_high": high, "observed_prior_p1": p1, "observed_prior_p99": p99,
            "crossover_status": status,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=columns)
    sign = np.sign(result.crossover_prior_return)
    result["k_consistency"] = result.assign(_sign=sign).groupby(
        ["flow", "rolling_window", "outcome"]
    )["_sign"].transform(lambda x: x.value_counts(normalize=True).max())
    short = result[result.rolling_window.isin([126, 252])].assign(_sign=lambda x: np.sign(x.crossover_prior_return))
    window_consistency = short.groupby(["flow", "k", "outcome"])["_sign"].agg(
        lambda x: x.value_counts(normalize=True).max()
    )
    result["short_window_consistency"] = result.set_index(["flow", "k", "outcome"]).index.map(window_consistency)
    result["horizon_consistency"] = result.assign(_sign=sign).groupby(
        ["flow", "k", "rolling_window"]
    )["_sign"].transform(lambda x: x.value_counts(normalize=True).max())
    return result.reindex(columns=columns)


def short_overlap_diagnostics(
    features: pd.DataFrame, config: ShortFlowSymmetryConfig | None = None
) -> pd.DataFrame:
    config = config or ShortFlowSymmetryConfig()
    raw_ratios = features.attrs.get("raw_ratio_series", {})
    rows = []
    for k in config.k_values:
        for window in config.rolling_windows:
            signals = _signal_frame(features, k, window)
            complete = signals.dropna()
            ss, sc, sr = (complete[name].eq(1) for name in FLOW_KEYS)
            row = {
                "k": k, "rolling_window": window, "complete_N": len(complete),
                "SS_high_N": int(ss.sum()), "SC_high_N": int(sc.sum()), "SR_high_N": int(sr.sum()),
                "SS_SC_both_high_N": int((ss & sc).sum()), "SS_SR_both_high_N": int((ss & sr).sum()),
                "SC_SR_both_high_N": int((sc & sr).sum()), "all_three_high_N": int((ss & sc & sr).sum()),
            }
            for left, right in (("short_sell", "short_cover"), ("short_sell", "short_repayment"),
                                ("short_cover", "short_repayment")):
                left_n, both = complete[left].eq(1).sum(), (complete[left].eq(1) & complete[right].eq(1)).sum()
                row[f"P_{right}_given_{left}"] = both / left_n if left_n else np.nan
                row[f"phi_{left}_{right}"] = complete[left].corr(complete[right]) if len(complete) > 1 else np.nan
                left_ratio = raw_ratios.get(("raw", left, k))
                right_ratio = raw_ratios.get(("raw", right, k))
                if left_ratio is not None and right_ratio is not None:
                    raw = pd.concat({"left": left_ratio, "right": right_ratio}, axis=1).dropna()
                    pearson = raw.left.corr(raw.right, method="pearson") if len(raw) > 1 else np.nan
                    spearman = raw.left.corr(raw.right, method="spearman") if len(raw) > 1 else np.nan
                else:
                    pearson = spearman = np.nan
                row[f"pearson_{left}_{right}"] = pearson
                row[f"spearman_{left}_{right}"] = spearman
            rows.append(row)
    return pd.DataFrame(rows)


def _evidence_status(levels: pd.Series) -> str:
    if levels.isin(["Level A", "Level B"]).any():
        return "Supported"
    if levels.eq("Level C").any():
        return "Partially Supported"
    if levels.eq("Not Testable").all():
        return "Unable to Determine"
    return "Not Supported"


def run_margin_short_symmetry(
    inputs: dict[str, object], short_interaction: pd.DataFrame,
    short_main: pd.DataFrame, turnover: pd.DataFrame,
    clusters: pd.DataFrame, crossover: pd.DataFrame | None = None,
    short_marginal: pd.DataFrame | None = None,
) -> pd.DataFrame:
    margin_interaction = inputs["margin_interaction_fdr"]
    margin_main = inputs["margin_main_fdr"]
    baseline = inputs["baseline_fdr"]
    margin_marginal = inputs.get("margin_marginal", pd.DataFrame())
    margin_clusters = inputs.get("margin_clusters", pd.DataFrame())
    crossover = pd.DataFrame() if crossover is None else crossover
    short_marginal = pd.DataFrame() if short_marginal is None else short_marginal
    pairs = (("Margin Buy", "margin_buy", "Short Sell", "short_sell"),
             ("Margin Sell", "margin_sell", "Short Cover", "short_cover"),
             ("Cash Repayment", "margin_cash_repayment", "Short Repayment", "short_repayment"))
    rows = []
    for margin_label, margin_key, short_label, short_key in pairs:
        if margin_key == "margin_cash_repayment":
            margin_levels = baseline.loc[baseline.family.eq(margin_key), "evidence_level"] if "evidence_level" in baseline else pd.Series(dtype="object")
            margin_evidence = _evidence_status(margin_levels) if len(margin_levels) else "Unable to Determine"
            margin_interaction_sign = np.nan
            symmetry = "Unable to Determine"
        else:
            margin_levels = margin_main.loc[margin_main.hypothesis.eq(margin_key), "evidence_level"]
            margin_evidence = _evidence_status(margin_levels)
            hypothesis = "buy_x_prior" if margin_key == "margin_buy" else "sell_x_prior"
            mi = margin_interaction[margin_interaction.hypothesis.eq(hypothesis)]
            margin_interaction_sign = np.sign(mi.loc[mi.evidence_level.isin(["Level A", "Level B"]), "beta"]).median()
            si = short_interaction[short_interaction.flow.eq(short_key)]
            short_sign = np.sign(si.loc[si.evidence_level.isin(["Level A", "Level B"]), "beta"]).median()
            if pd.isna(margin_interaction_sign) or pd.isna(short_sign):
                symmetry = "Unable to Determine"
            elif margin_interaction_sign == -short_sign:
                symmetry = "Supported"
            elif margin_interaction_sign == short_sign:
                symmetry = "Not Supported"
            else:
                symmetry = "Partially Supported"
        short_levels = short_main.loc[short_main.flow.eq(short_key), "evidence_level"]
        short_evidence = _evidence_status(short_levels)
        si = short_interaction[short_interaction.flow.eq(short_key)]
        short_interaction_sign = np.sign(si.loc[si.evidence_level.isin(["Level A", "Level B"]), "beta"]).median()
        margin_hypothesis = "buy_x_prior" if margin_key == "margin_buy" else "sell_x_prior"
        mi = (margin_interaction[margin_interaction.hypothesis.eq(margin_hypothesis)]
              if margin_key != "margin_cash_repayment" else pd.DataFrame())
        mi_evidence = mi[mi.evidence_level.isin(["Level A", "Level B"])] if len(mi) else mi
        si_evidence = si[si.evidence_level.isin(["Level A", "Level B"])]
        margin_family = margin_key
        margin_cluster_ratio = np.nan
        if len(margin_clusters) and {"family", "daily_signal_N", "independent_cluster_N"}.issubset(margin_clusters):
            mc = margin_clusters[margin_clusters.family.eq(margin_family)].drop_duplicates(
                [column for column in ("family", "k", "window") if column in margin_clusters]
            )
            margin_cluster_ratio = (
                mc.independent_cluster_N.sum() / mc.daily_signal_N.sum()
                if len(mc) and mc.daily_signal_N.sum() else np.nan
            )
        short_cluster = clusters[
            clusters.record_type.eq("cluster_summary") & clusters.flow.eq(short_key)
        ]
        cross = crossover[crossover.flow.eq(short_key)] if len(crossover) else crossover
        short_marginal_direction = "unable"
        if len(short_marginal) and {"flow", "prior_level", "effect"}.issubset(short_marginal):
            sm = short_marginal[short_marginal.flow.eq(short_key)]
            low, high = sm[sm.prior_level.eq("p25")].effect.median(), sm[sm.prior_level.eq("p75")].effect.median()
            if pd.notna(low) and pd.notna(high):
                short_marginal_direction = "increases_with_prior" if high > low else "decreases_with_prior" if high < low else "flat"
        margin_marginal_direction = "unable"
        prefix = "buy" if margin_key == "margin_buy" else "sell"
        low_col, high_col = f"{prefix}_effect_p25", f"{prefix}_effect_p75"
        if margin_key != "margin_cash_repayment" and len(margin_marginal) and {low_col, high_col}.issubset(margin_marginal):
            low, high = margin_marginal[low_col].median(), margin_marginal[high_col].median()
            if pd.notna(low) and pd.notna(high):
                margin_marginal_direction = "increases_with_prior" if high > low else "decreases_with_prior" if high < low else "flat"
        rows.append({
            "Pair": f"{margin_label} ↔ {short_label}", "Margin evidence": margin_evidence,
            "Short evidence": short_evidence,
            "Margin interaction sign": margin_interaction_sign,
            "Short interaction sign": short_interaction_sign,
            "Margin interaction magnitude median": mi_evidence.beta.abs().median() if len(mi_evidence) else np.nan,
            "Short interaction magnitude median": si_evidence.beta.abs().median() if len(si_evidence) else np.nan,
            "Margin effective horizons": ",".join(sorted(mi_evidence.outcome.astype(str).unique())) if len(mi_evidence) else "none",
            "Short effective horizons": ",".join(sorted(si_evidence.outcome.astype(str).unique())) if len(si_evidence) else "none",
            "Margin k pattern": ",".join(map(str, sorted(mi_evidence.k.unique()))) if len(mi_evidence) else "none",
            "Short k pattern": ",".join(map(str, sorted(si_evidence.k.unique()))) if len(si_evidence) else "none",
            "Margin rolling-window pattern": ",".join(map(str, sorted(mi_evidence.rolling_window.unique()))) if len(mi_evidence) else "none",
            "Short rolling-window pattern": ",".join(map(str, sorted(si_evidence.rolling_window.unique()))) if len(si_evidence) else "none",
            "Same/opposite interaction direction": (
                "opposite" if pd.notna(margin_interaction_sign) and pd.notna(short_interaction_sign)
                and margin_interaction_sign == -short_interaction_sign else
                "same" if pd.notna(margin_interaction_sign) and margin_interaction_sign == short_interaction_sign
                else "unable"
            ),
            "Short turnover robustness": _evidence_status(
                pd.Series(["Level C" if (turnover.flow.eq(short_key) & turnover.classification.eq("survive")).any() else "No Evidence"])
            ),
            "Short clustering severity": clusters.loc[
                (clusters.record_type.eq("cluster_summary") & clusters.flow.eq(short_key)),
                "cluster_daily_ratio",
            ].median(),
            "Margin clustering severity": margin_cluster_ratio,
            "Margin crossover location": "not_available_in_formal_margin_joint_study",
            "Short crossover location median": cross.crossover_prior_return.median() if len(cross) else np.nan,
            "Margin marginal-effect direction": margin_marginal_direction,
            "Short marginal-effect direction": short_marginal_direction,
            "Symmetry status": symmetry,
        })
    return pd.DataFrame(rows)


def annual_short_robustness(
    candidates: pd.DataFrame, features: pd.DataFrame, prior: pd.Series,
    outcomes: pd.DataFrame, config: ShortFlowSymmetryConfig | None = None,
) -> pd.DataFrame:
    config = config or ShortFlowSymmetryConfig()
    columns = ["flow", "k", "rolling_window", "outcome", "year", "signal_N", "cluster_N",
               "interaction_beta", "direction", "status", "years_available",
               "same_direction_years", "direction_consistency", "largest_year_sample_share"]
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for candidate in candidates.itertuples():
        signals = _signal_frame(features, int(candidate.k), int(candidate.rolling_window))
        for year in sorted(set(outcomes.index.year)):
            mask = outcomes.index.year == year
            year_index = outcomes.index[mask]
            fit, _, _ = _fit_short_model(
                signals.reindex(year_index), prior.reindex(year_index),
                outcomes[candidate.outcome].reindex(year_index), config
            )
            active = signals[candidate.flow].reindex(year_index).eq(1)
            positions = np.flatnonzero(active.to_numpy())
            cluster_n = int(sum(i == 0 or positions[i] != positions[i - 1] + 1 for i in range(len(positions))))
            beta = fit[f"{candidate.flow}_prior_beta"]
            rows.append({
                "flow": candidate.flow, "k": candidate.k, "rolling_window": candidate.rolling_window,
                "outcome": candidate.outcome, "year": year, "signal_N": int(active.sum()),
                "cluster_N": cluster_n, "interaction_beta": beta,
                "direction": "positive" if beta > 0 else "negative" if beta < 0 else "missing",
                "status": fit["status"] if fit["status"] == "estimated" else "Unable to Determine",
            })
    result = pd.DataFrame(rows)
    keys = ["flow", "k", "rolling_window", "outcome"]
    for _, group in result.groupby(keys):
        usable = group[group.status.eq("estimated")]
        total = group.signal_N.sum()
        if len(usable):
            dominant = np.sign(usable.interaction_beta).value_counts().index[0]
            same = int(np.sign(usable.interaction_beta).eq(dominant).sum())
            consistency = same / len(usable)
        else:
            same, consistency = 0, np.nan
        result.loc[group.index, "years_available"] = len(usable)
        result.loc[group.index, "same_direction_years"] = same
        result.loc[group.index, "direction_consistency"] = consistency
        result.loc[group.index, "largest_year_sample_share"] = group.signal_N.max() / total if total else np.nan
    return result.reindex(columns=columns)


def _create_run_directory(config: ShortFlowSymmetryConfig) -> Path:
    stamp = datetime.now(ZoneInfo(config.timezone)).strftime("%Y%m%d_%H%M%S")
    short = _git_value(["rev-parse", "--short", "HEAD"])
    path = config.output_root / f"{stamp}_{short}_short_flow_prior_return_symmetry"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_run_info(run_dir: Path, baseline: Path, margin: Path,
                    config: ShortFlowSymmetryConfig) -> None:
    try:
        finlab_version = importlib.metadata.version("finlab")
    except importlib.metadata.PackageNotFoundError:
        finlab_version = "unknown"
    definitions = short_flow_definitions()
    values = {
        "repository": config.repository, "branch": _git_value(["branch", "--show-current"]),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "baseline_run_dir": baseline, "baseline_commit": config.baseline_commit,
        "margin_joint_run_dir": margin, "margin_joint_commit": config.margin_joint_commit,
        "price_source_open": "etl:adj_open", "price_source_close": "etl:adj_close",
        "outcome_price_adjusted": True,
        "short_sell_definition": definitions["short_sell"],
        "short_cover_definition": definitions["short_cover"],
        "short_repayment_definition": definitions["short_repayment"],
        "short_turnover_definition": definitions["short_turnover"],
        "short_native_amount_ratio_available": False,
        "primary_short_normalization": PRIMARY_VARIANT,
        "prior_return_definition": "adjusted_close[t] / adjusted_close[t-5] - 1",
        "k": config.k_values, "rolling_windows": config.rolling_windows,
        "outcome_horizons": config.outcome_horizons,
        "baseline_fdr_scope": BASELINE_FDR_SCOPE,
        "interaction_fdr_scope": INTERACTION_FDR_SCOPE,
        "main_effect_fdr_scope": MAIN_EFFECT_FDR_SCOPE,
        "cluster_inference_method": "Newey-West HAC; maxlags=outcome horizon",
        "min_group_n": config.min_group_n, "max_vif": config.max_vif,
        "timezone": config.timezone,
        "run_timestamp": datetime.now(ZoneInfo(config.timezone)).isoformat(),
        "python_version": platform.python_version(), "finlab_version": finlab_version,
    }
    (run_dir / "run_info.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in values.items()) + "\n", encoding="utf-8"
    )


def _summary_markdown(result: dict[str, object]) -> str:
    baseline, interaction, main = result["primary_fdr"], result["interaction_results"], result["main_effect_fdr"]
    clusters, crossover, symmetry = result["cluster_robustness"], result["crossover"], result["margin_short_symmetry"]
    def status(frame: pd.DataFrame, flow: str) -> str:
        dimension = "flow" if "flow" in frame.columns else "family"
        return _evidence_status(frame.loc[frame[dimension].eq(flow), "evidence_level"])
    hypotheses = {f"H{i + 1}": status(interaction, flow) for i, flow in enumerate(FLOW_KEYS)}
    hypotheses["H4"] = (
        "Supported" if all(status(main, flow) == "Supported" for flow in FLOW_KEYS)
        else "Unable to Determine" if any(status(main, flow) == "Unable to Determine" for flow in FLOW_KEYS)
        else "Partially Supported" if any(status(main, flow) in ("Supported", "Partially Supported") for flow in FLOW_KEYS)
        else "Not Supported"
    )
    table = "| Hypothesis | Status |\n|---|---|\n" + "\n".join(
        f"| {key} | {value} |" for key, value in hypotheses.items()
    )
    estimable = int(interaction.raw_p_value.notna().sum())
    effective = interaction[interaction.raw_p_value.notna()].rolling_window.value_counts().to_dict()
    hac = clusters[clusters.record_type.eq("HAC_inference")]
    stable_crossovers = int(crossover.crossover_status.eq("stable_crossover").sum()) if len(crossover) else 0
    overall = "保留" if all(value == "Supported" for value in hypotheses.values()) else "修改後再測" if estimable else "淘汰"
    sections = {
        "A. 目前假設": "三種 short flows 可能具有不同 conditional structure；不預設方向或因果。\n\n" + table,
        "B. 市場機制": "Short Sell、Cover、Repayment 分開；反對解釋是它們都只是價格變動後的同步 Short Activity。",
        "C. Data provenance": "Frozen adjusted baseline 與 margin joint commits 均須 PASS。Short 無 native amount ratio，正式沿用既有 volume ratio。",
        "D. Short Sell baseline": status(baseline, "short_sell"),
        "E. Short Cover baseline": status(baseline, "short_cover"),
        "F. Short Repayment baseline": status(baseline, "short_repayment"),
        "G. Short Sell × Prior Return": status(interaction, "short_sell"),
        "H. Short Cover × Prior Return": status(interaction, "short_cover"),
        "I. Short Repayment × Prior Return": status(interaction, "short_repayment"),
        "J. Joint model / nested comparison": "Single-flow models與joint model使用相同定義；attenuation與sign flip見CSV。",
        "K. Turnover robustness": "只加入 Short Sell+Cover volume-ratio turnover，不擴大primary FDR。",
        "L. Event clustering": "各flow/k/window輸出daily N、cluster N與cluster length。",
        "M. Cluster-aware inference": f"Level A/B候選使用Newey-West HAC，maxlags=outcome horizon；估計列數={len(hac)}。",
        "N. Marginal effects": "固定prior-return P25/P50/P75與95% CI，不搜尋threshold。",
        "O. Crossover stability": f"只由Level A/B係數推導；stable crossover數={stable_crossovers}，P1/P99外標為extrapolation。",
        "P. Collinearity": f"VIF>{result['config'].max_vif}或condition number過高即unstable_collinearity。",
        "Q. Annual robustness": "只分析Level A/B；年度樣本不足寫Unable to Determine。",
        "R. Margin vs Short symmetry": f"三組狀態={symmetry['Symmetry status'].value_counts().to_dict()}。",
        "S. 主要風險": "Look-ahead、survivorship、selection/data snooping、multiple testing、overlapping outcomes、serial dependence、tail sample、rolling dependence、collinearity、prior-return endogeneity、turnover confounding、年度集中、制度改變與缺乏out-of-sample。",
        "T. 反對者觀點": "只有joint、turnover與HAC後仍保留，才提高conditional predictive information可信度。",
        "U. 保留、修改後再測或淘汰": overall,
    }
    questions = [
        f"1. Short Sell固定main effect？{status(main, 'short_sell')}",
        f"2. Short Cover固定main effect？{status(main, 'short_cover')}",
        f"3. Short Repayment固定main effect？{status(main, 'short_repayment')}",
        f"4. Short Sell×Prior？{status(interaction, 'short_sell')}",
        f"5. Short Cover×Prior？{status(interaction, 'short_cover')}",
        f"6. Short Repayment×Prior？{status(interaction, 'short_repayment')}",
        "7. 最強horizon依FDR results比較，不以單一raw p選模。",
        f"8. W126/W252一致性見FDR與crossover；effective estimable counts={effective}。",
        "9. W504/W756若Not Testable即Unable to Determine，不降低門檻。",
        "10. 共線性見pairwise、VIF與condition number。",
        "11. Joint control後是否survive見nested comparison。",
        "12. Short turnover後是否survive見turnover robustness。",
        "13. 是否只是少數連續events見cluster_daily_ratio。",
        "14. Cluster-aware後是否成立見HAC p-value。",
        f"15. 穩定crossover數={stable_crossovers}。",
        "16. Crossover超出P1/P99不可作實務threshold。",
        f"17–19. Margin/Short三組對稱性見margin_short_symmetry.csv；Cash Repayment interaction未正式研究時為Unable to Determine。",
        "20. Flow×Prior或固定方向須依兩個獨立FDR universe比較。",
        "21–22. 不強迫鏡像；若joint/turnover/HAC未保留，反對解釋仍較合理。",
    ]
    lines = ["# Short Flow × Prior Return Conditional Structure", ""]
    for heading, text in sections.items():
        lines += [f"## {heading}", "", text, ""]
    lines += ["## 最終必答問題", "", *questions, ""]
    return "\n".join(lines)


def run_short_flow_prior_return_symmetry_study(
    baseline_run_dir: str | Path,
    margin_joint_run_dir: str | Path,
    config: ShortFlowSymmetryConfig | None = None,
    provider=None,
    export: bool = True,
    stage_zero: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run the six staged short-flow analyses without rerunning earlier research."""
    config = config or ShortFlowSymmetryConfig()
    inputs = validate_short_flow_inputs(baseline_run_dir, margin_joint_run_dir, config)
    stage = run_stage_zero(provider=provider) if stage_zero is None else stage_zero
    datasets = stage["datasets"]
    features, feature_catalog, turnover_controls = build_short_flow_features(datasets, config)
    prices = pd.concat({"open": datasets["open"][config.target_symbol],
                        "close": datasets["close"][config.target_symbol]}, axis=1).dropna().sort_index()
    feature_attrs = features.attrs.copy()
    features = features.reindex(prices.index)
    features.attrs = feature_attrs
    prior = prior_5d_return(prices.close)
    outcomes = build_outcomes(prices.open, prices.close, config.outcome_horizons)
    price_diagnostics = validate_split_window(prices.open, prices.close, outcomes)
    primary, primary_fdr, primary_diag = run_short_flow_baseline(features, outcomes, config)
    interaction = run_short_prior_interaction(features, prior, outcomes, config)
    turnover = run_short_turnover_robustness(
        features, prior, outcomes, interaction["joint_results"], turnover_controls, config
    )
    clusters = run_short_cluster_validation(
        features, prior, outcomes, interaction["interaction_fdr"], config
    )
    crossover = run_short_crossover_analysis(interaction["interaction_fdr"], prior, config)
    overlap = short_overlap_diagnostics(features, config)
    symmetry = run_margin_short_symmetry(
        inputs, interaction["interaction_fdr"], interaction["main_effect_fdr"], turnover,
        clusters, crossover, interaction["marginal_effects"]
    )
    candidates = interaction["interaction_fdr"][
        interaction["interaction_fdr"].evidence_level.isin(["Level A", "Level B"])
    ]
    annual = annual_short_robustness(candidates, features, prior, outcomes, config)
    fdr_diagnostics = pd.concat([
        primary_diag,
        interaction["interaction_fdr_diagnostics"],
        interaction["main_effect_fdr_diagnostics"],
    ], ignore_index=True)
    result: dict[str, object] = {
        "config": config, "input_validation": inputs["diagnostics"],
        "stage_zero": stage, "feature_catalog": feature_catalog,
        "features": features, "prior_5d_return": prior, "outcomes": outcomes,
        "outcome_price_diagnostics": price_diagnostics,
        "primary_results": primary, "primary_fdr": primary_fdr,
        "fdr_diagnostics": fdr_diagnostics,
        "interaction_joint_models": interaction["joint_results"],
        "interaction_results": interaction["interaction_fdr"],
        "main_effect_fdr": interaction["main_effect_fdr"],
        "marginal_effects": interaction["marginal_effects"],
        "nested_comparison": interaction["nested_comparison"],
        "turnover_robustness": turnover, "cluster_robustness": clusters,
        "crossover": crossover, "overlap_diagnostics": overlap,
        "collinearity": interaction["collinearity"],
        "margin_short_symmetry": symmetry, "annual_robustness": annual,
    }
    if export:
        run_dir = _create_run_directory(config)
        result["run_dir"] = run_dir
        outputs = {
            "short_flow_primary_results.csv": primary,
            "short_flow_fdr_results.csv": primary_fdr,
            "short_flow_fdr_diagnostics.csv": fdr_diagnostics,
            "short_flow_joint_model_results.csv": interaction["joint_results"],
            "short_flow_prior_interaction_results.csv": interaction["interaction_fdr"],
            "short_flow_main_effect_fdr_results.csv": interaction["main_effect_fdr"],
            "short_flow_marginal_effects.csv": interaction["marginal_effects"],
            "short_flow_nested_comparison.csv": interaction["nested_comparison"],
            "short_flow_turnover_robustness.csv": turnover,
            "short_flow_cluster_robustness.csv": clusters,
            "short_flow_crossover.csv": crossover,
            "short_flow_overlap_diagnostics.csv": overlap,
            "short_flow_collinearity.csv": interaction["collinearity"],
            "margin_short_symmetry.csv": symmetry,
            "short_flow_annual_robustness.csv": annual,
            "input_validation.csv": inputs["diagnostics"],
        }
        for filename, frame in outputs.items():
            frame.to_csv(run_dir / filename, index=False)
        (run_dir / "short_flow_summary.md").write_text(_summary_markdown(result), encoding="utf-8")
        _write_run_info(run_dir, Path(baseline_run_dir), Path(margin_joint_run_dir), config)
    return result
