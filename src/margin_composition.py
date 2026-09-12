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

from .fdr import apply_fdr, fdr_diagnostics
from .features import pr_group, trailing_percentile
from .margin_turnover import (
    BASELINE_COMMIT,
    _baseline_original_features,
    _read_run_info,
    _sum_columns,
    _turnover_base_series,
    load_frozen_baseline,
    load_turnover_data,
    margin_turnover_variant_series,
)
from .outcomes import build_outcomes
from .price_validation import validate_split_window
from .statistics import (
    _fit_base_ols,
    annual_results,
    annual_robustness_summary,
    neighborhood_consistency,
    run_controlled_tests,
    run_pr_bin_descriptive,
    run_primary_tests,
)
from .universe import build_universe_diagnostics


TURNOVER_BASE_COMMIT = "e613b5e7176c998bd1e8abe191f0e7a4caf2abd3"
FDR_SCOPE = "margin_composition_incremental_study"
REQUIRED_TURNOVER_FILES = (
    "run_info_turnover.txt",
    "turnover_primary_results.csv",
    "turnover_fdr_results.csv",
    "turnover_annual_results.csv",
    "turnover_annual_robustness_summary.csv",
    "turnover_absorption_results.csv",
)


@dataclass(frozen=True)
class MarginCompositionConfig:
    repository: str = "taiwan-margin-short-0050-research"
    timezone: str = "Asia/Taipei"
    target_symbol: str = "0050"
    baseline_commit: str = BASELINE_COMMIT
    turnover_base_commit: str = TURNOVER_BASE_COMMIT
    k_values: tuple[int, ...] = (1, 3, 5, 10)
    rolling_windows: tuple[int, ...] = (126, 252, 504, 756)
    outcome_horizons: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    pr_edges: tuple[int, ...] = (0, 5, 20, 40, 60, 80, 95, 100)
    min_group_n: int = 20
    controlled_leverage_tolerance: float = 1e-10
    controlled_max_condition_number: float = 1e12
    robustness_min_years: int = 3
    robustness_max_year_sample_share: float = 0.50
    equivalence_tolerance: float = 1e-10
    absorption_partial_threshold: float = 0.30
    absorption_large_threshold: float = 0.70
    absorption_survival_retention: float = 0.70
    absorption_significance_threshold: float = 0.05
    absorption_beta_epsilon: float = 1e-12
    fixed_turnover_k: int = 10
    fixed_turnover_window: int = 504
    fixed_turnover_tail: float = 5.0
    crash_concentration_threshold: float = 0.60
    output_root: Path = Path("outputs_composition")


def _git_value(args: list[str], default: str = "unknown") -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return default


def load_frozen_turnover(
    turnover_run_dir: str | Path,
    expected_commit: str = TURNOVER_BASE_COMMIT,
    expected_baseline: str = BASELINE_COMMIT,
) -> dict[str, object]:
    root = Path(turnover_run_dir).expanduser()
    missing = [name for name in REQUIRED_TURNOVER_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Turnover output is missing required files: {missing}")
    info = _read_run_info(root / "run_info_turnover.txt")
    if info.get("git_commit") != expected_commit:
        raise ValueError(
            f"Turnover commit mismatch: expected {expected_commit}, "
            f"got {info.get('git_commit')!r}"
        )
    if info.get("baseline_commit") != expected_baseline:
        raise ValueError(
            f"Turnover baseline mismatch: expected {expected_baseline}, "
            f"got {info.get('baseline_commit')!r}"
        )
    result: dict[str, object] = {"run_dir": root, "run_info": info}
    for filename in REQUIRED_TURNOVER_FILES[1:]:
        result[filename.removesuffix(".csv")] = pd.read_csv(root / filename)
    return result


def margin_composition_kday(
    buy_amount: pd.Series, sell_amount: pd.Series, k: int
) -> pd.DataFrame:
    """Compute composition from sums first; invalid denominators remain missing."""
    buy = buy_amount.rolling(k, min_periods=k).sum()
    sell = sell_amount.rolling(k, min_periods=k).sum()
    denominator = buy + sell
    valid = denominator.gt(0) & denominator.notna()
    buy_share = (buy / denominator).where(valid)
    imbalance = ((buy - sell) / denominator).where(valid)
    return pd.DataFrame(
        {"buy_amount_k": buy, "sell_amount_k": sell, "denominator": denominator,
         "buy_share": buy_share, "imbalance": imbalance}
    )


def _composition_amounts(datasets: dict[str, pd.DataFrame]) -> tuple[pd.Series, pd.Series]:
    columns = ("上市融資交易金額", "上櫃融資交易金額")
    return (
        _sum_columns(datasets["aggregate_buy"], columns),
        _sum_columns(datasets["aggregate_sell"], columns),
    )


def build_margin_composition_features(
    datasets: dict[str, pd.DataFrame],
    config: MarginCompositionConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return primary buy-share percentiles, catalog, and equivalence diagnostics."""
    config = config or MarginCompositionConfig()
    buy_amount, sell_amount = _composition_amounts(datasets)
    values: dict[str, pd.Series] = {}
    catalog: dict[str, dict] = {}
    validations: list[dict] = []
    for k in config.k_values:
        raw = margin_composition_kday(buy_amount, sell_amount, k)
        for window in config.rolling_windows:
            buy_pr = trailing_percentile(raw["buy_share"], window)
            imbalance_pr = trailing_percentile(raw["imbalance"], window)
            name = f"margin_composition__buy_share__k{k}__w{window}_percentile"
            values[name] = buy_pr
            catalog[name] = {
                "family": "margin_composition", "variant": "buy_share", "k": k,
                "window": window, "primary": True,
                "invalid_denominator_rows": int(raw["denominator"].le(0).fillna(False).sum()),
                "valid_raw_rows": int(raw["buy_share"].notna().sum()),
                "valid_percentile_rows": int(buy_pr.notna().sum()),
                "missing_raw_rows": int(raw["buy_share"].isna().sum()),
                "first_valid_date": raw["buy_share"].first_valid_index(),
                "last_valid_date": raw["buy_share"].last_valid_index(),
            }
            validations.append(_equivalence_row(raw, buy_pr, imbalance_pr, k, window, config))
    features = pd.DataFrame(values).sort_index()
    features.attrs["catalog"] = catalog
    validation = pd.DataFrame(validations)
    failed = validation.loc[~validation["formula_valid"]]
    if not failed.empty:
        raise ValueError(
            "buy_share/imbalance equivalence validation failed; stop before inference: "
            f"{failed[['k', 'rolling_window']].to_dict('records')}"
        )
    feature_catalog = pd.DataFrame(
        [{"predictor": name, **meta} for name, meta in catalog.items()]
    )
    return features, feature_catalog, validation


def _equivalence_row(
    raw: pd.DataFrame,
    buy_pr: pd.Series,
    imbalance_pr: pd.Series,
    k: int,
    window: int,
    config: MarginCompositionConfig,
) -> dict:
    raw_pair = pd.concat({"buy": raw["buy_share"], "imb": raw["imbalance"]}, axis=1).dropna()
    percentile_pair = pd.concat({"buy_pr": buy_pr, "imb_pr": imbalance_pr}, axis=1).dropna()
    formula_diff = (raw_pair["imb"] - (2.0 * raw_pair["buy"] - 1.0)).abs()
    percentile_diff = (percentile_pair["buy_pr"] - percentile_pair["imb_pr"]).abs()
    bins_equal = bool(
        (pd.Series(pr_group(percentile_pair["buy_pr"]), index=percentile_pair.index).astype("string") ==
         pd.Series(pr_group(percentile_pair["imb_pr"]), index=percentile_pair.index).astype("string")).all()
    )
    pearson = raw_pair["buy"].corr(raw_pair["imb"], method="pearson") if len(raw_pair) > 1 else np.nan
    spearman = raw_pair["buy"].corr(raw_pair["imb"], method="spearman") if len(raw_pair) > 1 else np.nan
    max_formula = formula_diff.max() if len(raw_pair) else np.nan
    max_percentile = percentile_diff.max() if len(percentile_pair) else np.nan
    enough = len(raw_pair) > 1 and len(percentile_pair) > 1
    valid = bool(
        enough and max_formula <= config.equivalence_tolerance
        and abs(pearson - 1.0) <= config.equivalence_tolerance
        and abs(spearman - 1.0) <= config.equivalence_tolerance
        and max_percentile <= config.equivalence_tolerance and bins_equal
    )
    return {
        "k": k, "rolling_window": window, "N_raw_complete": len(raw_pair),
        "N_percentile_complete": len(percentile_pair),
        "max_abs_formula_diff": max_formula, "pearson": pearson, "spearman": spearman,
        "max_abs_percentile_rank_diff": max_percentile,
        "percentile_ordering_equal": bool(max_percentile <= config.equivalence_tolerance) if enough else False,
        "pr_bin_equivalence": bins_equal, "tolerance": config.equivalence_tolerance,
        "formula_valid": valid,
    }


def _attach_variant(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    out["variant"] = "buy_share"
    return out


def run_composition_primary(
    features: pd.DataFrame, outcomes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = _attach_variant(run_primary_tests(features, outcomes))
    if len(primary) > 192:
        raise ValueError(f"Composition inferential universe exceeds 192 cells: {len(primary)}")
    fdr = apply_fdr(primary) if not primary.empty else primary.copy()
    if "raw_p_value" in fdr:
        fdr.loc[fdr["raw_p_value"].isna(), "evidence_level"] = "Not Testable"
    fdr["fdr_scope"] = FDR_SCOPE
    diagnostics = fdr_diagnostics(fdr)
    if not fdr.empty:
        diagnostics = pd.concat([diagnostics, pd.DataFrame([{
            "metric": "not_testable_cells", "family": "ALL",
            "count": int(fdr["evidence_level"].eq("Not Testable").sum()),
        }])], ignore_index=True)
    diagnostics["fdr_scope"] = FDR_SCOPE
    return primary, fdr, diagnostics


def composition_neighborhood_consistency(results: pd.DataFrame) -> pd.DataFrame:
    out = neighborhood_consistency(results)
    if not out.empty:
        out.insert(1, "variant", "buy_share")
    return out


def classify_composition_dose_response(pr_bins: pd.DataFrame) -> pd.DataFrame:
    """Classify the seven pre-specified bins descriptively, never inferentially."""
    if pr_bins.empty:
        return pr_bins.copy()
    order = ["PR0-5", "PR5-20", "PR20-40", "PR40-60", "PR60-80", "PR80-95", "PR95-100"]
    keys = ["predictor", "family", "variant", "k", "rolling_window", "outcome_horizon"]
    labels = []
    for key, group in pr_bins.groupby(keys, dropna=False):
        means = group.set_index("pr_group").reindex(order)["mean_return"]
        if means.isna().any():
            shape = "insufficient_data"
        else:
            delta = np.diff(means.to_numpy())
            minimum, maximum = int(np.argmin(means)), int(np.argmax(means))
            if np.all(delta >= 0):
                shape = "monotonic_bullish"
            elif np.all(delta <= 0):
                shape = "monotonic_bearish"
            elif 0 < minimum < 6 and np.all(delta[:minimum] <= 0) and np.all(delta[minimum:] >= 0):
                shape = "U_shape"
            elif 0 < maximum < 6 and np.all(delta[:maximum] >= 0) and np.all(delta[maximum:] <= 0):
                shape = "inverted_U"
            elif maximum == 6 or minimum == 6:
                shape = "high_tail_extreme"
            elif maximum == 0 or minimum == 0:
                shape = "low_tail_extreme"
            else:
                shape = "mixed"
        labels.append(dict(zip(keys, key)) | {"shape": shape})
    return pr_bins.merge(pd.DataFrame(labels), on=keys, how="left")


def _composition_group(percentile: pd.Series) -> pd.Series:
    values = pd.to_numeric(percentile, errors="coerce")
    group = pd.Series(pd.NA, index=percentile.index, dtype="string")
    group.loc[values.ge(0) & values.lt(20)] = "sell_dominant_PR0-20"
    group.loc[values.ge(20) & values.lt(80)] = "neutral_PR20-80"
    group.loc[values.ge(80) & values.le(100)] = "buy_dominant_PR80-100"
    return group


def _sample_stats(sample: pd.Series, unconditional: float) -> dict:
    clean = sample.dropna()
    return {
        "N": len(clean), "mean_return": clean.mean(), "median_return": clean.median(),
        "win_rate": clean.gt(0).mean() if len(clean) else np.nan,
        "std": clean.std(ddof=1) if len(clean) > 1 else np.nan,
        "q05": clean.quantile(.05) if len(clean) else np.nan,
        "q25": clean.quantile(.25) if len(clean) else np.nan,
        "q75": clean.quantile(.75) if len(clean) else np.nan,
        "q95": clean.quantile(.95) if len(clean) else np.nan,
        "min_return": clean.min() if len(clean) else np.nan,
        "max_return": clean.max() if len(clean) else np.nan,
        "effect_vs_unconditional": clean.mean() - unconditional if len(clean) else np.nan,
        "downside_below_minus_5pct": clean.lt(-.05).mean() if len(clean) else np.nan,
        "downside_below_minus_10pct": clean.lt(-.10).mean() if len(clean) else np.nan,
    }


def run_low_turnover_composition(
    turnover_percentile: pd.Series,
    composition_percentile: pd.Series,
    outcomes: pd.DataFrame,
    config: MarginCompositionConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = config or MarginCompositionConfig()
    joined = pd.concat(
        {"turnover_pr": turnover_percentile, "composition_pr": composition_percentile}, axis=1
    ).join(outcomes, how="inner")
    joined["composition_group"] = _composition_group(joined["composition_pr"])
    low = joined["turnover_pr"].le(config.fixed_turnover_tail) & joined["turnover_pr"].notna()
    horizons = [f"O1_C{h}" for h in (5, 10, 20)]
    pooled_rows, annual_rows = [], []
    for outcome in horizons:
        valid = joined[outcome].where(joined["turnover_pr"].notna() & joined["composition_pr"].notna())
        unconditional = valid.mean()
        for label in ("sell_dominant_PR0-20", "neutral_PR20-80", "buy_dominant_PR80-100"):
            selected = low & joined["composition_group"].eq(label)
            pooled_rows.append({"composition_group": label, "outcome_horizon": outcome,
                                **_sample_stats(joined.loc[selected, outcome], unconditional)})
            for year in sorted(set(joined.index.year)):
                year_mask = joined.index.year == year
                year_valid = valid[year_mask]
                annual_rows.append({
                    "composition_group": label, "outcome_horizon": outcome, "year": year,
                    **_sample_stats(joined.loc[selected & year_mask, outcome], year_valid.mean()),
                })
    pooled = pd.DataFrame(pooled_rows)
    annual = pd.DataFrame(annual_rows)
    annual_summary = _low_turnover_annual_summary(annual, config)
    contribution = _low_turnover_year_contribution(joined, low, config)
    clusters = _low_turnover_clusters(joined, low)
    return pooled, annual, annual_summary, contribution, clusters


def _low_turnover_annual_summary(
    annual: pd.DataFrame, config: MarginCompositionConfig
) -> pd.DataFrame:
    rows = []
    keys = ["composition_group", "outcome_horizon"]
    for key, group in annual.groupby(keys, dropna=False):
        sampled = group[group["N"] > 0]
        total = sampled["N"].sum()
        rows.append(dict(zip(keys, key)) | {
            "years_with_samples": len(sampled),
            "positive_effect_years": int(sampled["effect_vs_unconditional"].gt(0).sum()),
            "negative_effect_years": int(sampled["effect_vs_unconditional"].lt(0).sum()),
            "positive_effect_year_ratio": sampled["effect_vs_unconditional"].gt(0).mean() if len(sampled) else np.nan,
            "direction_consistency": max(
                sampled["effect_vs_unconditional"].gt(0).mean(),
                sampled["effect_vs_unconditional"].lt(0).mean(),
            ) if len(sampled) else np.nan,
            "median_annual_effect": sampled["effect_vs_unconditional"].median(),
            "min_annual_effect": sampled["effect_vs_unconditional"].min(),
            "max_annual_effect": sampled["effect_vs_unconditional"].max(),
            "median_annual_return": sampled["mean_return"].median(),
            "largest_year_sample_share": sampled["N"].max() / total if total else np.nan,
            "few_year_concentration_flag": bool(
                len(sampled) < config.robustness_min_years or
                (total and sampled["N"].max() / total > config.robustness_max_year_sample_share)
            ),
        })
    return pd.DataFrame(rows)


def _low_turnover_year_contribution(
    joined: pd.DataFrame, low: pd.Series, config: MarginCompositionConfig
) -> pd.DataFrame:
    sample = joined.loc[low, "O1_C20"].dropna()
    rows = []
    total_n = len(sample)
    negative_sums = sample.groupby(sample.index.year).sum().clip(upper=0).abs()
    negative_total = negative_sums.sum()
    for year, values in sample.groupby(sample.index.year):
        year_sum = values.sum()
        rows.append({
            "year": year, "N_year": len(values), "sum_return_year": year_sum,
            "mean_return_year": values.mean(),
            "weighted_contribution_to_total_mean": year_sum / total_n if total_n else np.nan,
            "share_of_total_negative_contribution": negative_sums.get(year, 0.0) / negative_total if negative_total else 0.0,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        top3 = result["share_of_total_negative_contribution"].nlargest(3).sum()
        result["top3_negative_contribution_share"] = top3
        result["crash_regime_concentration"] = bool(top3 >= config.crash_concentration_threshold)
    return result


def _low_turnover_clusters(joined: pd.DataFrame, low: pd.Series) -> pd.DataFrame:
    signal = low.fillna(False).to_numpy()
    cluster_id = np.cumsum(signal & ~np.r_[False, signal[:-1]])
    raw_days = int(signal.sum())
    cluster_count = int(cluster_id.max()) if len(cluster_id) else 0
    rows = []
    for cid in range(1, cluster_count + 1):
        positions = np.flatnonzero(signal & (cluster_id == cid))
        first = positions[0]
        block = joined.iloc[positions]
        row = {
            "cluster_id": cid, "start_date": block.index[0], "end_date": block.index[-1],
            "trading_days": len(block), "first_signal_date": joined.index[first],
            "composition_median": block["composition_pr"].median(),
            "composition_group": _composition_group(
                pd.Series([block["composition_pr"].median()])
            ).iloc[0],
            "year": joined.index[first].year,
            "number_of_raw_signal_days": raw_days,
            "number_of_independent_clusters": cluster_count,
        }
        for outcome in ("O1_C5", "O1_C10", "O1_C20"):
            row[f"first_signal_{outcome}"] = joined.iloc[first][outcome]
        rows.append(row)
    return pd.DataFrame(rows)


def _fit_nested_models(
    signal: pd.Series,
    turnover: pd.Series,
    buy_share: pd.Series,
    future: pd.Series,
    prior: pd.Series,
    config: MarginCompositionConfig,
) -> dict:
    """Fit Model B and C on exactly the same complete-case observations."""
    frame = pd.concat({
        "future": future, "signal": signal, "turnover": turnover,
        "buy_share": buy_share, "prior": prior,
    }, axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    base = {
        "N": len(frame), "signal_n": int(frame["signal"].eq(1).sum()),
        "control_n": int(frame["signal"].eq(0).sum()),
        "beta_model_b": np.nan, "p_model_b": np.nan,
        "beta_model_c": np.nan, "p_model_c": np.nan,
        "turnover_beta_model_c": np.nan, "turnover_p_model_c": np.nan,
        "buy_share_beta_model_c": np.nan, "buy_share_p_model_c": np.nan,
        "condition_number_b": np.nan, "condition_number_c": np.nan,
        "rank_b": np.nan, "rank_c": np.nan,
        "max_leverage_b": np.nan, "max_leverage_c": np.nan,
        "correlation_signal_turnover": np.nan,
        "correlation_signal_buy_share": np.nan,
        "correlation_turnover_buy_share": np.nan,
        "hc3_finite_b": False, "hc3_finite_c": False,
    }
    if len(frame) < config.min_group_n:
        return base | {"status": "insufficient_sample"}
    if frame["signal"].nunique() < 2 or base["signal_n"] < config.min_group_n or base["control_n"] < config.min_group_n:
        return base | {"status": "insufficient_group_sample"}
    if min(frame[column].nunique() for column in ("turnover", "buy_share", "prior")) < 2:
        return base | {"status": "no_control_variation"}
    base["correlation_signal_turnover"] = frame["signal"].corr(frame["turnover"])
    base["correlation_signal_buy_share"] = frame["signal"].corr(frame["buy_share"])
    base["correlation_turnover_buy_share"] = frame["turnover"].corr(frame["buy_share"])
    designs = {
        "b": pd.DataFrame({
            "const": 1.0, "signal": frame["signal"].astype(float),
            "turnover": frame["turnover"].astype(float), "prior": frame["prior"].astype(float),
        }, index=frame.index),
        "c": pd.DataFrame({
            "const": 1.0, "signal": frame["signal"].astype(float),
            "turnover": frame["turnover"].astype(float),
            "buy_share": frame["buy_share"].astype(float), "prior": frame["prior"].astype(float),
        }, index=frame.index),
    }
    robust = {}
    for label, design in designs.items():
        matrix = design.to_numpy()
        rank = int(np.linalg.matrix_rank(matrix))
        condition = float(np.linalg.cond(matrix))
        base[f"rank_{label}"] = rank
        base[f"condition_number_{label}"] = condition
        if rank < design.shape[1] or not np.isfinite(condition) or condition >= config.controlled_max_condition_number:
            return base | {"status": "unstable_collinearity"}
        fit = _fit_base_ols(frame["future"], design)
        leverage = np.asarray(fit.get_influence().hat_matrix_diag, dtype=float)
        max_leverage = float(np.max(leverage)) if len(leverage) and np.isfinite(leverage).all() else np.nan
        base[f"max_leverage_{label}"] = max_leverage
        if not np.isfinite(leverage).all() or np.any(leverage >= 1.0 - config.controlled_leverage_tolerance):
            return base | {"status": "unstable_collinearity"}
        robust[label] = fit.get_robustcov_results(cov_type="HC3")
        finite = np.isfinite(np.asarray(robust[label].params)).all() and np.isfinite(
            np.asarray(robust[label].pvalues)
        ).all() and np.isfinite(np.asarray(robust[label].cov_params())).all()
        base[f"hc3_finite_{label}"] = bool(finite)
        if not finite:
            return base | {"status": "unstable_collinearity"}
    names_b = list(designs["b"].columns)
    names_c = list(designs["c"].columns)
    params_b = dict(zip(names_b, np.asarray(robust["b"].params)))
    pvalues_b = dict(zip(names_b, np.asarray(robust["b"].pvalues)))
    params_c = dict(zip(names_c, np.asarray(robust["c"].params)))
    pvalues_c = dict(zip(names_c, np.asarray(robust["c"].pvalues)))
    return base | {
        "beta_model_b": params_b["signal"], "p_model_b": pvalues_b["signal"],
        "beta_model_c": params_c["signal"], "p_model_c": pvalues_c["signal"],
        "turnover_beta_model_c": params_c["turnover"],
        "turnover_p_model_c": pvalues_c["turnover"],
        "buy_share_beta_model_c": params_c["buy_share"],
        "buy_share_p_model_c": pvalues_c["buy_share"],
        "status": "estimated",
    }


def _classify_absorption(fit: dict, config: MarginCompositionConfig) -> tuple[float, str]:
    if fit.get("status") in {"insufficient_sample", "insufficient_group_sample", "no_control_variation"}:
        return np.nan, "insufficient_sample"
    if fit.get("status") != "estimated":
        return np.nan, "unstable_collinearity"
    before, after = fit["beta_model_b"], fit["beta_model_c"]
    if abs(before) <= config.absorption_beta_epsilon:
        return np.nan, "undefined"
    attenuation = 1.0 - abs(after) / abs(before)
    if np.sign(before) != np.sign(after):
        return attenuation, "sign_reversal"
    retained = abs(after) / abs(before)
    if retained >= config.absorption_survival_retention and fit["p_model_c"] < config.absorption_significance_threshold:
        return attenuation, "survives_composition_control"
    if config.absorption_partial_threshold <= attenuation <= config.absorption_large_threshold:
        return attenuation, "partially_absorbed"
    if attenuation > config.absorption_large_threshold or fit["p_model_c"] >= config.absorption_significance_threshold:
        return attenuation, "largely_absorbed"
    return attenuation, "survives_composition_control"


def select_composition_absorption_candidates(turnover_absorption: pd.DataFrame) -> pd.DataFrame:
    required = {"status", "original_family", "original_predictor", "turnover_variant"}
    if not required.issubset(turnover_absorption.columns):
        raise ValueError(f"Turnover absorption lacks fields: {sorted(required - set(turnover_absorption.columns))}")
    mask = (
        turnover_absorption["status"].eq("estimated")
        & turnover_absorption["original_family"].isin(["margin_buy", "margin_sell"])
        & turnover_absorption["original_predictor"].astype(str).str.contains("__amount_ratio__", regex=False)
        & turnover_absorption["turnover_variant"].eq("amount_ratio")
    )
    return turnover_absorption.loc[mask].copy()


def run_composition_absorption(
    turnover_absorption: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    turnover_features: pd.DataFrame,
    composition_features: pd.DataFrame,
    outcomes: pd.DataFrame,
    close_0050: pd.Series,
    config: MarginCompositionConfig | None = None,
) -> pd.DataFrame:
    config = config or MarginCompositionConfig()
    candidates = select_composition_absorption_candidates(turnover_absorption)
    if candidates.empty:
        return pd.DataFrame()
    candidate_specs = candidates.rename(columns={
        "original_predictor": "predictor", "original_family": "family",
    })
    originals = _baseline_original_features(datasets, candidate_specs)
    joined = originals.join(turnover_features, how="outer").join(composition_features, how="outer").join(outcomes, how="inner")
    priors = {k: close_0050.pct_change(k, fill_method=None).reindex(joined.index) for k in config.k_values}
    rows = []
    dedupe = ["original_predictor", "original_family", "original_pr_group", "k", "rolling_window", "outcome_horizon", "prior_k"]
    for row in candidates.drop_duplicates([c for c in dedupe if c in candidates]).itertuples():
        predictor = row.original_predictor
        percentile = joined[predictor]
        signal = (percentile.le(5) if row.original_pr_group == "PR0-5" else percentile.ge(95)).astype(float).where(percentile.notna())
        k, window = int(row.k), int(row.rolling_window)
        turnover_name = f"margin_turnover__amount_ratio__k{k}__w{window}_percentile"
        composition_name = f"margin_composition__buy_share__k{k}__w{window}_percentile"
        prior_values = [int(row.prior_k)] if hasattr(row, "prior_k") and pd.notna(row.prior_k) else list(config.k_values)
        for prior_k in prior_values:
            fit = _fit_nested_models(
                signal, joined[turnover_name] / 100.0, joined[composition_name] / 100.0,
                joined[row.outcome_horizon], priors[prior_k], config,
            )
            attenuation, classification = _classify_absorption(fit, config)
            rows.append({
                "original_predictor": predictor, "original_family": row.original_family,
                "original_pr_group": row.original_pr_group, "k": k,
                "rolling_window": window, "outcome_horizon": row.outcome_horizon,
                "prior_k": prior_k, "turnover_variant": "amount_ratio",
                "composition_variant": "buy_share", **fit,
                "horizon": row.outcome_horizon,
                "beta_original_model_B": fit.get("beta_model_b", np.nan),
                "p_original_model_B": fit.get("p_model_b", np.nan),
                "beta_original_model_C": fit.get("beta_model_c", np.nan),
                "p_original_model_C": fit.get("p_model_c", np.nan),
                "original_beta_change": fit.get("beta_model_c", np.nan) - fit.get("beta_model_b", np.nan),
                "original_attenuation_ratio": attenuation,
                "correlation_original_turnover": fit.get("correlation_signal_turnover", np.nan),
                "correlation_original_buy_share": fit.get("correlation_signal_buy_share", np.nan),
                "condition_number": fit.get("condition_number_c", np.nan),
                "classification": classification,
                "beta_change_b_to_c": fit.get("beta_model_c", np.nan) - fit.get("beta_model_b", np.nan),
                "attenuation_ratio_b_to_c": attenuation,
                "absorption_classification": classification,
            })
    return pd.DataFrame(rows)


def _create_run_directory(config: MarginCompositionConfig) -> Path:
    short = _git_value(["rev-parse", "--short", "HEAD"])
    stamp = datetime.now(ZoneInfo(config.timezone)).strftime("%Y%m%d_%H%M%S")
    path = config.output_root / f"{stamp}_{short}_margin_composition_adjusted"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_run_info(
    run_dir: Path, baseline_run_dir: Path, turnover_run_dir: Path,
    config: MarginCompositionConfig,
) -> None:
    try:
        finlab_version = importlib.metadata.version("finlab")
    except importlib.metadata.PackageNotFoundError:
        finlab_version = "unknown"
    fields = {
        "repository": config.repository, "branch": _git_value(["branch", "--show-current"]),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "turnover_base_commit": config.turnover_base_commit,
        "baseline_commit": config.baseline_commit,
        "baseline_output_dir": str(baseline_run_dir), "turnover_output_dir": str(turnover_run_dir),
        "run_timestamp": datetime.now(ZoneInfo(config.timezone)).isoformat(),
        "timezone": config.timezone, "python_version": platform.python_version(),
        "finlab_version": finlab_version, "study": "margin_composition_incremental",
        "price_source_open": "etl:adj_open", "price_source_close": "etl:adj_close",
        "outcome_price_adjusted": True, "corporate_action_fix": "0050_split_2025_06",
        "previous_composition_commit": "4067347b10cbf98e89fcd4bcd29799c1132fe171",
        "previous_composition_output": "/content/drive/MyDrive/Quant_Research/taiwan-margin-short-0050-research/margin_composition/20260912_164123_4067347_margin_composition_incremental",
        "primary_variable": "buy_share", "derived_variable": "imbalance",
        "k_values": config.k_values, "rolling_windows": config.rolling_windows,
        "pr_bins": config.pr_edges, "outcome_horizons": config.outcome_horizons,
        "fdr_scope": FDR_SCOPE, "inferential_cells_max": 192,
        "fdr_universe_note": "incremental FDR only; not comparable to original full-global FDR universe",
        "baseline_full_research_rerun": False, "turnover_full_research_rerun": False,
        "low_turnover_primary_definition": "amount_ratio,k10,w504,PR0-5",
        "low_turnover_composition_bins": "PR0-20 / PR20-80 / PR80-100",
    }
    (run_dir / "run_info_composition.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in fields.items()) + "\n", encoding="utf-8"
    )


def _summary_markdown(result: dict[str, object]) -> str:
    fdr = result["fdr"]
    low = result["low_turnover_composition"]
    absorption = result["composition_absorption"]
    level_a = int(fdr["evidence_level"].eq("Level A").sum()) if len(fdr) else 0
    not_testable = int(fdr["evidence_level"].eq("Not Testable").sum()) if len(fdr) else 0
    sections = {
        "A. 目前假設": "Activity 與 Composition 可能共同描述融資市場狀態。",
        "B. 市場機制": "Turnover 描述信用交易活躍度；Composition 只描述 Buy/Sell transaction composition。",
        "C. Composition definition": "`buy_share = BuyAmount_k / (BuyAmount_k + SellAmount_k)`；先加總 k 日再取比率。",
        "D. Buy Share / Imbalance equivalence validation": "`imbalance = 2 * buy_share - 1` 僅作等價驗證，不形成第二個 FDR family。",
        "E. Composition primary FDR results": f"Level A cells: {level_a}; Not Testable cells: {not_testable}。Scope: `{FDR_SCOPE}`。",
        "F. Horizon：C1 / C2-C3 / C5-C10-C20": "各 horizon 使用原定 O1 進場 outcome；不得把 overlapping horizons 當獨立事件。",
        "G. Dose-response / PR shape": "七個固定 bins 僅作描述性 shape 分類，不增加 FDR tests。",
        "H. Annual robustness": "至少三年、方向一致率與單一年集中度須通過既有門檻，否則無法判定。",
        "I. Neighborhood robustness": "只檢查 parameter plateau，不取代 FDR。",
        "J. Low Turnover 年度異質性": "固定 amount_ratio k10/W504/PR0-5，未搜尋其他組合。",
        "K. Low Turnover × Composition": f"Pooled descriptive rows: {len(low)}；固定 PR0-20、PR20-80、PR80-100。",
        "L. Year contribution": "以 C20 年度加權 contribution 與負報酬 contribution 檢查集中度。",
        "M. Event cluster concentration": "連續訊號交易日合併為同一 cluster，首日 outcome 代表獨立事件診斷。",
        "N. Turnover + Composition controlled model": f"Model B→C rows: {len(absorption)}；使用相同 complete cases。",
        "O. Margin Buy residual absorption": "只使用 frozen turnover absorption 中 estimated primary amount-ratio candidates。",
        "P. Margin Sell residual absorption": "數值不穩定一律標為 `unstable_collinearity`，不作經濟吸收解讀。",
        "Q. 主要風險": "年度集中、tail 樣本不足、共線性、overlapping outcomes 與 incremental data snooping。",
        "R. 反對者觀點": "Composition 可能只是市場方向或成交狀態的結果，並非可交易因果訊號。",
        "S. 最終結論": "Margin Composition：無法判定。Low Turnover hypothesis：D. 無法判定。",
    }
    lines = ["# Margin Composition Incremental Study", "", "本研究沒有重跑或合併 baseline 與 turnover FDR。", ""]
    for title, body in sections.items():
        lines.extend([f"## {title}", "", body, ""])
    return "\n".join(lines)


def run_margin_composition_study(
    baseline_run_dir: str | Path,
    turnover_run_dir: str | Path,
    config: MarginCompositionConfig | None = None,
    provider=None,
    export: bool = True,
    datasets: dict[str, pd.DataFrame] | None = None,
) -> dict[str, object]:
    """Run only the composition increment; never call the full research pipeline."""
    config = config or MarginCompositionConfig()
    baseline = load_frozen_baseline(baseline_run_dir, config.baseline_commit)
    turnover = load_frozen_turnover(turnover_run_dir, config.turnover_base_commit, config.baseline_commit)
    data = load_turnover_data(provider) if datasets is None else datasets
    universe, primary_symbols, limitation = build_universe_diagnostics(data["margin_balance"], data["market_value"])
    features, catalog, equivalence = build_margin_composition_features(data, config)
    outcomes = build_outcomes(data["open"][config.target_symbol], data["close"][config.target_symbol], config.outcome_horizons)
    price_diagnostics = validate_split_window(
        data["open"][config.target_symbol], data["close"][config.target_symbol], outcomes
    )
    primary, fdr, fdr_diag = run_composition_primary(features, outcomes)
    pr_bins = classify_composition_dose_response(
        _attach_variant(run_pr_bin_descriptive(features, outcomes))
    )
    controlled = _attach_variant(run_controlled_tests(
        features, outcomes, data["close"][config.target_symbol],
        min_total_n=config.min_group_n, min_tail_n=config.min_group_n,
        min_control_n=config.min_group_n,
        leverage_tolerance=config.controlled_leverage_tolerance,
        max_condition_number=config.controlled_max_condition_number,
    ))
    annual = _attach_variant(annual_results(fdr, features, outcomes))
    annual_summary = _attach_variant(annual_robustness_summary(
        annual, min_years=config.robustness_min_years,
        max_year_sample_share=config.robustness_max_year_sample_share,
    ))
    neighborhood = composition_neighborhood_consistency(fdr)
    turnover_base = _turnover_base_series(data, primary_symbols)
    turnover_raw = margin_turnover_variant_series(turnover_base, config.fixed_turnover_k)["amount_ratio"]
    turnover_pr = trailing_percentile(turnover_raw, config.fixed_turnover_window)
    composition_name = (
        f"margin_composition__buy_share__k{config.fixed_turnover_k}__"
        f"w{config.fixed_turnover_window}_percentile"
    )
    low, low_annual, low_summary, contribution, clusters = run_low_turnover_composition(
        turnover_pr, features[composition_name], outcomes, config
    )
    turnover_features = pd.DataFrame()
    for k in config.k_values:
        for window in config.rolling_windows:
            name = f"margin_turnover__amount_ratio__k{k}__w{window}_percentile"
            turnover_features[name] = trailing_percentile(
                margin_turnover_variant_series(turnover_base, k)["amount_ratio"], window
            )
    absorption = run_composition_absorption(
        turnover["turnover_absorption_results"], data, turnover_features, features,
        outcomes, data["close"][config.target_symbol], config,
    )
    result: dict[str, object] = {
        "baseline": baseline, "turnover": turnover, "universe": universe,
        "UNIVERSE_LIMITATION": limitation, "features": features,
        "feature_catalog": catalog, "equivalence_validation": equivalence,
        "outcomes": outcomes, "primary": primary, "fdr": fdr,
        "outcome_price_diagnostics": price_diagnostics,
        "fdr_diagnostics": fdr_diag, "pr_bins": pr_bins, "controlled": controlled,
        "annual": annual, "annual_robustness_summary": annual_summary,
        "neighborhood": neighborhood, "low_turnover_composition": low,
        "low_turnover_annual": low_annual, "low_turnover_annual_summary": low_summary,
        "low_turnover_year_contribution": contribution,
        "low_turnover_event_clusters": clusters, "composition_absorption": absorption,
    }
    if export:
        run_dir = _create_run_directory(config)
        result["run_dir"] = run_dir
        outputs = {
            "composition_feature_catalog.csv": catalog,
            "composition_equivalence_validation.csv": equivalence,
            "composition_primary_results.csv": primary,
            "composition_fdr_results.csv": fdr,
            "composition_fdr_diagnostics.csv": fdr_diag,
            "composition_pr_bin_results.csv": pr_bins,
            "composition_controlled_results.csv": controlled,
            "composition_annual_results.csv": annual,
            "composition_annual_robustness_summary.csv": annual_summary,
            "composition_neighborhood_consistency.csv": neighborhood,
            "low_turnover_composition_results.csv": low,
            "low_turnover_composition_annual_results.csv": low_annual,
            "low_turnover_composition_annual_summary.csv": low_summary,
            "low_turnover_year_contribution.csv": contribution,
            "low_turnover_event_clusters.csv": clusters,
            "composition_absorption_results.csv": absorption,
            "outcome_price_diagnostics.csv": price_diagnostics,
        }
        for filename, frame in outputs.items():
            frame.to_csv(run_dir / filename, index=False)
        (run_dir / "composition_summary.md").write_text(_summary_markdown(result), encoding="utf-8")
        _write_run_info(run_dir, Path(baseline_run_dir), Path(turnover_run_dir), config)
    return result
