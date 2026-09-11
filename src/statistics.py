from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .features import pr_group


def compare_group(values: pd.Series, selected: pd.Series) -> dict:
    a = values[selected.fillna(False)].dropna()
    b = values[(~selected.fillna(False))].dropna()
    if len(a) < 2 or len(b) < 2:
        return {k: np.nan for k in ("mean_return", "median_return", "win_rate", "std", "q25", "q75", "effect_vs_zero", "effect_vs_unconditional", "raw_p_value", "zero_p_value") } | {"N": len(a)}
    return {
        "N": len(a), "mean_return": a.mean(), "median_return": a.median(), "win_rate": (a > 0).mean(),
        "std": a.std(ddof=1), "q25": a.quantile(.25), "q75": a.quantile(.75),
        "effect_vs_zero": a.mean(), "effect_vs_unconditional": a.mean() - values.dropna().mean(),
        "raw_p_value": stats.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue,
        "zero_p_value": stats.ttest_1samp(a, 0, nan_policy="omit").pvalue,
    }


def run_primary_tests(features: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    joined = features.join(outcomes, how="inner")
    outcome_cols = [c for c in outcomes if c.startswith("O1_C")]
    for feature in [c for c in features if c.endswith("_percentile")]:
        meta = features.attrs.get("catalog", {}).get(feature, {})
        for label, selected in {"PR0-5": joined[feature] <= 5, "PR95-100": joined[feature] >= 95}.items():
            for outcome in outcome_cols:
                result = compare_group(joined[outcome].where(joined[feature].notna()), selected)
                rows.append({"predictor": feature, "family": meta.get("family", feature.split("__")[0]), "k": meta.get("k"), "rolling_window": meta.get("window"), "pr_group": label, "outcome_horizon": outcome, **result})
    return pd.DataFrame(rows)


def run_pr_bin_descriptive(features: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Describe all seven fixed PR bins without adding tests to the FDR universe."""
    rows = []
    joined = features.join(outcomes, how="inner")
    outcome_cols = [c for c in outcomes if c.startswith("O1_C")]
    for feature in [c for c in features if c.endswith("_percentile")]:
        meta = features.attrs.get("catalog", {}).get(feature, {})
        groups = pd.Series(pr_group(joined[feature]), index=joined.index)
        valid_predictor = joined[feature].notna()
        for label in groups.cat.categories:
            selected = groups.eq(label)
            for outcome in outcome_cols:
                values = joined[outcome].where(valid_predictor)
                sample = values[selected].dropna()
                unconditional = values.dropna().mean()
                rows.append({
                    "predictor": feature,
                    "family": meta.get("family", feature.split("__")[0]),
                    "k": meta.get("k"),
                    "rolling_window": meta.get("window"),
                    "pr_group": str(label),
                    "outcome_horizon": outcome,
                    "N": len(sample),
                    "mean_return": sample.mean(),
                    "median_return": sample.median(),
                    "win_rate": (sample > 0).mean() if len(sample) else np.nan,
                    "std": sample.std(ddof=1) if len(sample) > 1 else np.nan,
                    "q25": sample.quantile(.25) if len(sample) else np.nan,
                    "q75": sample.quantile(.75) if len(sample) else np.nan,
                    "effect_vs_unconditional": sample.mean() - unconditional if len(sample) else np.nan,
                })
    return pd.DataFrame(rows)


def controlled_regression(signal: pd.Series, future: pd.Series, prior_return: pd.Series) -> dict:
    frame = pd.concat({"future": future, "signal": signal, "prior": prior_return}, axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 20:
        return {"N": len(frame), "signal_beta": np.nan, "signal_p_value": np.nan, "status": "insufficient_sample"}
    design = np.column_stack([np.ones(len(frame)), frame[["signal", "prior"]].to_numpy()])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return {"N": len(frame), "signal_beta": np.nan, "signal_p_value": np.nan, "status": "rank_deficient"}
    import statsmodels.api as sm
    fit = sm.OLS(frame["future"], pd.DataFrame(design, index=frame.index, columns=["const", "signal", "prior"])).fit(cov_type="HC3")
    return {"N": len(frame), "signal_beta": fit.params["signal"], "signal_p_value": fit.pvalues["signal"], "status": "estimated"}


def run_controlled_tests(features: pd.DataFrame, outcomes: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    rows = []
    joined = features.join(outcomes, how="inner")
    priors = {k: close.pct_change(k, fill_method=None).reindex(joined.index) for k in (1, 3, 5, 10)}
    for feature in [c for c in features if c.endswith("_percentile")]:
        meta = features.attrs.get("catalog", {}).get(feature, {})
        tails = {
            "PR0-5": joined[feature].le(5).astype(float).where(joined[feature].notna()),
            "PR95-100": joined[feature].ge(95).astype(float).where(joined[feature].notna()),
        }
        for label, signal in tails.items():
            for prior_k in (1, 3, 5, 10):
                prior = priors[prior_k]
                for outcome in [c for c in outcomes if c.startswith("O1_C")]:
                    fit = controlled_regression(signal, joined[outcome], prior)
                    rows.append({
                        "predictor": feature, "family": meta.get("family"), "pr_group": label,
                        "k": meta.get("k"), "rolling_window": meta.get("window"),
                        "prior_k": prior_k, "outcome_horizon": outcome, **fit,
                    })
    return pd.DataFrame(rows)


def annual_results(results: pd.DataFrame, features: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    candidates = results[results["evidence_level"].isin(["Level A", "Level B"])] if len(results) else results
    joined = features.join(outcomes, how="inner")
    rows = []
    for r in candidates.itertuples():
        selected = joined[r.predictor].le(5) if r.pr_group == "PR0-5" else joined[r.predictor].ge(95)
        values = joined[r.outcome_horizon]
        for year, idx in joined.groupby(joined.index.year).groups.items():
            sample = values.loc[idx][selected.loc[idx]].dropna()
            baseline = values.loc[idx].where(joined.loc[idx, r.predictor].notna()).dropna()
            rows.append({
                "predictor": r.predictor, "family": r.family, "k": r.k,
                "rolling_window": r.rolling_window, "pr_group": r.pr_group,
                "outcome_horizon": r.outcome_horizon, "year": year, "N": len(sample),
                "mean_return": sample.mean(), "median_return": sample.median(),
                "win_rate": (sample > 0).mean() if len(sample) else np.nan,
                "effect_vs_unconditional": sample.mean() - baseline.mean() if len(sample) and len(baseline) else np.nan,
            })
    return pd.DataFrame(rows)


def annual_robustness_summary(
    annual: pd.DataFrame,
    min_years: int = 3,
    max_year_sample_share: float = 0.5,
) -> pd.DataFrame:
    columns = [
        "predictor", "family", "k", "rolling_window", "pr_group", "outcome_horizon",
        "years_with_samples", "positive_effect_years", "negative_effect_years",
        "positive_year_ratio", "median_annual_return", "min_annual_return",
        "max_annual_return", "largest_year_sample_share", "few_year_concentration_flag",
    ]
    if annual.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    keys = ["predictor", "family", "k", "rolling_window", "pr_group", "outcome_horizon"]
    for key, group in annual.groupby(keys, dropna=False):
        sampled = group[group["N"] > 0]
        total_n = sampled["N"].sum()
        effect = sampled["effect_vs_unconditional"] if "effect_vs_unconditional" in sampled else sampled["mean_return"]
        positive = int(effect.gt(0).sum())
        negative = int(effect.lt(0).sum())
        years = len(sampled)
        largest_share = sampled["N"].max() / total_n if total_n else np.nan
        rows.append(dict(zip(keys, key)) | {
            "years_with_samples": years,
            "positive_effect_years": positive,
            "negative_effect_years": negative,
            "positive_year_ratio": positive / years if years else np.nan,
            "median_annual_return": sampled["mean_return"].median(),
            "min_annual_return": sampled["mean_return"].min(),
            "max_annual_return": sampled["mean_return"].max(),
            "largest_year_sample_share": largest_share,
            "few_year_concentration_flag": bool(years < min_years or (pd.notna(largest_share) and largest_share > max_year_sample_share)),
        })
    return pd.DataFrame(rows, columns=columns)


def regime_results(results: pd.DataFrame, features: pd.DataFrame, outcomes: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    candidates = results.nsmallest(min(30, len(results)), "raw_p_value") if len(results) else results
    joined = features.join(outcomes, how="inner")
    trend = close / close.rolling(252, min_periods=252).mean() - 1
    vol = close.pct_change(fill_method=None).rolling(60, min_periods=60).std()
    regime = pd.Series("sideways", index=close.index)
    regime[trend > 0.05] = "bull"; regime[trend < -0.05] = "bear"
    regime = regime.where(trend.notna())
    vol_regime = pd.Series(np.where(vol >= vol.rolling(252, min_periods=126).median(), "high_vol", "low_vol"), index=close.index)
    vol_regime = vol_regime.where(vol.notna() & vol.rolling(252, min_periods=126).median().notna())
    rows = []
    for r in candidates.itertuples():
        selected = joined[r.predictor].le(5) if r.pr_group == "PR0-5" else joined[r.predictor].ge(95)
        for kind, labels in {"trend": regime, "volatility": vol_regime}.items():
            for label in labels.dropna().unique():
                mask = selected & labels.reindex(joined.index).eq(label)
                sample = joined.loc[mask, r.outcome_horizon].dropna()
                rows.append({"predictor": r.predictor, "pr_group": r.pr_group, "outcome_horizon": r.outcome_horizon, "regime_type": kind, "regime": label, "N": len(sample), "mean_return": sample.mean(), "win_rate": (sample > 0).mean() if len(sample) else np.nan})
    return pd.DataFrame(rows)


def neighborhood_consistency(results: pd.DataFrame) -> pd.DataFrame:
    columns = ["family", "pr_group", "consistency_type", "fixed_parameter", "effect_sign", "n_cells", "positive_cells", "negative_cells", "consistency_score", "note"]
    if results.empty:
        return pd.DataFrame(columns=columns)
    out = []

    def append_group(keys, group, consistency_type, fixed_parameter):
        signs = np.sign(group["effect_vs_unconditional"].dropna())
        if signs.empty:
            score = np.nan; dominant = 0
        else:
            dominant = 1 if signs.mean() >= 0 else -1
            score = float((signs == dominant).mean())
        family, pr = keys[:2]
        out.append({
            "family": family, "pr_group": pr, "consistency_type": consistency_type,
            "fixed_parameter": fixed_parameter, "effect_sign": dominant, "n_cells": len(signs),
            "positive_cells": int(signs.gt(0).sum()), "negative_cells": int(signs.lt(0).sum()),
            "consistency_score": score, "note": "Robustness only; does not replace FDR.",
        })

    for keys, group in results.groupby(["family", "pr_group"], dropna=False):
        append_group(keys, group, "overall", "all")
    specifications = [
        ("fixed_horizon", "outcome_horizon"),
        ("fixed_k", "k"),
        ("fixed_window", "rolling_window"),
    ]
    for consistency_type, fixed_column in specifications:
        for keys, group in results.groupby(["family", "pr_group", fixed_column], dropna=False):
            append_group(keys, group, consistency_type, keys[2])
    adjacent = {"O1_C5", "O1_C10", "O1_C20"}
    for keys, group in results.groupby(["family", "pr_group", "k", "rolling_window"], dropna=False):
        use = group[group["outcome_horizon"].isin(adjacent)]
        if use["outcome_horizon"].nunique() >= 2:
            append_group(keys, use, "adjacent_horizon", f"k={keys[2]};window={keys[3]}")
    return pd.DataFrame(out, columns=columns)


def short_cover_variant_diagnostics(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sign_columns = ["raw_sign", "normalized_sign", "adjusted_raw_sign", "adjusted_normalized_sign"]
    if results.empty or "predictor" not in results:
        return pd.DataFrame(), pd.DataFrame(columns=[
            "k", "rolling_window", "pr_group", "outcome_horizon", *sign_columns,
            "all_same_direction", "conflict_flag",
        ])
    detail = results[
        results["predictor"].str.contains("short_cover", na=False)
    ].copy()
    if detail.empty:
        return detail, pd.DataFrame(columns=[
            "k", "rolling_window", "pr_group", "outcome_horizon", *sign_columns,
            "all_same_direction", "conflict_flag",
        ])
    def variant_name(value: str) -> str:
        adjusted = value.startswith("short_cover_adjusted__")
        normalized = "__volume_ratio__" in value
        if adjusted and normalized: return "adjusted_normalized_sign"
        if adjusted: return "adjusted_raw_sign"
        if normalized: return "normalized_sign"
        return "raw_sign"

    detail["variant"] = detail["predictor"].map(variant_name)
    detail["effect_sign"] = np.sign(detail["effect_vs_unconditional"])
    detail["statistical_evidence"] = detail.get("evidence_level", "No Evidence")
    keys = ["k", "rolling_window", "pr_group", "outcome_horizon"]
    pivot = detail.pivot_table(index=keys, columns="variant", values="effect_sign", aggfunc="first")
    pivot = pivot.reset_index()
    for column in sign_columns:
        if column not in pivot:
            pivot[column] = np.nan
    pivot["all_same_direction"] = pivot[sign_columns].nunique(axis=1, dropna=True).le(1) & pivot[sign_columns].notna().all(axis=1)
    pivot["conflict_flag"] = pivot[sign_columns].nunique(axis=1, dropna=True).gt(1)
    return detail, pivot[keys + sign_columns + ["all_same_direction", "conflict_flag"]]


def classify_pr_shapes(pr_bins: pd.DataFrame) -> pd.DataFrame:
    columns = ["predictor", "family", "k", "rolling_window", "outcome_horizon", "shape_classification"]
    if pr_bins.empty or "family" not in pr_bins:
        return pd.DataFrame(columns=columns)
    subset = pr_bins[pr_bins["family"].eq("approx_margin_maintenance")]
    if subset.empty:
        return pd.DataFrame(columns=columns)
    order = ["PR0-5", "PR5-20", "PR20-40", "PR40-60", "PR60-80", "PR80-95", "PR95-100"]
    rows = []
    keys = ["predictor", "family", "k", "rolling_window", "outcome_horizon"]
    for key, group in subset.groupby(keys, dropna=False):
        means = group.set_index("pr_group")["mean_return"].reindex(order)
        if means.isna().any():
            shape = "insufficient_data"
        else:
            delta = np.diff(means.to_numpy())
            minimum = int(np.argmin(means.to_numpy()))
            maximum = int(np.argmax(means.to_numpy()))
            if np.all(delta >= 0): shape = "monotonic_positive"
            elif np.all(delta <= 0): shape = "monotonic_negative"
            elif 0 < minimum < 6 and np.all(delta[:minimum] <= 0) and np.all(delta[minimum:] >= 0): shape = "U_shape"
            elif 0 < maximum < 6 and np.all(delta[:maximum] >= 0) and np.all(delta[maximum:] <= 0): shape = "inverted_U"
            else: shape = "mixed"
        rows.append(dict(zip(keys, key)) | {"shape_classification": shape})
    return pd.DataFrame(rows, columns=columns)
