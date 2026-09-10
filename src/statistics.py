from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


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
                result = compare_group(joined[outcome], selected)
                rows.append({"predictor": feature, "family": meta.get("family", feature.split("__")[0]), "k": meta.get("k"), "rolling_window": meta.get("window"), "pr_group": label, "outcome_horizon": outcome, **result})
    return pd.DataFrame(rows)


def controlled_regression(signal: pd.Series, future: pd.Series, prior_return: pd.Series) -> dict:
    frame = pd.concat({"future": future, "signal": signal, "prior": prior_return}, axis=1).dropna()
    if len(frame) < 20:
        return {"N": len(frame), "signal_beta": np.nan, "signal_p_value": np.nan}
    fit = sm.OLS(frame["future"], sm.add_constant(frame[["signal", "prior"]])).fit(cov_type="HC3")
    return {"N": len(frame), "signal_beta": fit.params["signal"], "signal_p_value": fit.pvalues["signal"]}


def run_controlled_tests(features: pd.DataFrame, outcomes: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    rows = []
    joined = features.join(outcomes, how="inner")
    for feature in [c for c in features if c.endswith("_percentile")]:
        meta = features.attrs.get("catalog", {}).get(feature, {})
        extreme = ((joined[feature] <= 5) | (joined[feature] >= 95)).astype(float)
        for prior_k in (1, 3, 5, 10):
            prior = close.pct_change(prior_k).reindex(joined.index)
            for outcome in [c for c in outcomes if c.startswith("O1_C")]:
                fit = controlled_regression(extreme, joined[outcome], prior)
                rows.append({"predictor": feature, "family": meta.get("family"), "prior_k": prior_k, "outcome_horizon": outcome, **fit})
    return pd.DataFrame(rows)


def annual_results(results: pd.DataFrame, features: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    candidates = results.nsmallest(min(30, len(results)), "raw_p_value") if len(results) else results
    joined = features.join(outcomes, how="inner")
    rows = []
    for r in candidates.itertuples():
        selected = joined[r.predictor].le(5) if r.pr_group == "PR0-5" else joined[r.predictor].ge(95)
        values = joined[r.outcome_horizon]
        for year, idx in joined.groupby(joined.index.year).groups.items():
            sample = values.loc[idx][selected.loc[idx]].dropna()
            rows.append({"predictor": r.predictor, "pr_group": r.pr_group, "outcome_horizon": r.outcome_horizon, "year": year, "N": len(sample), "mean_return": sample.mean(), "win_rate": (sample > 0).mean() if len(sample) else np.nan})
    return pd.DataFrame(rows)


def regime_results(results: pd.DataFrame, features: pd.DataFrame, outcomes: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    candidates = results.nsmallest(min(30, len(results)), "raw_p_value") if len(results) else results
    joined = features.join(outcomes, how="inner")
    trend = close / close.rolling(252, min_periods=252).mean() - 1
    vol = close.pct_change().rolling(60, min_periods=60).std()
    regime = pd.Series("sideways", index=close.index)
    regime[trend > 0.05] = "bull"; regime[trend < -0.05] = "bear"
    vol_regime = pd.Series(np.where(vol >= vol.rolling(252, min_periods=126).median(), "high_vol", "low_vol"), index=close.index)
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
    if results.empty:
        return pd.DataFrame(columns=["predictor", "family", "pr_group", "effect_sign", "neighborhood_consistency_score", "note"])
    out = []
    for (family, pr), group in results.groupby(["family", "pr_group"], dropna=False):
        signs = np.sign(group["effect_vs_unconditional"].dropna())
        if signs.empty:
            score = np.nan; dominant = 0
        else:
            dominant = 1 if signs.mean() >= 0 else -1
            score = float((signs == dominant).mean())
        out.append({"predictor": "family-neighborhood", "family": family, "pr_group": pr, "effect_sign": dominant, "neighborhood_consistency_score": score, "note": "Robustness only; does not replace FDR."})
    return pd.DataFrame(out)
