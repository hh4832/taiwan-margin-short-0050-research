from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def dataset_coverage(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in datasets.items():
        nonempty = frame.dropna(how="all")
        rows.append({
            "dataset": name,
            "first_date": frame.index.min(),
            "last_date": frame.index.max(),
            "rows": len(frame),
            "columns": frame.shape[1],
            "non_null_ratio": float(frame.notna().to_numpy().mean()) if frame.size else np.nan,
            "latest_observation_date": nonempty.index.max() if len(nonempty) else pd.NaT,
        })
    return pd.DataFrame(rows)


def reconciliation(lhs: pd.DataFrame, rhs: pd.DataFrame, name: str, tolerance: float = 1e-9) -> dict:
    left, right = lhs.align(rhs, join="inner", axis=None)
    valid = left.notna() & right.notna()
    error = (left - right).where(valid).stack(future_stack=True).abs()
    if error.empty:
        raise ValueError(f"{name}: no comparable observations")
    return {
        "check": name,
        "n": int(error.size),
        "exact_match_ratio": float((error <= tolerance).mean()),
        "mean_absolute_error": float(error.mean()),
        "max_absolute_error": float(error.max()),
    }


def run_reconciliation(d: Mapping[str, pd.DataFrame], tolerance: float, minimum: float) -> pd.DataFrame:
    checks = [
        reconciliation(d["margin_balance"] - d["margin_prev_balance"], d["margin_buy"] - d["margin_sell"] - d["margin_cash_repayment"], "margin", tolerance),
        reconciliation(d["short_balance"] - d["short_prev_balance"], d["short_sell"] - d["short_cover"] - d["short_stock_repayment"], "short", tolerance),
    ]
    result = pd.DataFrame(checks)
    if (result["exact_match_ratio"] < minimum).any():
        raise ValueError("Stage 0 reconciliation failed; research stopped")
    return result


def aggregate_comparison(individual: pd.Series, market: pd.Series, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair = pd.concat({"individual": individual, "market": market}, axis=1).dropna()
    pair["absolute_difference"] = pair["individual"] - pair["market"]
    pair["percentage_difference"] = pair["absolute_difference"] / pair["market"].replace(0, np.nan)
    summary = pd.DataFrame([{
        "check": name,
        "pearson": pair["individual"].corr(pair["market"], method="pearson"),
        "spearman": pair["individual"].corr(pair["market"], method="spearman"),
        "mean_absolute_difference": pair["absolute_difference"].abs().mean(),
    }])
    annual = pair.groupby(pair.index.year)[["absolute_difference", "percentage_difference"]].mean().reset_index(names="year")
    annual.insert(0, "check", name)
    return summary, annual
