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


def reconciliation_errors(lhs: pd.DataFrame, rhs: pd.DataFrame, name: str) -> pd.Series:
    left, right = lhs.align(rhs, join="inner", axis=None)
    valid = left.notna() & right.notna()
    error = (left - right).where(valid).stack(future_stack=True).abs()
    if error.empty:
        raise ValueError(f"{name}: no comparable observations")
    error.index = error.index.set_names(["date", "symbol"])
    return error


def reconciliation(lhs: pd.DataFrame, rhs: pd.DataFrame, name: str, tolerance: float = 1e-9) -> dict:
    error = reconciliation_errors(lhs, rhs, name)
    mismatches = error[error > tolerance]
    return {
        "check": name,
        "n": int(error.size),
        "mismatch_n": int(mismatches.size),
        "exact_match_ratio": float((error <= tolerance).mean()),
        "mean_absolute_error": float(error.mean()),
        "max_absolute_error": float(error.max()),
        "first_mismatch_date": mismatches.index.get_level_values("date").min() if len(mismatches) else pd.NaT,
        "last_mismatch_date": mismatches.index.get_level_values("date").max() if len(mismatches) else pd.NaT,
    }


def _identity_pairs(d: Mapping[str, pd.DataFrame]):
    return [
        (d["margin_balance"] - d["margin_prev_balance"], d["margin_buy"] - d["margin_sell"] - d["margin_cash_repayment"], "margin"),
        (d["short_balance"] - d["short_prev_balance"], d["short_sell"] - d["short_cover"] - d["short_stock_repayment"], "short"),
    ]


def run_reconciliation(d: Mapping[str, pd.DataFrame], tolerance: float, minimum: float) -> pd.DataFrame:
    checks = [
        reconciliation(lhs, rhs, name, tolerance) for lhs, rhs, name in _identity_pairs(d)
    ]
    result = pd.DataFrame(checks)
    if (result["exact_match_ratio"] < minimum).any():
        details = result[["check", "n", "mismatch_n", "exact_match_ratio", "max_absolute_error", "first_mismatch_date", "last_mismatch_date"]].to_string(index=False)
        raise ValueError(f"Stage 0 reconciliation failed; research stopped.\n{details}")
    return result


def stabilize_reconciliation(
    datasets: Mapping[str, pd.DataFrame],
    tolerance: float,
    minimum: float,
    max_trailing_dates: int = 3,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Exclude only a short, inconsistent data-provider tail; never historical mismatches."""
    try:
        result = run_reconciliation(datasets, tolerance, minimum)
        result["status"] = "PASS"
        result["excluded_trailing_dates"] = ""
        return dict(datasets), result
    except ValueError as initial_error:
        errors = [reconciliation_errors(lhs, rhs, name) for lhs, rhs, name in _identity_pairs(datasets)]
        all_dates = pd.DatetimeIndex(sorted(set().union(*(e.index.get_level_values("date") for e in errors))))
        bad_dates = pd.DatetimeIndex(sorted(set().union(*(
            e[e > tolerance].index.get_level_values("date") for e in errors
        ))))
        allowed_tail = set(all_dates[-max_trailing_dates:])
        if bad_dates.empty or not set(bad_dates).issubset(allowed_tail):
            raise initial_error
        cutoff_candidates = all_dates[all_dates < bad_dates.min()]
        if cutoff_candidates.empty:
            raise initial_error
        cutoff = cutoff_candidates.max()
        trimmed = {name: frame.loc[frame.index <= cutoff].copy() for name, frame in datasets.items()}
        try:
            result = run_reconciliation(trimmed, tolerance, minimum)
        except ValueError:
            raise initial_error
        excluded = ",".join(pd.Timestamp(d).strftime("%Y-%m-%d") for d in bad_dates)
        result["status"] = "PASS_AFTER_TRAILING_EXCLUSION"
        result["research_as_of_date"] = cutoff
        result["excluded_trailing_dates"] = excluded
        return trimmed, result


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
