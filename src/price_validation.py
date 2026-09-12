from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import FINLAB_FIELDS


SPLIT_START = pd.Timestamp("2025-06-10")
SPLIT_END = pd.Timestamp("2025-06-18")
SANITY_START = pd.Timestamp("2025-05-20")
SANITY_END = pd.Timestamp("2025-06-20")
OUTCOME_COLUMNS = tuple(f"O1_C{h}" for h in (1, 2, 3, 5, 10, 20))
COMPARISON_KEYS = ("family", "predictor", "k", "rolling_window", "pr_group", "outcome_horizon")


def assert_adjusted_outcome_mapping() -> None:
    if FINLAB_FIELDS.get("open") != "etl:adj_open" or FINLAB_FIELDS.get("close") != "etl:adj_close":
        raise RuntimeError("0050 outcome source must remain etl:adj_open / etl:adj_close")


def validate_split_window(adjusted_open: pd.Series, adjusted_close: pd.Series, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Validate the known 2025/6 split window without imposing a global return cap."""
    assert_adjusted_outcome_mapping()
    if adjusted_close.dropna().empty:
        raise ValueError("0050 adjusted close has no valid observations")
    if adjusted_close.dropna().index.max() < SPLIT_START:
        return pd.DataFrame([
            {"metric": "known_split_window_available", "value": False, "status": "not_testable"},
            {"metric": "target_symbol", "value": "0050", "status": "diagnostic"},
        ])
    if pd.concat({"open": adjusted_open, "close": adjusted_close}, axis=1).loc[SPLIT_START:SPLIT_END].dropna().empty:
        raise ValueError("0050 adjusted prices are missing in the 2025/6 split window")
    if SPLIT_START not in adjusted_close.index or SPLIT_END not in adjusted_close.index:
        raise ValueError("Known split validation dates are missing from adjusted close")
    adjusted_return = adjusted_close.loc[SPLIT_END] / adjusted_close.loc[SPLIT_START] - 1
    if not np.isfinite(adjusted_return) or adjusted_return <= -0.50:
        raise ValueError(f"0050 adjusted prices still contain a split-like artifact: {adjusted_return}")
    window = outcomes.loc[SANITY_START:SANITY_END, list(OUTCOME_COLUMNS)]
    minima = window.min(skipna=True)
    if minima.le(-0.50).any():
        raise ValueError(f"Known 2025/6 outcome window contains split-like artifacts: {minima[minima.le(-0.50)].to_dict()}")
    rows = [{"metric": "adjusted_close_return_2025_06_10_to_2025_06_18", "value": float(adjusted_return), "status": "pass"}]
    rows.extend({
        "metric": f"minimum_{column}_2025_05_20_to_2025_06_20",
        "value": float(minima[column]) if pd.notna(minima[column]) else np.nan,
        "status": "pass" if pd.notna(minima[column]) else "not_testable",
    } for column in OUTCOME_COLUMNS)
    rows.extend([
        {"metric": "adjusted_open_first_valid_date", "value": adjusted_open.first_valid_index(), "status": "diagnostic"},
        {"metric": "adjusted_open_last_valid_date", "value": adjusted_open.last_valid_index(), "status": "diagnostic"},
        {"metric": "adjusted_open_missing_count", "value": int(adjusted_open.isna().sum()), "status": "diagnostic"},
        {"metric": "adjusted_close_first_valid_date", "value": adjusted_close.first_valid_index(), "status": "diagnostic"},
        {"metric": "adjusted_close_last_valid_date", "value": adjusted_close.last_valid_index(), "status": "diagnostic"},
        {"metric": "adjusted_close_missing_count", "value": int(adjusted_close.isna().sum()), "status": "diagnostic"},
        {"metric": "target_symbol", "value": "0050", "status": "diagnostic"},
    ])
    return pd.DataFrame(rows)


def raw_vs_adjusted_split_validation(raw_close: pd.Series, adjusted_close: pd.Series) -> pd.DataFrame:
    for name, series in {"raw": raw_close, "adjusted": adjusted_close}.items():
        missing = {SPLIT_START, SPLIT_END} - set(series.index)
        if missing:
            raise ValueError(f"{name} close missing split validation dates: {sorted(missing)}")
    raw_return = raw_close.loc[SPLIT_END] / raw_close.loc[SPLIT_START] - 1
    adjusted_return = adjusted_close.loc[SPLIT_END] / adjusted_close.loc[SPLIT_START] - 1
    if raw_return > -0.50 or adjusted_return <= -0.50:
        raise ValueError("Raw/adjusted split diagnostic does not show the expected discontinuity removal")
    return pd.DataFrame([
        {"price_source": "price:收盤價", "start": raw_close.loc[SPLIT_START], "end": raw_close.loc[SPLIT_END], "return": raw_return},
        {"price_source": "etl:adj_close", "start": adjusted_close.loc[SPLIT_START], "end": adjusted_close.loc[SPLIT_END], "return": adjusted_return},
    ])


def split_artifact_observation_counts(
    raw_open: pd.Series,
    raw_close: pd.Series,
    adjusted_open: pd.Series,
    adjusted_close: pd.Series,
) -> pd.DataFrame:
    """Count known-window d0 rows where raw prices create a split-scale loss."""
    from .outcomes import build_outcomes

    horizons = (1, 2, 3, 5, 10, 20)
    raw = build_outcomes(raw_open, raw_close, horizons)
    adjusted = build_outcomes(adjusted_open, adjusted_close, horizons)
    columns = (*OUTCOME_COLUMNS, "C0_O1_descriptive_nontradable")
    rows = []
    for column in columns:
        pair = pd.concat({"raw": raw[column], "adjusted": adjusted[column]}, axis=1).loc[SANITY_START:SANITY_END].dropna()
        contaminated = pair["raw"].le(-0.50) & pair["adjusted"].gt(-0.50)
        rows.append({
            "outcome_horizon": column,
            "split_artifact_observations": int(contaminated.sum()),
            "affected_signal_dates": ",".join(pair.index[contaminated].strftime("%Y-%m-%d")),
            "minimum_raw_return": pair["raw"].min() if len(pair) else np.nan,
            "minimum_adjusted_return": pair["adjusted"].min() if len(pair) else np.nan,
        })
    return pd.DataFrame(rows)


def compare_research_results(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    keys = [column for column in COMPARISON_KEYS if column in old.columns and column in new.columns]
    if not keys:
        raise ValueError("Old/new result files have no research cell keys in common")
    metrics = ["N", "mean_return", "median_return", "win_rate", "effect_vs_unconditional", "raw_p_value", "global_fdr_q_value", "evidence_level", "research_decision"]
    compared = old[keys + [c for c in metrics if c in old]].merge(
        new[keys + [c for c in metrics if c in new]], on=keys, how="outer", suffixes=("_old", "_new")
    )
    if {"effect_vs_unconditional_old", "effect_vs_unconditional_new"}.issubset(compared):
        compared["effect_delta"] = compared["effect_vs_unconditional_new"] - compared["effect_vs_unconditional_old"]
        compared["sign_changed"] = np.sign(compared["effect_vs_unconditional_old"]) != np.sign(compared["effect_vs_unconditional_new"])
    if {"evidence_level_old", "evidence_level_new"}.issubset(compared):
        compared["evidence_changed"] = compared["evidence_level_old"] != compared["evidence_level_new"]
    return compared


def comparison_horizon_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    if "outcome_horizon" not in comparison:
        raise ValueError("Comparison lacks outcome_horizon")
    rows = []
    for horizon, group in comparison.groupby("outcome_horizon", dropna=False):
        old_level = group.get("evidence_level_old", pd.Series(index=group.index, dtype="object"))
        new_level = group.get("evidence_level_new", pd.Series(index=group.index, dtype="object"))
        sign_changed = group.get("sign_changed", pd.Series(False, index=group.index)).fillna(False)
        rows.append({
            "outcome_horizon": horizon, "cells": len(group),
            "unchanged": int((~sign_changed & old_level.eq(new_level)).sum()),
            "sign_flip": int(sign_changed.sum()),
            "level_a_lost": int((old_level.eq("Level A") & ~new_level.eq("Level A")).sum()),
            "level_a_gained": int((~old_level.eq("Level A") & new_level.eq("Level A")).sum()),
        })
    return pd.DataFrame(rows)


def write_corporate_action_impact_report(
    path: str | Path,
    validation: pd.DataFrame,
    comparison: pd.DataFrame | None = None,
    outcome_impact: pd.DataFrame | None = None,
) -> Path:
    path = Path(path)
    sources = validation.get("price_source", pd.Series(dtype="object"))
    raw = validation.loc[sources.eq("price:收盤價"), "return"] if "return" in validation else pd.Series(dtype=float)
    adjusted = validation.loc[sources.eq("etl:adj_close"), "return"] if "return" in validation else pd.Series(dtype=float)
    lines = [
        "# Corporate Action Impact Report", "", "## A. 污染來源", "", "未還原 0050 raw OHLC。", "",
        "## B. Corporate action", "", "2025/6 0050 1:4 split。", "",
        "## C. 受影響 outcome", "", "C0_O1 與跨分割尺度的 C2/C3/C5/C10/C20；C1 亦已實際驗證。", "",
        "## D. Old vs adjusted price", "", f"Raw return: {raw.iloc[0]:.6%}" if len(raw) else "Raw return: not available", f"Adjusted return: {adjusted.iloc[0]:.6%}" if len(adjusted) else "Adjusted return: not available", "",
        "## E. 受污染 observation 數量", "",
        f"Known-window split artifacts: {int(outcome_impact['split_artifact_observations'].sum())}" if outcome_impact is not None else "尚未執行 raw-vs-adjusted outcome comparison。", "",
        "## F. Evidence level 變化", "", f"Changed cells: {int(comparison['evidence_changed'].sum())}" if comparison is not None and "evidence_changed" in comparison else "尚未完成正式 rerun。", "",
        "## G. Survived conclusions", "", "正式 rerun 前無法判定。", "",
        "## H. Withdrawn conclusions", "", "所有依賴 raw-price 跨分割報酬的舊結論均暫時撤回，等待 adjusted rerun。", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
