from __future__ import annotations

import pandas as pd


def apply_fdr(results: pd.DataFrame, p_col: str = "raw_p_value") -> pd.DataFrame:
    from statsmodels.stats.multitest import multipletests

    out = results.copy()
    out["family_fdr_q_value"] = float("nan")
    for _, idx in out.groupby("family").groups.items():
        valid = out.loc[idx, p_col].notna()
        use = out.loc[idx].index[valid]
        if len(use):
            out.loc[use, "family_fdr_q_value"] = multipletests(out.loc[use, p_col], method="fdr_bh")[1]
    valid = out[p_col].notna()
    if valid.any():
        out.loc[valid, "global_fdr_q_value"] = multipletests(out.loc[valid, p_col], method="fdr_bh")[1]
    else:
        out["global_fdr_q_value"] = float("nan")
    out["evidence_level"] = "No Evidence"
    out.loc[out[p_col] < 0.05, "evidence_level"] = "Level C"
    out.loc[out["family_fdr_q_value"] < 0.05, "evidence_level"] = "Level B"
    out.loc[out["global_fdr_q_value"] < 0.05, "evidence_level"] = "Level A"
    return out


def fdr_diagnostics(results: pd.DataFrame, p_col: str = "raw_p_value") -> pd.DataFrame:
    """Report the inferential universe; descriptive tables never enter here."""
    valid = results[results[p_col].notna()] if p_col in results else results.iloc[0:0]
    rows = [{"metric": "number_of_tests_total", "family": "ALL", "count": len(valid)}]
    rows.extend(
        {"metric": "number_of_tests_by_family", "family": family, "count": len(group)}
        for family, group in valid.groupby("family", dropna=False)
    )
    counts = results.get("evidence_level", pd.Series(dtype="object")).value_counts()
    rows.extend(
        {"metric": "evidence_level_count", "family": level, "count": int(counts.get(level, 0))}
        for level in ("Level A", "Level B", "Level C", "No Evidence")
    )
    return pd.DataFrame(rows)
