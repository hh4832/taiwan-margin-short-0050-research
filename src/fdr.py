from __future__ import annotations

import pandas as pd
from statsmodels.stats.multitest import multipletests


def apply_fdr(results: pd.DataFrame, p_col: str = "raw_p_value") -> pd.DataFrame:
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
