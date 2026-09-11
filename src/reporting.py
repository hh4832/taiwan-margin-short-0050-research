from __future__ import annotations

import json
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


OUTPUT_NAMES = [
    "dataset_coverage.csv", "reconciliation_summary.csv", "universe_diagnostics.csv",
    "feature_catalog.csv", "level_feature_diagnostics.csv", "primary_results.csv",
    "fdr_results.csv", "fdr_diagnostics.csv", "controlled_results.csv",
    "controlled_regression_diagnostics.csv",
    "pr_bin_results.csv", "annual_results.csv", "annual_robustness_summary.csv",
    "neighborhood_consistency.csv", "short_cover_variant_results.csv",
    "variant_direction_consistency.csv", "shape_diagnostics.csv",
    "universe_sensitivity_results.csv", "robustness_results.csv",
    "signal_summary.md", "thermometer_signals.csv",
]


def git_value(args: list[str], default="unknown") -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return default


def create_run_directory(root: Path, timezone: str) -> tuple[Path, str]:
    commit = git_value(["rev-parse", "--short", "HEAD"])
    stamp = datetime.now(ZoneInfo(timezone)).strftime("%Y%m%d_%H%M%S")
    path = root / f"{stamp}_{commit}"
    path.mkdir(parents=True, exist_ok=False)
    return path, commit


def write_run_info(path: Path, config, coverage: pd.DataFrame, finlab_version="unknown") -> None:
    coverage_map = {r.dataset: r for r in coverage.itertuples()}
    lines = {
        "repository": config.repository, "branch": git_value(["branch", "--show-current"]),
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "run_timestamp": datetime.now(ZoneInfo(config.timezone)).isoformat(), "timezone": config.timezone,
        "python_version": platform.python_version(), "finlab_version": finlab_version,
        "suspension_analysis": "retrospective sensitivity; explicit start/end inclusive; missing end=start",
        "suspension_publication_time_verified": False,
        "rolling_windows": config.rolling_windows, "k_values": config.k_values,
        "pr_groups": config.pr_edges, "outcome_horizons": config.outcome_horizons,
        "robustness_min_years": config.robustness_min_years,
        "robustness_min_direction_ratio": config.robustness_min_direction_ratio,
        "robustness_min_neighborhood_score": config.robustness_min_neighborhood_score,
        "robustness_max_year_sample_share": config.robustness_max_year_sample_share,
        "controlled_min_total_n": config.min_group_n,
        "controlled_min_tail_n": config.min_group_n,
        "controlled_min_control_n": config.min_group_n,
        "controlled_leverage_tolerance": config.controlled_leverage_tolerance,
        "controlled_max_condition_number": config.controlled_max_condition_number,
        "primary_universe": config.primary_universe,
    }
    for dataset, prefix in [("open", "0050_price"), ("margin_balance", "margin_transactions"), ("aggregate_balance", "margin_balance"), ("market_amount", "market_transaction_info"), ("market_value", "market_value"), ("short_suspension", "short_suspension")]:
        row = coverage_map.get(dataset)
        lines[f"{prefix}_start"] = getattr(row, "first_date", "missing")
        lines[f"{prefix}_end"] = getattr(row, "latest_observation_date", "missing")
    (path / "run_info.txt").write_text("\n".join(f"{k}={v}" for k, v in lines.items()) + "\n", encoding="utf-8")


def get_latest_tradable_signal_date(close_0050: pd.Series):
    """Latest d0 with an observed target-market close, never an auxiliary date."""
    observed = close_0050.replace([float("inf"), float("-inf")], float("nan")).dropna()
    if observed.empty:
        raise ValueError("No valid 0050 close is available for a tradable signal date")
    return pd.Timestamp(observed.index.max())


def add_suspension_reporting_fields(frame: pd.DataFrame, coverage_limitation: bool) -> pd.DataFrame:
    """Add explicit research/live metadata to every predictor-level output."""
    out = frame.copy()
    if "predictor" not in out:
        return out
    adjusted = out["predictor"].astype("string").str.contains("_adjusted", na=False)
    out["retrospective_sensitivity"] = adjusted
    out["publication_time_verified"] = False
    out["SUSPENSION_COVERAGE_LIMITATION"] = adjusted & bool(coverage_limitation)
    out["is_live_eligible"] = ~adjusted
    return out


def assess_signals(
    results: pd.DataFrame,
    annual_summary: pd.DataFrame | None = None,
    neighborhood: pd.DataFrame | None = None,
    variant_consistency: pd.DataFrame | None = None,
    suspension_coverage_limitation: bool = False,
    shape_diagnostics: pd.DataFrame | None = None,
    min_years: int = 3,
    min_direction_ratio: float = 0.6,
    min_neighborhood_score: float = 0.6,
) -> pd.DataFrame:
    """Separate statistical evidence, robustness, live use, and decisions."""
    out = results.copy()
    if out.empty:
        for column in ("statistical_evidence", "robustness_status", "live_eligibility", "research_decision", "retrospective_sensitivity", "publication_time_verified", "SUSPENSION_COVERAGE_LIMITATION"):
            out[column] = pd.Series(dtype="object")
        return out
    out["statistical_evidence"] = out["evidence_level"]
    out = add_suspension_reporting_fields(out, suspension_coverage_limitation)
    out["live_eligibility"] = out["is_live_eligible"]
    out["robustness_status"] = "insufficient"

    annual_keys = ["predictor", "family", "k", "rolling_window", "pr_group", "outcome_horizon"]
    if annual_summary is not None and not annual_summary.empty:
        annual_cols = annual_keys + ["years_with_samples", "positive_year_ratio", "largest_year_sample_share", "few_year_concentration_flag"]
        out = out.merge(annual_summary[annual_cols], on=annual_keys, how="left")
        effect_positive = out["effect_vs_unconditional"].ge(0)
        out["annual_direction_ratio"] = out["positive_year_ratio"].where(effect_positive, 1 - out["positive_year_ratio"])
    else:
        out["years_with_samples"] = float("nan")
        out["annual_direction_ratio"] = float("nan")
        out["few_year_concentration_flag"] = True

    if neighborhood is not None and not neighborhood.empty:
        fixed = neighborhood[neighborhood["consistency_type"].eq("fixed_horizon")][
            ["family", "pr_group", "fixed_parameter", "consistency_score"]
        ].rename(columns={"fixed_parameter": "outcome_horizon", "consistency_score": "neighborhood_consistency_score"})
        out = out.merge(fixed, on=["family", "pr_group", "outcome_horizon"], how="left")
    else:
        out["neighborhood_consistency_score"] = float("nan")

    out["variant_conflict_flag"] = False
    if variant_consistency is not None and not variant_consistency.empty:
        keys = ["k", "rolling_window", "pr_group", "outcome_horizon"]
        out = out.merge(variant_consistency[keys + ["conflict_flag"]], on=keys, how="left")
        is_cover = out["family"].eq("short_cover")
        conflict_values = out.loc[is_cover, "conflict_flag"].astype("boolean").fillna(False).astype(bool)
        out.loc[is_cover, "variant_conflict_flag"] = conflict_values
        out = out.drop(columns="conflict_flag")

    out["shape_classification"] = pd.NA
    if shape_diagnostics is not None and not shape_diagnostics.empty:
        shape_keys = ["predictor", "family", "k", "rolling_window", "outcome_horizon"]
        out = out.merge(shape_diagnostics[shape_keys + ["shape_classification"]], on=shape_keys, how="left", suffixes=("", "_new"))
        out["shape_classification"] = out.pop("shape_classification_new").combine_first(out["shape_classification"])

    years_with_samples = pd.to_numeric(out["years_with_samples"], errors="coerce")
    enough = years_with_samples.fillna(0).ge(min_years)
    annual_ok = pd.to_numeric(out["annual_direction_ratio"], errors="coerce").ge(min_direction_ratio)
    neighborhood_ok = pd.to_numeric(out["neighborhood_consistency_score"], errors="coerce").ge(min_neighborhood_score)
    concentration = out["few_year_concentration_flag"].astype("boolean").fillna(True).astype(bool)
    conflict = out["variant_conflict_flag"].astype("boolean").fillna(False).astype(bool)
    robust = enough & annual_ok & neighborhood_ok & ~concentration & ~conflict
    non_directional_shape = out["shape_classification"].isin(["U_shape", "inverted_U", "mixed"])
    robust &= ~non_directional_shape
    assessed = enough & out["neighborhood_consistency_score"].notna()
    out.loc[assessed, "robustness_status"] = "mixed"
    out.loc[robust, "robustness_status"] = "robust"
    out["research_decision"] = "無法判定"
    evidence = out["statistical_evidence"].isin(["Level A", "Level B", "Level C"])
    out.loc[evidence, "research_decision"] = "修改後再測"
    out.loc[out["statistical_evidence"].eq("Level A") & robust & out["live_eligibility"], "research_decision"] = "保留"
    return out


def signal_summary(results: pd.DataFrame) -> str:
    header = "# 融資融券訊號研究摘要\n\n未執行實際 FinLab 資料時，不產生績效敘述。\n\n## 反對者觀點\n\n融資可能是追價；融券可能是避險；停券可製造假回補；全市場訊號未必代表 0050；制度會跨年代改變；多重檢定可產生不可重現單點；市值正規化受價格機械性影響。\n"
    if results.empty:
        return header
    strong = results[results["statistical_evidence"].isin(["Level A", "Level B", "Level C"])]
    body = []
    for r in strong.itertuples():
        direction = "非單調／無法判定" if r.shape_classification in {"U_shape", "inverted_U", "mixed"} else "偏多" if r.effect_vs_unconditional > 0 else "偏空"
        body.append(f"## {r.predictor}\n\n{r.pr_group}，{r.outcome_horizon}：平均報酬 {r.mean_return:.2%}，相對一般交易日 {r.effect_vs_unconditional:+.2%}，勝率 {r.win_rate:.1%}，N={r.N}，統計證據 {r.statistical_evidence}；robustness={r.robustness_status}；live eligible={r.live_eligibility}；方向：{direction}；結論：{r.research_decision}。\n")
    return header + "\n".join(body or ["\n目前沒有 raw p < 0.05 的訊號；結論：無法判定。\n"])


def thermometer_table(results: pd.DataFrame, signal_date) -> pd.DataFrame:
    columns = ["signal_date", "signal_name", "family", "direction", "lookback_k", "rolling_window", "percentile", "pr_group", "evidence_level", "statistical_evidence", "robustness_status", "live_eligibility", "retrospective_sensitivity", "publication_time_verified", "SUSPENSION_COVERAGE_LIMITATION", "research_decision", "target_horizon", "effect_vs_unconditional", "win_rate", "sample_size", "plain_language_description"]
    if results.empty:
        return pd.DataFrame(columns=columns)
    selected = results[results["evidence_level"].isin(["Level A", "Level B", "Level C"])].copy()
    rows = []
    for r in selected.itertuples():
        direction = "非單調／無法判定" if r.shape_classification in {"U_shape", "inverted_U", "mixed"} else "偏多" if r.effect_vs_unconditional > 0 else "偏空" if r.effect_vs_unconditional < 0 else "無法判定"
        rows.append({"signal_date": signal_date, "signal_name": r.predictor, "family": r.family, "direction": direction, "lookback_k": r.k, "rolling_window": r.rolling_window, "percentile": "<=5" if r.pr_group == "PR0-5" else ">=95", "pr_group": r.pr_group, "evidence_level": r.evidence_level, "statistical_evidence": r.statistical_evidence, "robustness_status": r.robustness_status, "live_eligibility": r.live_eligibility, "retrospective_sensitivity": r.retrospective_sensitivity, "publication_time_verified": r.publication_time_verified, "SUSPENSION_COVERAGE_LIMITATION": r.SUSPENSION_COVERAGE_LIMITATION, "research_decision": r.research_decision, "target_horizon": r.outcome_horizon, "effect_vs_unconditional": r.effect_vs_unconditional, "win_rate": r.win_rate, "sample_size": r.N, "plain_language_description": f"{r.predictor} 位於 {r.pr_group} 時，相對一般交易日報酬差 {r.effect_vs_unconditional:+.2%}；證據 {r.statistical_evidence}，robustness {r.robustness_status}，結論 {r.research_decision}。"})
    return pd.DataFrame(rows, columns=columns)


def backup_to_drive(run_dir: Path, drive_root: Path) -> Path:
    if not drive_root.parent.exists():
        raise RuntimeError("Google Drive is not mounted; core output remains local")
    target = drive_root / run_dir.name
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    return Path(shutil.copytree(run_dir, target))
