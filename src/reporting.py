from __future__ import annotations

import json
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


OUTPUT_NAMES = ["dataset_coverage.csv", "reconciliation_summary.csv", "universe_diagnostics.csv", "feature_catalog.csv", "primary_results.csv", "fdr_results.csv", "annual_results.csv", "robustness_results.csv", "signal_summary.md", "thermometer_signals.csv"]


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
        "rolling_windows": config.rolling_windows, "k_values": config.k_values,
        "pr_groups": config.pr_edges, "outcome_horizons": config.outcome_horizons,
        "primary_universe": config.primary_universe,
    }
    for dataset, prefix in [("open", "0050_price"), ("margin_balance", "margin_transactions"), ("aggregate_balance", "margin_balance"), ("market_amount", "market_transaction_info"), ("market_value", "market_value"), ("short_suspension", "short_suspension")]:
        row = coverage_map.get(dataset)
        lines[f"{prefix}_start"] = getattr(row, "first_date", "missing")
        lines[f"{prefix}_end"] = getattr(row, "latest_observation_date", "missing")
    (path / "run_info.txt").write_text("\n".join(f"{k}={v}" for k, v in lines.items()) + "\n", encoding="utf-8")


def signal_summary(results: pd.DataFrame) -> str:
    header = "# 融資融券訊號研究摘要\n\n未執行實際 FinLab 資料時，不產生績效敘述。\n\n## 反對者觀點\n\n融資可能是追價；融券可能是避險；停券可製造假回補；全市場訊號未必代表 0050；制度會跨年代改變；多重檢定可產生不可重現單點；市值正規化受價格機械性影響。\n"
    if results.empty:
        return header
    strong = results[results["evidence_level"].isin(["Level A", "Level B", "Level C"])]
    body = []
    for r in strong.itertuples():
        direction = "偏多" if r.effect_vs_unconditional > 0 else "偏空"
        decision = "保留" if r.evidence_level == "Level A" else "修改後再測"
        body.append(f"## {r.predictor}\n\n{r.pr_group}，{r.outcome_horizon}：平均報酬 {r.mean_return:.2%}，相對一般交易日 {r.effect_vs_unconditional:+.2%}，勝率 {r.win_rate:.1%}，N={r.N}，證據 {r.evidence_level}；方向：{direction}；結論：{decision}。\n")
    return header + "\n".join(body or ["\n目前沒有 raw p < 0.05 的訊號；結論：無法判定。\n"])


def thermometer_table(results: pd.DataFrame, signal_date) -> pd.DataFrame:
    columns = ["signal_date", "signal_name", "family", "direction", "lookback_k", "rolling_window", "percentile", "pr_group", "evidence_level", "target_horizon", "effect_vs_unconditional", "win_rate", "sample_size", "plain_language_description"]
    if results.empty:
        return pd.DataFrame(columns=columns)
    selected = results[results["evidence_level"].isin(["Level A", "Level B", "Level C"])].copy()
    rows = []
    for r in selected.itertuples():
        direction = "偏多" if r.effect_vs_unconditional > 0 else "偏空" if r.effect_vs_unconditional < 0 else "無法判定"
        rows.append({"signal_date": signal_date, "signal_name": r.predictor, "family": r.family, "direction": direction, "lookback_k": r.k, "rolling_window": r.rolling_window, "percentile": "<=5" if r.pr_group == "PR0-5" else ">=95", "pr_group": r.pr_group, "evidence_level": r.evidence_level, "target_horizon": r.outcome_horizon, "effect_vs_unconditional": r.effect_vs_unconditional, "win_rate": r.win_rate, "sample_size": r.N, "plain_language_description": f"{r.predictor} 位於 {r.pr_group} 時，相對一般交易日報酬差 {r.effect_vs_unconditional:+.2%}；{r.evidence_level}。"})
    return pd.DataFrame(rows, columns=columns)


def backup_to_drive(run_dir: Path, drive_root: Path) -> Path:
    if not drive_root.parent.exists():
        raise RuntimeError("Google Drive is not mounted; core output remains local")
    target = drive_root / run_dir.name
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    return Path(shutil.copytree(run_dir, target))
