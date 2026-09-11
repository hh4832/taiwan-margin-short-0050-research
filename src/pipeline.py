from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ResearchConfig
from .data_loader import load_finlab_data
from .diagnostics import aggregate_comparison, dataset_coverage, stabilize_reconciliation
from .features import adjusted_short_change, adjust_short_for_suspensions, build_feature_catalog, kday_change, position_market_value, rolling_ratio, trailing_percentile
from .outcomes import build_outcomes
from .reporting import create_run_directory, signal_summary, thermometer_table, write_run_info
from .statistics import annual_results, neighborhood_consistency, regime_results, run_controlled_tests, run_primary_tests
from .universe import build_universe_diagnostics


def _sum_columns(frame: pd.DataFrame, names: list[str]) -> pd.Series:
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required aggregate columns: {sorted(missing)}")
    return frame[names].sum(axis=1, min_count=1)


def _market_totals(d: dict[str, pd.DataFrame]) -> tuple[pd.Series, pd.Series]:
    return d["market_amount"][["TAIEX", "OTC"]].sum(axis=1, min_count=1), d["market_volume"][["TAIEX", "OTC"]].sum(axis=1, min_count=1)


def _build_base_series(d: dict[str, pd.DataFrame], primary_symbols: set[str]) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    amount, volume = _market_totals(d)
    adjusted, suspension_mask = adjust_short_for_suspensions(d, d["short_suspension"])
    close = d["close"]
    market_cap = d["market_value"].sum(axis=1, min_count=1)
    margin_position = position_market_value(d["margin_balance"], close, primary_symbols)
    short_position = position_market_value(d["short_balance"], close, primary_symbols)
    short_position_adjusted = position_market_value(adjusted["short_balance"], close, primary_symbols)
    credit = _sum_columns(d["aggregate_balance"], ["上市融資交易金額", "上櫃融資交易金額"])
    series = {
        "margin_buy_raw": d["margin_buy"].sum(axis=1, min_count=1),
        "margin_sell_raw": d["margin_sell"].sum(axis=1, min_count=1),
        "margin_cash_repayment_raw": d["margin_cash_repayment"].sum(axis=1, min_count=1),
        "short_sell_raw": d["short_sell"].sum(axis=1, min_count=1),
        "short_cover_raw": d["short_cover"].sum(axis=1, min_count=1),
        "short_repayment_raw": d["short_stock_repayment"].sum(axis=1, min_count=1),
        "short_sell_adjusted": adjusted["short_sell"].sum(axis=1, min_count=1),
        "short_cover_adjusted": adjusted["short_cover"].sum(axis=1, min_count=1),
        "short_repayment_adjusted": adjusted["short_stock_repayment"].sum(axis=1, min_count=1),
        "margin_balance": d["margin_balance"].sum(axis=1, min_count=1),
        "short_balance": d["short_balance"].sum(axis=1, min_count=1),
        "short_balance_adjusted": adjusted["short_balance"].sum(axis=1, min_count=1),
        "margin_position_level": margin_position / market_cap,
        "margin_credit_level": credit / market_cap,
        "approx_margin_maintenance": margin_position / credit.replace(0, np.nan),
        "short_position_level": short_position / market_cap,
        "short_position_level_adjusted": short_position_adjusted / market_cap,
        "short_margin_ratio": short_position / margin_position.replace(0, np.nan),
        "market_amount": amount, "market_volume": volume,
    }
    suspension_diag = pd.DataFrame({
        "date": suspension_mask.index,
        "affected_security_count": suspension_mask.sum(axis=1).to_numpy(),
        "SUSPENSION_COVERAGE_LIMITATION": bool(d["short_suspension"].attrs.get("quarantined_rows", 0)),
        "limitation": d["short_suspension"].attrs.get(
            "limitation",
            "Retrospective sensitivity: key_date is not verified publication time.",
        ),
    })
    return series, suspension_diag


def build_features(d: dict[str, pd.DataFrame], primary_symbols: set[str], config: ResearchConfig) -> pd.DataFrame:
    s, _ = _build_base_series(d, primary_symbols)
    _, affected = adjust_short_for_suspensions(d, d["short_suspension"])
    values: dict[str, pd.Series] = {}
    catalog: dict[str, dict] = {}
    flow_families = ["margin_buy", "margin_sell", "margin_cash_repayment", "short_sell", "short_cover", "short_repayment"]
    for family in flow_families:
        variants = [f"{family}_raw"] + ([f"{family}_adjusted"] if family.startswith("short") else [])
        for variant in variants:
            for k in config.k_values:
                raw = s[variant].rolling(k, min_periods=k).sum()
                candidates = {f"{variant}__k{k}__raw": raw}
                # Lots are converted to shares before division by market traded shares.
                candidates[f"{variant}__k{k}__volume_ratio"] = rolling_ratio(s[variant] * 1000.0, s["market_volume"], k)
                if family in {"margin_buy", "margin_sell"}:
                    agg_key = "aggregate_buy" if family == "margin_buy" else "aggregate_sell"
                    amount_flow = _sum_columns(d[agg_key], ["上市融資交易金額", "上櫃融資交易金額"])
                    candidates[f"{family}__k{k}__amount_ratio"] = rolling_ratio(amount_flow, s["market_amount"], k)
                for name, base in candidates.items():
                    for window in config.rolling_windows:
                        col = f"{name}__w{window}_percentile"
                        values[col] = trailing_percentile(base, window)
                        catalog[col] = {"family": family, "k": k, "window": window, "variant": name}
    for family in ("margin_balance", "short_balance", "short_balance_adjusted", "approx_margin_maintenance"):
        for k in config.k_values:
            base = kday_change(s[family], k)
            if family == "short_balance_adjusted":
                base = adjusted_short_change(d["short_balance"], affected, k)
            for window in config.rolling_windows:
                col = f"{family}_change__k{k}__w{window}_percentile"
                values[col] = trailing_percentile(base, window)
                catalog[col] = {"family": family.replace("_adjusted", "_change"), "k": k, "window": window}
    for family in ("margin_position_level", "margin_credit_level", "approx_margin_maintenance", "short_position_level", "short_position_level_adjusted", "short_margin_ratio"):
        for window in config.rolling_windows:
            col = f"{family}__w{window}_percentile"
            values[col] = trailing_percentile(s[family], window)
            catalog[col] = {"family": family.replace("_adjusted", ""), "k": 1, "window": window}
    frame = pd.DataFrame(values).sort_index()
    frame.attrs["catalog"] = catalog
    return frame


def run_stage_zero(config: ResearchConfig | None = None, provider=None) -> dict:
    """Load data and run diagnostics without starting the statistical study."""
    config = config or ResearchConfig()
    d = load_finlab_data(provider)
    suspension_invalid = d.pop("short_suspension_invalid")
    suspension_validation = d.pop("short_suspension_validation")
    coverage = dataset_coverage(d)
    d, reconciliation = stabilize_reconciliation(d, config.reconciliation_tolerance, config.min_reconciliation_ratio)
    universe_diag, primary_symbols, universe_limitation = build_universe_diagnostics(d["margin_balance"], d["market_value"])
    universe_diag = pd.concat([universe_diag, pd.DataFrame([{"category": "UNIVERSE_LIMITATION", "count": int(universe_limitation), "symbols": "Ticker pattern is not a point-in-time security master; raw all-security sensitivity must be retained."}])], ignore_index=True)
    base, suspension_diag = _build_base_series(d, primary_symbols)
    agg_summary = []
    annual_diffs = []
    for name, individual, market in [
        ("margin_balance", base["margin_balance"], _sum_columns(d["aggregate_balance"], ["上市融資交易張數", "上櫃融資交易張數"])),
        ("short_balance", base["short_balance"], _sum_columns(d["aggregate_balance"], ["上市融券交易張數", "上櫃融券交易張數"])),
    ]:
        summary, annual = aggregate_comparison(individual, market, name)
        agg_summary.append(summary); annual_diffs.append(annual)
    reconciliation = pd.concat([reconciliation, *agg_summary], ignore_index=True, sort=False)
    return {
        "datasets": d,
        "coverage": coverage,
        "reconciliation": reconciliation,
        "universe": universe_diag,
        "primary_symbols": primary_symbols,
        "base_series": base,
        "annual_differences": annual_diffs,
        "suspension": suspension_diag,
        "suspension_validation": suspension_validation,
        "suspension_invalid_events": suspension_invalid,
    }


def run(config: ResearchConfig | None = None, provider=None, export: bool = True) -> dict:
    from .fdr import apply_fdr

    config = config or ResearchConfig()
    stage_zero = run_stage_zero(config, provider)
    d = stage_zero["datasets"]
    coverage = stage_zero["coverage"]
    reconciliation = stage_zero["reconciliation"]
    universe_diag = stage_zero["universe"]
    primary_symbols = stage_zero["primary_symbols"]
    suspension_diag = stage_zero["suspension"]
    suspension_validation = stage_zero["suspension_validation"]
    suspension_invalid = stage_zero["suspension_invalid_events"]
    features = build_features(d, primary_symbols, config)
    outcomes = build_outcomes(d["open"]["0050"], d["close"]["0050"], config.outcome_horizons)
    primary = run_primary_tests(features, outcomes)
    fdr = apply_fdr(primary) if not primary.empty else primary
    close_0050 = d["close"]["0050"]
    controlled = run_controlled_tests(features, outcomes, close_0050)
    annual_signal = annual_results(fdr, features, outcomes)
    annual = pd.concat([pd.concat(stage_zero["annual_differences"], ignore_index=True), annual_signal], ignore_index=True, sort=False)
    neighborhood = neighborhood_consistency(fdr)
    regimes = regime_results(fdr, features, outcomes, close_0050)
    robustness = pd.concat([neighborhood, regimes], ignore_index=True, sort=False)
    result = {"coverage": coverage, "reconciliation": reconciliation, "universe": universe_diag, "suspension": suspension_diag, "suspension_validation": suspension_validation, "suspension_invalid_events": suspension_invalid, "features": features, "outcomes": outcomes, "primary": primary, "fdr": fdr, "controlled": controlled, "annual": annual, "robustness": robustness}
    if export:
        run_dir, _ = create_run_directory(config.output_root, config.timezone)
        result["run_dir"] = run_dir
        coverage.to_csv(run_dir / "dataset_coverage.csv", index=False)
        reconciliation.to_csv(run_dir / "reconciliation_summary.csv", index=False)
        universe_diag.to_csv(run_dir / "universe_diagnostics.csv", index=False)
        suspension_diag.to_csv(run_dir / "suspension_diagnostics.csv", index=False)
        d["short_suspension"].to_csv(run_dir / "suspension_events.csv", index=False)
        suspension_invalid.to_csv(run_dir / "suspension_invalid_events.csv", index=False)
        suspension_validation.to_csv(run_dir / "suspension_validation_summary.csv", index=False)
        build_feature_catalog().to_csv(run_dir / "feature_catalog.csv", index=False)
        primary.to_csv(run_dir / "primary_results.csv", index=False)
        fdr.to_csv(run_dir / "fdr_results.csv", index=False)
        annual.to_csv(run_dir / "annual_results.csv", index=False)
        robustness.to_csv(run_dir / "robustness_results.csv", index=False)
        (run_dir / "signal_summary.md").write_text(signal_summary(fdr), encoding="utf-8")
        thermometer_table(fdr, coverage["latest_observation_date"].max()).to_csv(run_dir / "thermometer_signals.csv", index=False)
        controlled.to_csv(run_dir / "controlled_results.csv", index=False)
        features.join(outcomes).to_parquet(run_dir / "research_dataset.parquet")
        try: finlab_version = importlib.metadata.version("finlab")
        except importlib.metadata.PackageNotFoundError: finlab_version = "unknown"
        write_run_info(run_dir, config, coverage, finlab_version)
        (run_dir / "bias_checklist.json").write_text(json.dumps({k: "REVIEW_REQUIRED" for k in ["look-ahead bias", "survivorship bias", "data snooping", "selection bias", "transaction costs", "slippage", "liquidity", "sample size", "few-year concentration", "few-stock concentration", "suspension distortion"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
