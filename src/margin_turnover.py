from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import FINLAB_FIELDS
from .data_loader import _as_datetime_frame
from .fdr import apply_fdr, fdr_diagnostics
from .features import pr_group, rolling_ratio, trailing_percentile
from .outcomes import build_outcomes
from .price_validation import validate_split_window
from .statistics import (
    _fit_base_ols,
    annual_results,
    annual_robustness_summary,
    controlled_regression,
    neighborhood_consistency,
    run_controlled_tests,
    run_pr_bin_descriptive,
    run_primary_tests,
)
from .universe import build_universe_diagnostics


BASELINE_COMMIT = "88d9f267ddfa4b5e80cadcb1709d34ab0d9ab2ce"
FDR_SCOPE = "margin_turnover_incremental_study"
TURNOVER_VARIANTS = ("raw_lots", "volume_ratio", "amount_ratio")
REQUIRED_BASELINE_FILES = (
    "run_info.txt",
    "fdr_results.csv",
    "controlled_results.csv",
    "annual_robustness_summary.csv",
    "neighborhood_consistency.csv",
)
TURNOVER_FIELDS = (
    "open",
    "close",
    "margin_buy",
    "margin_sell",
    "margin_balance",
    "aggregate_buy",
    "aggregate_sell",
    "market_volume",
    "market_amount",
    "market_value",
)


@dataclass(frozen=True)
class MarginTurnoverConfig:
    repository: str = "taiwan-margin-short-0050-research"
    timezone: str = "Asia/Taipei"
    target_symbol: str = "0050"
    baseline_commit: str = BASELINE_COMMIT
    k_values: tuple[int, ...] = (1, 3, 5, 10)
    rolling_windows: tuple[int, ...] = (126, 252, 504, 756)
    outcome_horizons: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    pr_edges: tuple[int, ...] = (0, 5, 20, 40, 60, 80, 95, 100)
    min_group_n: int = 20
    controlled_leverage_tolerance: float = 1e-10
    controlled_max_condition_number: float = 1e12
    robustness_min_years: int = 3
    robustness_min_direction_ratio: float = 0.60
    robustness_min_neighborhood_score: float = 0.60
    robustness_max_year_sample_share: float = 0.50
    absorption_partial_threshold: float = 0.30
    absorption_large_threshold: float = 0.70
    absorption_survival_retention: float = 0.70
    absorption_significance_threshold: float = 0.05
    absorption_beta_epsilon: float = 1e-12
    output_root: Path = Path("outputs_turnover")


def _git_value(args: list[str], default: str = "unknown") -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return default


def _read_run_info(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def load_frozen_baseline(
    baseline_run_dir: str | Path,
    expected_commit: str = BASELINE_COMMIT,
) -> dict[str, object]:
    """Validate and load the immutable baseline artifacts used by this study."""
    root = Path(baseline_run_dir).expanduser()
    missing = [name for name in REQUIRED_BASELINE_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Baseline output is missing required files: {missing}")
    run_info = _read_run_info(root / "run_info.txt")
    actual_commit = run_info.get("git_commit")
    if actual_commit != expected_commit:
        raise ValueError(
            f"Baseline commit mismatch: expected {expected_commit}, got {actual_commit!r}; "
            "stop and confirm the baseline run directory."
        )
    return {
        "run_dir": root,
        "run_info": run_info,
        "fdr_results": pd.read_csv(root / "fdr_results.csv"),
        "controlled_results": pd.read_csv(root / "controlled_results.csv"),
        "annual_robustness_summary": pd.read_csv(root / "annual_robustness_summary.csv"),
        "neighborhood_consistency": pd.read_csv(root / "neighborhood_consistency.csv"),
    }


def load_turnover_data(provider=None) -> dict[str, pd.DataFrame]:
    """Load only datasets needed by the incremental turnover experiment."""
    if provider is None:
        from finlab import data as provider
    loaded = {}
    for alias in TURNOVER_FIELDS:
        field = FINLAB_FIELDS[alias]
        try:
            loaded[alias] = _as_datetime_frame(provider.get(field), field)
        except Exception as exc:
            raise RuntimeError(f"Failed loading turnover dataset {field!r}: {exc}") from exc
    for key in ("open", "close"):
        if "0050" not in loaded[key].columns.astype(str):
            raise ValueError(f"0050 missing from {FINLAB_FIELDS[key]}; stopping")
    for key in ("market_volume", "market_amount"):
        missing = {"TAIEX", "OTC"} - set(loaded[key].columns)
        if missing:
            raise ValueError(f"{FINLAB_FIELDS[key]} missing columns: {sorted(missing)}")
    amount_columns = {"上市融資交易金額", "上櫃融資交易金額"}
    for key in ("aggregate_buy", "aggregate_sell"):
        missing = amount_columns - set(loaded[key].columns)
        if missing:
            raise ValueError(f"{FINLAB_FIELDS[key]} missing columns: {sorted(missing)}")
    return loaded


def _sum_columns(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return frame[list(names)].sum(axis=1, min_count=1)


def _sum_primary(frame: pd.DataFrame, primary_symbols: set[str]) -> pd.Series:
    columns = [column for column in frame.columns if str(column) in primary_symbols]
    if not columns:
        raise ValueError("No primary-universe symbols are available for margin turnover")
    return frame[columns].sum(axis=1, min_count=1)


def _turnover_base_series(
    datasets: dict[str, pd.DataFrame], primary_symbols: set[str]
) -> dict[str, pd.Series]:
    buy_lots = _sum_primary(datasets["margin_buy"], primary_symbols)
    sell_lots = _sum_primary(datasets["margin_sell"], primary_symbols)
    turnover_lots = buy_lots.add(sell_lots, fill_value=np.nan)
    market_volume = _sum_columns(datasets["market_volume"], ("TAIEX", "OTC"))
    market_amount = _sum_columns(datasets["market_amount"], ("TAIEX", "OTC"))
    buy_amount = _sum_columns(
        datasets["aggregate_buy"], ("上市融資交易金額", "上櫃融資交易金額")
    )
    sell_amount = _sum_columns(
        datasets["aggregate_sell"], ("上市融資交易金額", "上櫃融資交易金額")
    )
    return {
        "turnover_lots": turnover_lots,
        "turnover_shares": turnover_lots * 1000.0,
        "turnover_amount": buy_amount.add(sell_amount, fill_value=np.nan),
        "market_volume": market_volume,
        "market_amount": market_amount,
    }


def build_margin_turnover_features(
    datasets: dict[str, pd.DataFrame],
    primary_symbols: set[str],
    config: MarginTurnoverConfig | None = None,
) -> pd.DataFrame:
    """Build three normalizations of one economic variable without future fill."""
    config = config or MarginTurnoverConfig()
    base = _turnover_base_series(datasets, primary_symbols)
    values: dict[str, pd.Series] = {}
    catalog: dict[str, dict] = {}
    for k in config.k_values:
        variant_values = margin_turnover_variant_series(base, k)
        for variant, series in variant_values.items():
            for window in config.rolling_windows:
                name = f"margin_turnover__{variant}__k{k}__w{window}_percentile"
                values[name] = trailing_percentile(series, window)
                catalog[name] = {
                    "family": "margin_turnover",
                    "variant": variant,
                    "k": k,
                    "window": window,
                    "primary_normalized": variant == "amount_ratio",
                }
    result = pd.DataFrame(values).sort_index()
    result.attrs["catalog"] = catalog
    return result


def margin_turnover_variant_series(
    base: dict[str, pd.Series], k: int
) -> dict[str, pd.Series]:
    """Return k-day raw and ratio-of-sums variants before percentile ranking."""
    return {
        "raw_lots": base["turnover_lots"].rolling(k, min_periods=k).sum(),
        "volume_ratio": rolling_ratio(base["turnover_shares"], base["market_volume"], k),
        "amount_ratio": rolling_ratio(base["turnover_amount"], base["market_amount"], k),
    }


def turnover_feature_catalog(features: pd.DataFrame) -> pd.DataFrame:
    rows = [{"predictor": name, **meta} for name, meta in features.attrs.get("catalog", {}).items()]
    return pd.DataFrame(rows)


def _attach_variant(results: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    variants = {name: meta.get("variant") for name, meta in features.attrs.get("catalog", {}).items()}
    out["variant"] = (
        out["predictor"].map(variants)
        if "predictor" in out
        else pd.Series(index=out.index, dtype="object")
    )
    return out


def run_turnover_primary(
    features: pd.DataFrame, outcomes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = _attach_variant(run_primary_tests(features, outcomes), features)
    fdr = apply_fdr(primary) if not primary.empty else primary.copy()
    fdr["fdr_scope"] = FDR_SCOPE
    diagnostics = fdr_diagnostics(fdr)
    diagnostics["fdr_scope"] = FDR_SCOPE
    return primary, fdr, diagnostics


def turnover_neighborhood_consistency(results: pd.DataFrame) -> pd.DataFrame:
    """Reuse baseline neighborhood logic while keeping normalizations separate."""
    if results.empty:
        return pd.DataFrame(columns=[
            "family", "variant", "pr_group", "consistency_type", "fixed_parameter",
            "effect_sign", "n_cells", "positive_cells", "negative_cells",
            "consistency_score", "note",
        ])
    rows = []
    for variant, group in results.groupby("variant", dropna=False):
        incremental = group.copy()
        incremental["family"] = "margin_turnover"
        summary = neighborhood_consistency(incremental)
        summary.insert(1, "variant", variant)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def select_retained_margin_signals(fdr_results: pd.DataFrame) -> pd.DataFrame:
    """Select only frozen retained buy/sell rows; never mutate the input frame."""
    out = fdr_results.copy(deep=True)
    family = out.get("family", pd.Series(index=out.index, dtype="object"))
    mask = family.isin(["margin_buy", "margin_sell"])
    if "research_decision" in out:
        mask &= out["research_decision"].eq("保留")
    else:
        required = {"evidence_level", "robustness_status", "live_eligibility"}
        if not required.issubset(out.columns):
            raise ValueError(f"Baseline FDR output lacks retained-signal fields: {sorted(required - set(out.columns))}")
        live = out["live_eligibility"].astype("boolean").fillna(False)
        mask &= out["evidence_level"].eq("Level A") & out["robustness_status"].eq("robust") & live
    return out.loc[mask].copy()


def _baseline_original_features(
    datasets: dict[str, pd.DataFrame], candidates: pd.DataFrame
) -> pd.DataFrame:
    """Reconstruct only retained baseline buy/sell predictors, using baseline formulas."""
    values: dict[str, pd.Series] = {}
    market_volume = _sum_columns(datasets["market_volume"], ("TAIEX", "OTC"))
    market_amount = _sum_columns(datasets["market_amount"], ("TAIEX", "OTC"))
    for row in candidates.drop_duplicates("predictor").itertuples():
        family = row.family
        k = int(row.k)
        window = int(row.rolling_window)
        if "__amount_ratio__" in row.predictor:
            aggregate = datasets["aggregate_buy" if family == "margin_buy" else "aggregate_sell"]
            amount = _sum_columns(aggregate, ("上市融資交易金額", "上櫃融資交易金額"))
            base = rolling_ratio(amount, market_amount, k)
        elif "__volume_ratio__" in row.predictor:
            lots = datasets[family].sum(axis=1, min_count=1)
            base = rolling_ratio(lots * 1000.0, market_volume, k)
        elif "__raw__" in row.predictor:
            base = datasets[family].sum(axis=1, min_count=1).rolling(k, min_periods=k).sum()
        else:
            raise ValueError(f"Unsupported retained baseline predictor: {row.predictor}")
        values[row.predictor] = trailing_percentile(base, window)
    return pd.DataFrame(values).sort_index()


def _after_turnover_regression(
    signal: pd.Series,
    turnover: pd.Series,
    future: pd.Series,
    prior: pd.Series,
    config: MarginTurnoverConfig,
) -> dict:
    frame = pd.concat(
        {"future": future, "signal": signal, "turnover": turnover, "prior": prior}, axis=1
    ).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(frame)
    signal_n = int(frame["signal"].eq(1).sum())
    control_n = int(frame["signal"].eq(0).sum())
    base = {
        "N": n,
        "beta_after_turnover": np.nan,
        "p_after_turnover": np.nan,
        "turnover_beta": np.nan,
        "turnover_p_value": np.nan,
        "condition_number": np.nan,
        "max_leverage": np.nan,
    }
    if n < config.min_group_n:
        return base | {"status": "insufficient_sample"}
    if frame["signal"].nunique() < 2 or signal_n < config.min_group_n or control_n < config.min_group_n:
        return base | {"status": "insufficient_group_sample"}
    if frame["prior"].nunique() < 2 or frame["turnover"].nunique() < 2:
        return base | {"status": "no_control_variation"}
    design = pd.DataFrame(
        {
            "const": 1.0,
            "signal": frame["signal"].astype(float),
            "turnover": frame["turnover"].astype(float),
            "prior": frame["prior"].astype(float),
        },
        index=frame.index,
    )
    matrix = design.to_numpy()
    rank = int(np.linalg.matrix_rank(matrix))
    condition = float(np.linalg.cond(matrix))
    base["condition_number"] = condition
    if rank < design.shape[1] or not np.isfinite(condition) or condition >= config.controlled_max_condition_number:
        return base | {"status": "rank_deficient"}
    fit = _fit_base_ols(frame["future"], design)
    leverage = np.asarray(fit.get_influence().hat_matrix_diag, dtype=float)
    finite_leverage = np.isfinite(leverage)
    max_leverage = float(np.max(leverage)) if finite_leverage.all() and len(leverage) else np.nan
    base["max_leverage"] = max_leverage
    if not finite_leverage.all() or np.any(leverage >= 1.0 - config.controlled_leverage_tolerance):
        return base | {"status": "high_leverage_unstable"}
    robust = fit.get_robustcov_results(cov_type="HC3")
    beta_after = float(robust.params[1])
    turnover_beta = float(robust.params[2])
    p_after = float(robust.pvalues[1])
    turnover_p = float(robust.pvalues[2])
    covariance = np.asarray(robust.cov_params(), dtype=float)
    if not np.isfinite([beta_after, turnover_beta, p_after, turnover_p]).all() or not np.isfinite(covariance).all():
        return base | {"status": "hc3_nonfinite"}
    return base | {
        "beta_after_turnover": beta_after,
        "p_after_turnover": p_after,
        "turnover_beta": turnover_beta,
        "turnover_p_value": turnover_p,
        "status": "estimated",
    }


def _absorption_classification(
    before: dict, after: dict, attenuation: float, config: MarginTurnoverConfig
) -> str:
    statuses = {before.get("status"), after.get("status")}
    if statuses & {"insufficient_sample", "insufficient_group_sample", "no_signal_variation", "no_prior_variation", "no_control_variation"}:
        return "insufficient_sample"
    if before.get("status") != "estimated" or after.get("status") != "estimated":
        return "unstable_collinearity"
    beta_before = before["signal_beta"]
    beta_after = after["beta_after_turnover"]
    if np.sign(beta_before) != np.sign(beta_after):
        return "sign_reversal"
    if not np.isfinite(attenuation):
        return "undefined"
    retained = abs(beta_after) / abs(beta_before)
    if retained >= config.absorption_survival_retention and after["p_after_turnover"] < config.absorption_significance_threshold:
        return "survives_turnover_control"
    if config.absorption_partial_threshold <= attenuation <= config.absorption_large_threshold:
        return "partially_absorbed"
    if attenuation > config.absorption_large_threshold or after["p_after_turnover"] >= config.absorption_significance_threshold:
        return "largely_absorbed"
    return "survives_turnover_control"


def run_turnover_absorption(
    baseline_fdr: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    turnover_features: pd.DataFrame,
    outcomes: pd.DataFrame,
    close_0050: pd.Series,
    config: MarginTurnoverConfig | None = None,
) -> pd.DataFrame:
    config = config or MarginTurnoverConfig()
    retained = select_retained_margin_signals(baseline_fdr)
    if retained.empty:
        return pd.DataFrame()
    originals = _baseline_original_features(datasets, retained)
    joined = originals.join(turnover_features, how="outer").join(outcomes, how="inner")
    priors = {k: close_0050.pct_change(k, fill_method=None).reindex(joined.index) for k in config.k_values}
    rows = []
    for row in retained.itertuples():
        tail = (
            joined[row.predictor].le(5)
            if row.pr_group == "PR0-5"
            else joined[row.predictor].ge(95)
        ).astype(float).where(joined[row.predictor].notna())
        for variant in ("amount_ratio", "volume_ratio"):
            control_name = f"margin_turnover__{variant}__k{int(row.k)}__w{int(row.rolling_window)}_percentile"
            turnover_control = joined[control_name] / 100.0
            for prior_k, prior in priors.items():
                common = turnover_control.notna()
                model_signal = tail.where(common)
                model_future = joined[row.outcome_horizon].where(common)
                model_prior = prior.where(common)
                before = controlled_regression(
                    model_signal,
                    model_future,
                    model_prior,
                    min_total_n=config.min_group_n,
                    min_tail_n=config.min_group_n,
                    min_control_n=config.min_group_n,
                    leverage_tolerance=config.controlled_leverage_tolerance,
                    max_condition_number=config.controlled_max_condition_number,
                )
                after = _after_turnover_regression(
                    model_signal, turnover_control, model_future, model_prior, config
                )
                beta_before = before.get("signal_beta", np.nan)
                beta_after = after.get("beta_after_turnover", np.nan)
                attenuation = (
                    1.0 - abs(beta_after) / abs(beta_before)
                    if np.isfinite(beta_before)
                    and np.isfinite(beta_after)
                    and abs(beta_before) > config.absorption_beta_epsilon
                    else np.nan
                )
                paired = pd.concat(
                    {
                        "signal": model_signal,
                        "turnover": turnover_control,
                        "future": model_future,
                        "prior": model_prior,
                    },
                    axis=1,
                ).replace([np.inf, -np.inf], np.nan).dropna()
                correlation = (
                    paired["signal"].corr(paired["turnover"])
                    if len(paired) > 1
                    and paired["signal"].nunique() > 1
                    and paired["turnover"].nunique() > 1
                    else np.nan
                )
                sign_before = float(np.sign(beta_before)) if np.isfinite(beta_before) else np.nan
                sign_after = float(np.sign(beta_after)) if np.isfinite(beta_after) else np.nan
                classification = _absorption_classification(before, after, attenuation, config)
                rows.append({
                    "original_predictor": row.predictor,
                    "original_family": row.family,
                    "original_pr_group": row.pr_group,
                    "k": int(row.k),
                    "rolling_window": int(row.rolling_window),
                    "outcome_horizon": row.outcome_horizon,
                    "prior_k": prior_k,
                    "turnover_variant": variant,
                    "N": after["N"],
                    "baseline_effect_vs_unconditional": getattr(row, "effect_vs_unconditional", np.nan),
                    "baseline_N": getattr(row, "N", np.nan),
                    "baseline_win_rate": getattr(row, "win_rate", np.nan),
                    "baseline_evidence": getattr(row, "evidence_level", np.nan),
                    "baseline_research_decision": getattr(row, "research_decision", np.nan),
                    "beta_before_turnover": beta_before,
                    "p_before_turnover": before.get("signal_p_value", np.nan),
                    "beta_after_turnover": beta_after,
                    "p_after_turnover": after.get("p_after_turnover", np.nan),
                    "turnover_beta": after.get("turnover_beta", np.nan),
                    "turnover_p_value": after.get("turnover_p_value", np.nan),
                    "beta_change": beta_after - beta_before if np.isfinite(beta_after) and np.isfinite(beta_before) else np.nan,
                    "attenuation_ratio": attenuation,
                    "sign_before": sign_before,
                    "sign_after": sign_after,
                    "sign_flip": bool(np.isfinite(sign_before) and np.isfinite(sign_after) and sign_before != sign_after),
                    "correlation_signal_turnover": correlation,
                    "condition_number": after.get("condition_number", np.nan),
                    "status": after.get("status"),
                    "before_status": before.get("status"),
                    "absorption_classification": classification,
                })
    return pd.DataFrame(rows)


def _create_run_directory(config: MarginTurnoverConfig) -> Path:
    commit = _git_value(["rev-parse", "--short", "HEAD"])
    stamp = datetime.now(ZoneInfo(config.timezone)).strftime("%Y%m%d_%H%M%S")
    path = config.output_root / f"{stamp}_{commit}_margin_turnover_adjusted"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_turnover_run_info(
    run_dir: Path,
    baseline_run_dir: Path,
    config: MarginTurnoverConfig,
) -> None:
    try:
        finlab_version = importlib.metadata.version("finlab")
    except importlib.metadata.PackageNotFoundError:
        finlab_version = "unknown"
    lines = {
        "repository": config.repository,
        "branch": _git_value(["branch", "--show-current"]),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "baseline_commit": config.baseline_commit,
        "baseline_run_dir": str(baseline_run_dir),
        "run_timestamp": datetime.now(ZoneInfo(config.timezone)).isoformat(),
        "timezone": config.timezone,
        "python_version": platform.python_version(),
        "finlab_version": finlab_version,
        "price_source_open": "etl:adj_open",
        "price_source_close": "etl:adj_close",
        "outcome_price_adjusted": True,
        "corporate_action_fix": "0050_split_2025_06",
        "previous_turnover_commit": "08b519686c42a209374d98fe84d0c36c634324f7",
        "previous_turnover_output": "/content/drive/MyDrive/Quant_Research/taiwan-margin-short-0050-research/margin_turnover/20260912_071957_08b5196_margin_turnover_incremental",
        "study": "margin_turnover_incremental",
        "k_values": config.k_values,
        "rolling_windows": config.rolling_windows,
        "pr_groups": config.pr_edges,
        "outcome_horizons": config.outcome_horizons,
        "turnover_variants": TURNOVER_VARIANTS,
        "fdr_scope": FDR_SCOPE,
        "baseline_full_research_rerun": False,
        "absorption_partial_threshold": config.absorption_partial_threshold,
        "absorption_large_threshold": config.absorption_large_threshold,
        "absorption_survival_retention": config.absorption_survival_retention,
        "absorption_significance_threshold": config.absorption_significance_threshold,
    }
    (run_dir / "run_info_turnover.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in lines.items()) + "\n", encoding="utf-8"
    )


def classify_turnover_dose_response(pr_bins: pd.DataFrame) -> pd.DataFrame:
    """Descriptively classify the seven fixed bins; this is not a hypothesis test."""
    columns = ["predictor", "variant", "k", "rolling_window", "outcome_horizon", "shape"]
    if pr_bins.empty:
        return pd.DataFrame(columns=columns)
    order = ["PR0-5", "PR5-20", "PR20-40", "PR40-60", "PR60-80", "PR80-95", "PR95-100"]
    rows = []
    keys = ["predictor", "variant", "k", "rolling_window", "outcome_horizon"]
    for key, group in pr_bins.groupby(keys, dropna=False):
        ordered = group.set_index("pr_group").reindex(order)
        means = ordered["mean_return"]
        effects = ordered["effect_vs_unconditional"]
        if means.isna().any():
            shape = "insufficient_data"
        else:
            delta = np.diff(means.to_numpy())
            minimum = int(np.argmin(means.to_numpy()))
            maximum = int(np.argmax(means.to_numpy()))
            if np.all(delta >= 0):
                shape = "monotonic_positive"
            elif np.all(delta <= 0):
                shape = "monotonic_negative"
            elif 0 < minimum < 6 and np.all(delta[:minimum] <= 0) and np.all(delta[minimum:] >= 0):
                shape = "U_shape"
            elif 0 < maximum < 6 and np.all(delta[:maximum] >= 0) and np.all(delta[maximum:] <= 0):
                shape = "inverted_U"
            elif effects.notna().all() and int(np.argmax(np.abs(effects.to_numpy()))) == 6:
                shape = "high_tail_extreme_only"
            elif effects.notna().all() and int(np.argmax(np.abs(effects.to_numpy()))) == 0:
                shape = "low_tail_extreme_only"
            else:
                shape = "mixed"
        rows.append(dict(zip(keys, key)) | {"shape": shape})
    return pd.DataFrame(rows, columns=columns)


def _md_value(value, percent=False) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.2%}" if percent else str(value)


def _turnover_summary(
    fdr: pd.DataFrame,
    pr_bins: pd.DataFrame,
    annual_summary: pd.DataFrame,
    neighborhood: pd.DataFrame,
    absorption: pd.DataFrame,
    config: MarginTurnoverConfig,
) -> str:
    evidence = fdr[fdr.get("evidence_level", pd.Series(index=fdr.index, dtype="object")).ne("No Evidence")].copy()
    annual_keys = ["predictor", "k", "rolling_window", "pr_group", "outcome_horizon"]
    if not evidence.empty and not annual_summary.empty:
        evidence = evidence.merge(
            annual_summary[annual_keys + ["years_with_samples", "positive_year_ratio", "largest_year_sample_share", "few_year_concentration_flag"]],
            on=annual_keys, how="left", suffixes=("", "_annual"),
        )
    fixed = neighborhood[neighborhood.get("consistency_type", pd.Series(index=neighborhood.index, dtype="object")).eq("fixed_horizon")]
    if not evidence.empty and not fixed.empty:
        fixed = fixed.rename(columns={"fixed_parameter": "outcome_horizon", "consistency_score": "neighborhood_score"})
        evidence = evidence.merge(
            fixed[["family", "variant", "pr_group", "outcome_horizon", "neighborhood_score"]],
            on=["family", "variant", "pr_group", "outcome_horizon"], how="left",
        )
    lines = [
        "# Margin Turnover Incremental Study",
        "",
        "本摘要只屬於 margin-turnover incremental experiment；Level A 不是與 baseline 4,884 tests 合併後的 global FDR。",
        "",
        "## 1. Margin Turnover 本身",
        "",
        "| Horizon | Variant | k | Rolling | Tail | N | Mean | Median | Win rate | Effect vs unconditional | Evidence | Annual | Neighborhood |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    if evidence.empty:
        lines.append("| - | - | - | - | - | 0 | NA | NA | NA | NA | No Evidence | insufficient | insufficient |")
    else:
        for row in evidence.sort_values(["outcome_horizon", "variant", "k", "rolling_window", "pr_group"]).itertuples():
            years = getattr(row, "years_with_samples", np.nan)
            concentration = getattr(row, "few_year_concentration_flag", True)
            positive_ratio = getattr(row, "positive_year_ratio", np.nan)
            pooled_positive = row.effect_vs_unconditional >= 0
            direction_ratio = positive_ratio if pooled_positive else 1 - positive_ratio
            annual_ok = pd.notna(years) and years >= config.robustness_min_years and pd.notna(direction_ratio) and direction_ratio >= config.robustness_min_direction_ratio and not bool(concentration)
            neighborhood_score = getattr(row, "neighborhood_score", np.nan)
            lines.append(
                f"| {row.outcome_horizon} | {row.variant} | {row.k} | {row.rolling_window} | {row.pr_group} | {row.N} | "
                f"{_md_value(row.mean_return, True)} | {_md_value(row.median_return, True)} | {_md_value(row.win_rate, True)} | "
                f"{_md_value(row.effect_vs_unconditional, True)} | {row.evidence_level} | {'robust' if annual_ok else 'insufficient/mixed'} | "
                f"{_md_value(neighborhood_score)} |"
            )
    shapes = classify_turnover_dose_response(pr_bins)
    lines.extend([
        "",
        "## 2. Dose-response",
        "",
        "| Predictor | Variant | k | Rolling | Horizon | Shape |",
        "|---|---|---:|---:|---|---|",
    ])
    if shapes.empty:
        lines.append("| - | - | - | - | - | insufficient_data |")
    else:
        for row in shapes.itertuples():
            lines.append(f"| {row.predictor} | {row.variant} | {row.k} | {row.rolling_window} | {row.outcome_horizon} | {row.shape} |")
    lines.extend([
        "",
        "## 3. Absorption",
        "",
        "| Original signal | Before turnover beta | After turnover beta | Attenuation | p before | p after | Conclusion |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    if absorption.empty:
        lines.append("| - | NA | NA | NA | NA | NA | Frozen baseline 沒有 retained margin buy/sell signal |")
    else:
        primary_absorption = absorption[absorption["turnover_variant"].eq("amount_ratio")]
        keys = ["original_predictor", "original_pr_group", "outcome_horizon"]
        for key, group in primary_absorption.groupby(keys, dropna=False):
            valid = group[group["status"].eq("estimated")]
            use = valid if not valid.empty else group
            conclusion = use["absorption_classification"].mode().iloc[0] if not use.empty else "insufficient_sample"
            lines.append(
                f"| {key[0]} / {key[1]} / {key[2]} | {_md_value(use['beta_before_turnover'].median())} | "
                f"{_md_value(use['beta_after_turnover'].median())} | {_md_value(use['attenuation_ratio'].median(), True)} | "
                f"{_md_value(use['p_before_turnover'].median())} | {_md_value(use['p_after_turnover'].median())} | {conclusion} |"
            )

    middle = {"O1_C5", "O1_C10", "O1_C20"}
    amount_strong = (
        evidence[
            evidence["variant"].eq("amount_ratio")
            & evidence["pr_group"].eq("PR95-100")
            & evidence["outcome_horizon"].isin(middle)
            & evidence["evidence_level"].isin(["Level A", "Level B"])
        ]
        if {"variant", "pr_group", "outcome_horizon", "evidence_level"}.issubset(evidence.columns)
        else evidence.iloc[0:0]
    )
    absorption_signal = absorption[
        absorption.get("turnover_variant", pd.Series(index=absorption.index, dtype="object")).eq("amount_ratio")
        & absorption.get("status", pd.Series(index=absorption.index, dtype="object")).eq("estimated")
    ]
    absorbed_families = set()
    surviving_families = set()
    absorption_groups = (
        absorption_signal.groupby("original_family")
        if "original_family" in absorption_signal
        else []
    )
    for family, group in absorption_groups:
        counts = group["absorption_classification"].value_counts(normalize=True)
        if counts.get("largely_absorbed", 0) >= config.robustness_min_direction_ratio:
            absorbed_families.add(family)
        if counts.get("survives_turnover_control", 0) >= config.robustness_min_direction_ratio:
            surviving_families.add(family)
    if not amount_strong.empty and absorbed_families == {"margin_buy", "margin_sell"}:
        hypothesis = "A. 支持 credit-activity latent factor"
    elif not amount_strong.empty and absorbed_families:
        hypothesis = "B. 部分支持"
    elif amount_strong.empty and surviving_families == {"margin_buy", "margin_sell"}:
        hypothesis = "C. 不支持"
    else:
        hypothesis = "D. 無法判定"
    if not amount_strong.empty:
        family_decision = "修改後再測"
    else:
        family_decision = "無法判定"
    lines.extend([
        "",
        "## 4. 最終假設判斷",
        "",
        hypothesis,
        "",
        "## 5. Family 決策",
        "",
        f"Margin Turnover：{family_decision}。正式保留仍需 amount ratio、年度與 neighborhood robustness 一致，不能由單一 pooled cell 決定。",
        "",
        "## 6. 反對者觀點",
        "",
        "- turnover 可能只是總市場成交熱度或 bull-market consequence。",
        "- overlapping C10/C20 outcomes 不代表獨立證據。",
        "- W756/PR95 可能集中少數年份。",
        "- amount/volume normalization 可能方向衝突。",
        "- margin buy/sell 與 turnover 可能共線。",
        "- 本次新增研究另有 data-snooping 風險。",
    ])
    return "\n".join(lines) + "\n"


def run_margin_turnover_study(
    baseline_run_dir: str | Path,
    config: MarginTurnoverConfig | None = None,
    provider=None,
    export: bool = True,
    datasets: dict[str, pd.DataFrame] | None = None,
) -> dict[str, object]:
    """Run only the incremental turnover experiment; never call pipeline.run()."""
    config = config or MarginTurnoverConfig()
    baseline = load_frozen_baseline(baseline_run_dir, config.baseline_commit)
    data = load_turnover_data(provider) if datasets is None else datasets
    universe, primary_symbols, universe_limitation = build_universe_diagnostics(
        data["margin_balance"], data["market_value"]
    )
    features = build_margin_turnover_features(data, primary_symbols, config)
    outcomes = build_outcomes(
        data["open"][config.target_symbol],
        data["close"][config.target_symbol],
        config.outcome_horizons,
    )
    price_diagnostics = validate_split_window(
        data["open"][config.target_symbol], data["close"][config.target_symbol], outcomes
    )
    primary, fdr, fdr_diag = run_turnover_primary(features, outcomes)
    pr_bins = _attach_variant(run_pr_bin_descriptive(features, outcomes), features)
    controlled = _attach_variant(
        run_controlled_tests(
            features,
            outcomes,
            data["close"][config.target_symbol],
            min_total_n=config.min_group_n,
            min_tail_n=config.min_group_n,
            min_control_n=config.min_group_n,
            leverage_tolerance=config.controlled_leverage_tolerance,
            max_condition_number=config.controlled_max_condition_number,
        ),
        features,
    )
    annual = _attach_variant(annual_results(fdr, features, outcomes), features)
    annual_summary = _attach_variant(annual_robustness_summary(
        annual,
        min_years=config.robustness_min_years,
        max_year_sample_share=config.robustness_max_year_sample_share,
    ), features)
    neighborhood = turnover_neighborhood_consistency(fdr)
    absorption = run_turnover_absorption(
        baseline["fdr_results"], data, features, outcomes,
        data["close"][config.target_symbol], config,
    )
    result: dict[str, object] = {
        "baseline": baseline,
        "universe": universe,
        "UNIVERSE_LIMITATION": universe_limitation,
        "features": features,
        "feature_catalog": turnover_feature_catalog(features),
        "outcomes": outcomes,
        "outcome_price_diagnostics": price_diagnostics,
        "primary": primary,
        "fdr": fdr,
        "fdr_diagnostics": fdr_diag,
        "pr_bins": pr_bins,
        "controlled": controlled,
        "absorption": absorption,
        "annual": annual,
        "annual_robustness_summary": annual_summary,
        "neighborhood": neighborhood,
    }
    if export:
        run_dir = _create_run_directory(config)
        result["run_dir"] = run_dir
        outputs = {
            "turnover_feature_catalog.csv": result["feature_catalog"],
            "turnover_primary_results.csv": primary,
            "turnover_fdr_results.csv": fdr,
            "turnover_fdr_diagnostics.csv": fdr_diag,
            "turnover_pr_bin_results.csv": pr_bins,
            "turnover_controlled_results.csv": controlled,
            "turnover_absorption_results.csv": absorption,
            "turnover_annual_results.csv": annual,
            "turnover_annual_robustness_summary.csv": annual_summary,
            "turnover_neighborhood_consistency.csv": neighborhood,
            "outcome_price_diagnostics.csv": price_diagnostics,
        }
        for filename, frame in outputs.items():
            frame.to_csv(run_dir / filename, index=False)
        (run_dir / "turnover_summary.md").write_text(
            _turnover_summary(fdr, pr_bins, annual_summary, neighborhood, absorption, config), encoding="utf-8"
        )
        _write_turnover_run_info(run_dir, Path(baseline_run_dir), config)
    return result
