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
from scipy import stats

from .margin_composition import TURNOVER_BASE_COMMIT, load_frozen_turnover
from .margin_turnover import (
    BASELINE_COMMIT,
    _baseline_original_features,
    _read_run_info,
    _turnover_base_series,
    load_frozen_baseline,
    load_turnover_data,
    margin_turnover_variant_series,
)
from .outcomes import build_outcomes
from .price_validation import validate_split_window
from .universe import build_universe_diagnostics


COMPOSITION_COMMIT = "cf16e4aa7056c4c2c54e1d6248312bee6acac9da"
FDR_SCOPE = "margin_regime_interaction_incremental_study"
REQUIRED_COMPOSITION_FILES = (
    "run_info_composition.txt",
    "composition_primary_results.csv",
    "composition_fdr_results.csv",
    "composition_annual_results.csv",
    "composition_annual_robustness_summary.csv",
    "composition_absorption_results.csv",
)
OUTCOME_COLUMNS = tuple(f"O1_C{h}" for h in (1, 2, 3, 5, 10, 20))


@dataclass(frozen=True)
class MarginRegimeInteractionConfig:
    repository: str = "taiwan-margin-short-0050-research"
    timezone: str = "Asia/Taipei"
    target_symbol: str = "0050"
    baseline_commit: str = BASELINE_COMMIT
    turnover_commit: str = TURNOVER_BASE_COMMIT
    composition_commit: str = COMPOSITION_COMMIT
    outcome_horizons: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    prior_return_days: int = 5
    primary_tail: str = "PR95-100"
    min_group_n: int = 20
    leverage_tolerance: float = 1e-10
    max_condition_number: float = 1e12
    robustness_min_years: int = 3
    robustness_min_direction_ratio: float = 0.60
    robustness_max_year_sample_share: float = 0.50
    output_root: Path = Path("outputs_regime_interaction")


def _git_value(args: list[str], default: str = "unknown") -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return default


def load_frozen_composition(
    composition_run_dir: str | Path,
    expected_commit: str = COMPOSITION_COMMIT,
    expected_turnover: str = TURNOVER_BASE_COMMIT,
    expected_baseline: str = BASELINE_COMMIT,
) -> dict[str, object]:
    """Validate the third immutable link in the adjusted-price research chain."""
    root = Path(composition_run_dir).expanduser()
    missing = [name for name in REQUIRED_COMPOSITION_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Composition output is missing required files: {missing}")
    info = _read_run_info(root / "run_info_composition.txt")
    checks = {
        "git_commit": expected_commit,
        "turnover_base_commit": expected_turnover,
        "baseline_commit": expected_baseline,
        "price_source_open": "etl:adj_open",
        "price_source_close": "etl:adj_close",
    }
    for key, expected in checks.items():
        if info.get(key) != expected:
            raise ValueError(
                f"Composition {key} mismatch: expected {expected}, got {info.get(key)!r}"
            )
    result: dict[str, object] = {"run_dir": root, "run_info": info}
    for filename in REQUIRED_COMPOSITION_FILES[1:]:
        result[filename.removesuffix(".csv")] = pd.read_csv(root / filename)
    return result


def prior_5d_return(close_0050: pd.Series) -> pd.Series:
    """Return C[t]/C[t-5]-1; only the signal date and its past are used."""
    close = pd.to_numeric(close_0050, errors="coerce").sort_index()
    return close.pct_change(5, fill_method=None).rename("prior_5d_return")


def binary_regime(prior: pd.Series) -> pd.Series:
    regime = pd.Series(pd.NA, index=prior.index, dtype="string", name="regime")
    valid = prior.notna()
    regime.loc[valid & prior.gt(0)] = "Up"
    regime.loc[valid & prior.le(0)] = "Down"
    return regime


def select_primary_signal_specs(
    baseline_fdr: pd.DataFrame,
    primary_tail: str = "PR95-100",
) -> pd.DataFrame:
    """Use all pre-existing high amount-ratio Buy/Sell definitions, without outcome selection."""
    columns = ["predictor", "family", "k", "rolling_window", "pr_group"]
    required = set(columns)
    if not required.issubset(baseline_fdr.columns):
        raise ValueError(f"Baseline FDR output lacks signal fields: {sorted(required - set(baseline_fdr))}")
    mask = (
        baseline_fdr["family"].isin(["margin_buy", "margin_sell"])
        & baseline_fdr["predictor"].astype(str).str.contains("__amount_ratio__", regex=False)
        & baseline_fdr["pr_group"].eq(primary_tail)
    )
    specs = baseline_fdr.loc[mask, columns].drop_duplicates().copy()
    specs["k"] = specs["k"].astype(int)
    specs["rolling_window"] = specs["rolling_window"].astype(int)
    return specs.sort_values(["family", "predictor"]).reset_index(drop=True)


def build_signal_frame(
    datasets: dict[str, pd.DataFrame], specs: pd.DataFrame
) -> pd.DataFrame:
    percentiles = _baseline_original_features(datasets, specs)
    signals = pd.DataFrame(index=percentiles.index)
    for row in specs.itertuples():
        values = percentiles[row.predictor]
        if row.pr_group == "PR95-100":
            selected = values.ge(95)
        elif row.pr_group == "PR0-5":
            selected = values.le(5)
        else:
            raise ValueError(f"Unsupported inferential tail: {row.pr_group}")
        signals[row.predictor] = selected.astype(float).where(values.notna())
    signals.attrs["specs"] = specs.copy()
    signals.attrs["percentiles"] = percentiles
    return signals


def regime_signal_distribution(
    signals: pd.DataFrame, regime: pd.Series, specs: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for row in specs.itertuples():
        signal = signals[row.predictor]
        active = signal.eq(1) & regime.notna()
        up_n = int((active & regime.eq("Up")).sum())
        down_n = int((active & regime.eq("Down")).sum())
        total = up_n + down_n
        rows.append({
            "family": row.family, "signal": row.predictor, "pr_group": row.pr_group,
            "Up N": up_n, "Down N": down_n,
            "Up %": up_n / total if total else np.nan,
            "Down %": down_n / total if total else np.nan,
        })
    return pd.DataFrame(rows)


def _sample_description(values: pd.Series) -> dict:
    clean = values.dropna()
    return {
        "N": len(clean),
        "mean_return": clean.mean(),
        "median_return": clean.median(),
        "win_rate": clean.gt(0).mean() if len(clean) else np.nan,
        "standard_deviation": clean.std(ddof=1) if len(clean) > 1 else np.nan,
    }


def regime_primary_results(
    signals: pd.DataFrame,
    regime: pd.Series,
    outcomes: pd.DataFrame,
    specs: pd.DataFrame,
) -> pd.DataFrame:
    """Describe signal and non-signal within the same regime only."""
    rows = []
    joined = signals.join(regime.rename("regime")).join(outcomes, how="inner")
    for spec in specs.itertuples():
        for outcome in [c for c in OUTCOME_COLUMNS if c in outcomes]:
            for label in ("Up", "Down"):
                valid = joined[spec.predictor].notna() & joined["regime"].eq(label)
                signal_mask = valid & joined[spec.predictor].eq(1)
                control_mask = valid & joined[spec.predictor].eq(0)
                descriptions = {
                    0: _sample_description(joined.loc[control_mask, outcome]),
                    1: _sample_description(joined.loc[signal_mask, outcome]),
                }
                effect = descriptions[1]["mean_return"] - descriptions[0]["mean_return"]
                state = {("Up", 0): "A", ("Up", 1): "B", ("Down", 0): "C", ("Down", 1): "D"}
                for signal_state in (0, 1):
                    rows.append({
                        "family": spec.family, "signal": spec.predictor,
                        "pr_group": spec.pr_group, "k": spec.k,
                        "rolling_window": spec.rolling_window,
                        "outcome": outcome, "regime": label,
                        "signal_state": signal_state, "state": state[(label, signal_state)],
                        "regime_N": descriptions[0]["N"] + descriptions[1]["N"],
                        "signal_N": descriptions[1]["N"],
                        **descriptions[signal_state],
                        "effect_vs_same_regime_non_signal": effect if signal_state == 1 else 0.0,
                    })
    return pd.DataFrame(rows)


def _empty_fit(status: str, n: int, counts: dict[str, int]) -> dict:
    return {
        "N": n, **counts, "beta_up": np.nan, "p_up": np.nan,
        "beta_down": np.nan, "p_down": np.nan,
        "down_regime_beta": np.nan, "down_regime_p": np.nan,
        "interaction_beta": np.nan, "interaction_p": np.nan,
        "condition_number": np.nan, "max_leverage": np.nan, "status": status,
    }


def _hc3_ols(y: pd.Series, design: pd.DataFrame) -> dict[str, object]:
    """Small dependency-light OLS/HC3 implementation for a prevalidated full-rank design."""
    x = design.to_numpy(dtype=float)
    target = y.to_numpy(dtype=float)
    xtx_inverse = np.linalg.inv(x.T @ x)
    beta = xtx_inverse @ x.T @ target
    residual = target - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inverse, x)
    scaled = residual / (1.0 - leverage)
    meat = x.T @ (x * np.square(scaled)[:, None])
    covariance = xtx_inverse @ meat @ xtx_inverse
    standard_errors = np.sqrt(np.diag(covariance))
    dof = max(len(target) - x.shape[1], 1)
    t_values = np.divide(
        beta, standard_errors, out=np.full_like(beta, np.nan), where=standard_errors > 0
    )
    pvalues = 2 * stats.t.sf(np.abs(t_values), dof)
    return {
        "params": pd.Series(beta, index=design.columns),
        "pvalues": pd.Series(pvalues, index=design.columns),
        "covariance": pd.DataFrame(covariance, index=design.columns, columns=design.columns),
        "leverage": leverage,
        "df_resid": dof,
    }


def fit_regime_interaction(
    signal: pd.Series,
    down_regime: pd.Series,
    future: pd.Series,
    config: MarginRegimeInteractionConfig | None = None,
    turnover: pd.Series | None = None,
) -> dict:
    """Fit HC3 interaction model, optionally controlling continuous turnover."""
    config = config or MarginRegimeInteractionConfig()
    values = {"future": future, "signal": signal, "down": down_regime}
    if turnover is not None:
        values["turnover"] = turnover
    frame = pd.concat(values, axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    frame["interaction"] = frame["signal"] * frame["down"]
    counts = {
        "signal_up_N": int((frame.signal.eq(1) & frame.down.eq(0)).sum()),
        "non_signal_up_N": int((frame.signal.eq(0) & frame.down.eq(0)).sum()),
        "signal_down_N": int((frame.signal.eq(1) & frame.down.eq(1)).sum()),
        "non_signal_down_N": int((frame.signal.eq(0) & frame.down.eq(1)).sum()),
    }
    if len(frame) < 4 * config.min_group_n:
        return _empty_fit("insufficient_sample", len(frame), counts)
    if min(counts.values()) < config.min_group_n:
        return _empty_fit("insufficient_group_sample", len(frame), counts)
    columns = ["const", "signal", "down", "interaction"]
    design = pd.DataFrame({
        "const": 1.0, "signal": frame.signal.astype(float),
        "down": frame.down.astype(float), "interaction": frame.interaction.astype(float),
    }, index=frame.index)
    if turnover is not None:
        design["turnover"] = frame.turnover.astype(float)
        columns.append("turnover")
        if frame.turnover.nunique() < 2:
            return _empty_fit("no_turnover_variation", len(frame), counts)
    matrix = design[columns].to_numpy()
    rank = int(np.linalg.matrix_rank(matrix))
    condition = float(np.linalg.cond(matrix))
    if rank < len(columns) or not np.isfinite(condition) or condition >= config.max_condition_number:
        out = _empty_fit("rank_deficient", len(frame), counts)
        out["condition_number"] = condition
        return out
    fit = _hc3_ols(frame.future, design[columns])
    leverage = np.asarray(fit["leverage"], dtype=float)
    max_leverage = float(np.max(leverage)) if len(leverage) and np.isfinite(leverage).all() else np.nan
    if not np.isfinite(leverage).all() or np.any(leverage >= 1 - config.leverage_tolerance):
        out = _empty_fit("high_leverage_unstable", len(frame), counts)
        out.update(condition_number=condition, max_leverage=max_leverage)
        return out
    params = fit["params"]
    pvalues = fit["pvalues"]
    covariance = fit["covariance"]
    beta_down = params["signal"] + params["interaction"]
    down_variance = (
        covariance.loc["signal", "signal"]
        + covariance.loc["interaction", "interaction"]
        + 2 * covariance.loc["signal", "interaction"]
    )
    if down_variance < 0 or not np.isfinite(down_variance):
        return _empty_fit("hc3_nonfinite", len(frame), counts)
    down_se = float(np.sqrt(down_variance))
    dof = float(fit["df_resid"])
    p_down = float(2 * stats.t.sf(abs(beta_down / down_se), dof)) if down_se > 0 else np.nan
    finite = np.isfinite([
        params["signal"], pvalues["signal"], beta_down, p_down,
        params["interaction"], pvalues["interaction"],
    ]).all() and np.isfinite(covariance.to_numpy()).all()
    if not finite:
        return _empty_fit("hc3_nonfinite", len(frame), counts)
    result = {
        "N": len(frame), **counts,
        "beta_up": float(params["signal"]), "p_up": float(pvalues["signal"]),
        "beta_down": float(beta_down), "p_down": p_down,
        "down_regime_beta": float(params["down"]),
        "down_regime_p": float(pvalues["down"]),
        "interaction_beta": float(params["interaction"]),
        "interaction_p": float(pvalues["interaction"]),
        "condition_number": condition, "max_leverage": max_leverage,
        "status": "estimated",
    }
    if turnover is not None:
        result.update(
            turnover_beta=float(params["turnover"]),
            turnover_p=float(pvalues["turnover"]),
        )
    return result


def run_interaction_models(
    signals: pd.DataFrame,
    regime: pd.Series,
    outcomes: pd.DataFrame,
    specs: pd.DataFrame,
    config: MarginRegimeInteractionConfig | None = None,
    turnover_controls: dict[tuple[int, int], pd.Series] | None = None,
) -> pd.DataFrame:
    config = config or MarginRegimeInteractionConfig()
    down = regime.eq("Down").astype(float).where(regime.notna())
    rows = []
    for spec in specs.itertuples():
        turnover = None
        if turnover_controls is not None:
            turnover = turnover_controls[(int(spec.k), int(spec.rolling_window))]
        for outcome in [c for c in OUTCOME_COLUMNS if c in outcomes]:
            fit = fit_regime_interaction(
                signals[spec.predictor], down, outcomes[outcome], config, turnover
            )
            rows.append({
                "family": spec.family, "signal": spec.predictor,
                "pr_group": spec.pr_group, "k": int(spec.k),
                "rolling_window": int(spec.rolling_window), "outcome": outcome,
                "model": "turnover_controlled" if turnover is not None else "binary_primary",
                **fit,
            })
    return pd.DataFrame(rows)


def apply_interaction_fdr(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correct only primary binary interaction p-values in this incremental universe."""
    primary = results.loc[results["model"].eq("binary_primary")].copy()
    fdr = primary.copy()
    fdr["family_fdr_q_value"] = np.nan
    for _, indices in fdr.groupby("family").groups.items():
        use = fdr.loc[indices].index[fdr.loc[indices, "interaction_p"].notna()]
        if len(use):
            fdr.loc[use, "family_fdr_q_value"] = _bh_adjust(fdr.loc[use, "interaction_p"])
    fdr["global_fdr_q_value"] = np.nan
    use = fdr.index[fdr["interaction_p"].notna()]
    if len(use):
        fdr.loc[use, "global_fdr_q_value"] = _bh_adjust(fdr.loc[use, "interaction_p"])
    fdr["evidence_level"] = "No Evidence"
    fdr.loc[fdr.interaction_p.lt(.05), "evidence_level"] = "Level C"
    fdr.loc[fdr.family_fdr_q_value.lt(.05), "evidence_level"] = "Level B"
    fdr.loc[fdr.global_fdr_q_value.lt(.05), "evidence_level"] = "Level A"
    fdr.loc[fdr["interaction_p"].isna(), "evidence_level"] = "Not Testable"
    fdr["fdr_scope"] = FDR_SCOPE
    levels = ("Level A", "Level B", "Level C", "No Evidence", "Not Testable")
    rows = [
        {"metric": "number_of_tests_total", "value": len(fdr)},
        {"metric": "number_of_tests_estimable", "value": int(fdr.interaction_p.notna().sum())},
    ]
    rows += [
        {"metric": level, "value": int(fdr.evidence_level.eq(level).sum())}
        for level in levels
    ]
    diagnostics = pd.DataFrame(rows)
    diagnostics["fdr_scope"] = FDR_SCOPE
    return fdr, diagnostics


def _bh_adjust(pvalues: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values, aligned to the source index."""
    values = pvalues.astype(float)
    order = np.argsort(values.to_numpy())
    ranked = values.to_numpy()[order]
    n = len(ranked)
    adjusted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    restored = np.empty(n, dtype=float)
    restored[order] = adjusted
    return pd.Series(restored, index=values.index)


def build_turnover_controls(
    datasets: dict[str, pd.DataFrame],
    primary_symbols: set[str],
    specs: pd.DataFrame,
) -> dict[tuple[int, int], pd.Series]:
    from .features import trailing_percentile

    base = _turnover_base_series(datasets, primary_symbols)
    controls = {}
    for k, window in specs[["k", "rolling_window"]].drop_duplicates().itertuples(index=False):
        raw = margin_turnover_variant_series(base, int(k))["amount_ratio"]
        controls[(int(k), int(window))] = trailing_percentile(raw, int(window)) / 100.0
    return controls


def continuous_prior_robustness(
    signals: pd.DataFrame,
    prior: pd.Series,
    outcomes: pd.DataFrame,
    specs: pd.DataFrame,
    config: MarginRegimeInteractionConfig | None = None,
) -> pd.DataFrame:
    """Secondary model only; its interaction p-values never enter primary FDR."""
    config = config or MarginRegimeInteractionConfig()
    rows = []
    for spec in specs.itertuples():
        for outcome in [c for c in OUTCOME_COLUMNS if c in outcomes]:
            frame = pd.concat({
                "future": outcomes[outcome], "signal": signals[spec.predictor], "prior": prior,
            }, axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            frame["interaction"] = frame.signal * frame.prior
            base = {"N": len(frame), "signal_beta": np.nan, "prior_beta": np.nan,
                    "continuous_interaction_beta": np.nan, "continuous_interaction_p": np.nan}
            if len(frame) < config.min_group_n * 2 or frame.signal.value_counts().min() < config.min_group_n:
                fit = base | {"status": "insufficient_sample"}
            else:
                design = pd.DataFrame({"const": 1., "signal": frame.signal, "prior": frame.prior,
                                       "interaction": frame.interaction}, index=frame.index)
                matrix = design.to_numpy()
                condition = np.linalg.cond(matrix)
                if np.linalg.matrix_rank(matrix) < 4 or not np.isfinite(condition) or condition >= config.max_condition_number:
                    fit = base | {"status": "rank_deficient"}
                else:
                    model = _hc3_ols(frame.future, design)
                    leverage = np.asarray(model["leverage"])
                    if not np.isfinite(leverage).all() or np.any(leverage >= 1 - config.leverage_tolerance):
                        fit = base | {"status": "high_leverage_unstable"}
                    else:
                        params = model["params"].to_numpy()
                        pvalues = model["pvalues"].to_numpy()
                        covariance = model["covariance"].to_numpy()
                        if not np.isfinite(np.r_[params, pvalues, covariance.ravel()]).all():
                            fit = base | {"status": "hc3_nonfinite"}
                        else:
                            fit = base | {"signal_beta": params[1], "prior_beta": params[2],
                                          "continuous_interaction_beta": params[3],
                                          "continuous_interaction_p": pvalues[3], "status": "estimated"}
            rows.append({"family": spec.family, "signal": spec.predictor, "pr_group": spec.pr_group,
                         "k": spec.k, "rolling_window": spec.rolling_window,
                         "outcome": outcome, "model": "secondary_continuous_prior", **fit})
    return pd.DataFrame(rows)


def annual_regime_results(
    candidates: pd.DataFrame,
    signals: pd.DataFrame,
    regime: pd.Series,
    outcomes: pd.DataFrame,
    config: MarginRegimeInteractionConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or MarginRegimeInteractionConfig()
    rows = []
    candidate_keys = candidates[["family", "signal", "pr_group", "k", "rolling_window", "outcome"]]
    for candidate in candidate_keys.drop_duplicates().itertuples():
        for year in sorted(set(outcomes.index.year)):
            year_mask = outcomes.index.year == year
            signal = signals[candidate.signal]
            effects = {}
            counts = {}
            for label in ("Up", "Down"):
                valid = year_mask & signal.notna() & regime.eq(label) & outcomes[candidate.outcome].notna()
                active = valid & signal.eq(1)
                inactive = valid & signal.eq(0)
                counts[label] = int(active.sum())
                effects[label] = (
                    outcomes.loc[active, candidate.outcome].mean()
                    - outcomes.loc[inactive, candidate.outcome].mean()
                    if active.any() and inactive.any() else np.nan
                )
            interaction = effects["Down"] - effects["Up"]
            rows.append({
                "family": candidate.family, "signal": candidate.signal,
                "pr_group": candidate.pr_group, "k": candidate.k,
                "rolling_window": candidate.rolling_window, "outcome": candidate.outcome,
                "year": year, "signal_up_N": counts["Up"], "signal_down_N": counts["Down"],
                "effect_up": effects["Up"], "effect_down": effects["Down"],
                "interaction_effect": interaction,
                "interaction_direction": "positive" if interaction > 0 else "negative" if interaction < 0 else "zero_or_missing",
            })
    annual = pd.DataFrame(rows)
    summaries = []
    keys = ["family", "signal", "pr_group", "k", "rolling_window", "outcome"]
    for key, group in annual.groupby(keys, dropna=False):
        usable = group[group.interaction_effect.notna()]
        total = group.signal_up_N.add(group.signal_down_N).sum()
        by_year = group.signal_up_N.add(group.signal_down_N)
        positive = int(usable.interaction_effect.gt(0).sum())
        negative = int(usable.interaction_effect.lt(0).sum())
        years = len(usable)
        largest = by_year.max() / total if total else np.nan
        direction_ratio = max(positive, negative) / years if years else np.nan
        robust = (
            years >= config.robustness_min_years
            and direction_ratio >= config.robustness_min_direction_ratio
            and pd.notna(largest)
            and largest <= config.robustness_max_year_sample_share
        )
        summaries.append(dict(zip(keys, key)) | {
            "years_with_effect": years, "positive_years": positive, "negative_years": negative,
            "largest_year_sample_share": largest,
            "interaction_direction_consistency": direction_ratio,
            "annual_robustness": "robust" if robust else "insufficient_or_mixed",
        })
    return annual, pd.DataFrame(summaries)


def regime_event_clusters(
    signals: pd.DataFrame,
    regime: pd.Series,
    outcomes: pd.DataFrame,
    specs: pd.DataFrame,
) -> pd.DataFrame:
    """Merge adjacent trading rows only while both signal and regime remain the same."""
    rows = []
    for spec in specs.itertuples():
        active = signals[spec.predictor].eq(1) & regime.notna()
        positions = np.flatnonzero(active.to_numpy())
        if not len(positions):
            continue
        cluster = [int(positions[0])]
        clusters: list[list[int]] = []
        for position in positions[1:]:
            position = int(position)
            previous = cluster[-1]
            same_regime = regime.iloc[position] == regime.iloc[previous]
            if position == previous + 1 and same_regime:
                cluster.append(position)
            else:
                clusters.append(cluster)
                cluster = [position]
        clusters.append(cluster)
        for members in clusters:
            first, last = members[0], members[-1]
            for outcome in [c for c in OUTCOME_COLUMNS if c in outcomes]:
                rows.append({
                    "family": spec.family, "signal": spec.predictor,
                    "pr_group": spec.pr_group, "regime": regime.iloc[first],
                    "cluster_start": signals.index[first], "cluster_end": signals.index[last],
                    "cluster_length": len(members), "first_signal_date": signals.index[first],
                    "outcome": outcome, "first_signal_return": outcomes[outcome].reindex(signals.index).iloc[first],
                    "daily_N": len(positions),
                })
    result = pd.DataFrame(rows)
    if not result.empty:
        counts = result.groupby(["signal", "outcome"]).size().rename("cluster_N")
        result = result.join(counts, on=["signal", "outcome"])
    return result


def _create_run_directory(config: MarginRegimeInteractionConfig) -> Path:
    stamp = datetime.now(ZoneInfo(config.timezone)).strftime("%Y%m%d_%H%M%S")
    short = _git_value(["rev-parse", "--short", "HEAD"])
    path = config.output_root / f"{stamp}_{short}_margin_regime_interaction_adjusted"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_run_info(
    run_dir: Path, baseline_dir: Path, turnover_dir: Path, composition_dir: Path,
    config: MarginRegimeInteractionConfig,
) -> None:
    try:
        finlab_version = importlib.metadata.version("finlab")
    except importlib.metadata.PackageNotFoundError:
        finlab_version = "unknown"
    values = {
        "repository": config.repository,
        "branch": _git_value(["branch", "--show-current"]),
        "baseline_commit": config.baseline_commit,
        "turnover_commit": config.turnover_commit,
        "composition_commit": config.composition_commit,
        "current_commit": _git_value(["rev-parse", "HEAD"]),
        "baseline_run_dir": baseline_dir, "turnover_run_dir": turnover_dir,
        "composition_run_dir": composition_dir,
        "run_timestamp": datetime.now(ZoneInfo(config.timezone)).isoformat(),
        "python_version": platform.python_version(), "finlab_version": finlab_version,
        "price_source_open": "etl:adj_open", "price_source_close": "etl:adj_close",
        "prior_5d_return": "adjusted_close[t] / adjusted_close[t-5] - 1",
        "regime_definition": "Up if prior_5d_return > 0; Down if prior_5d_return <= 0",
        "primary_predictors": "margin_buy amount_ratio; margin_sell amount_ratio",
        "primary_tail": config.primary_tail, "outcome_horizons": config.outcome_horizons,
        "fdr_scope": FDR_SCOPE,
        "baseline_full_research_rerun": False, "turnover_full_research_rerun": False,
        "composition_full_research_rerun": False,
    }
    (run_dir / "run_info_regime_interaction.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in values.items()) + "\n", encoding="utf-8"
    )


def _hypothesis_label(condition: bool | None, partial: bool = False) -> str:
    if condition is None:
        return "Unable to Determine"
    if condition:
        return "Partially Supported" if partial else "Supported"
    return "Not Supported"


def _combined_hypothesis(occurrence: bool | None, interaction: bool) -> str:
    if occurrence is None:
        return "Unable to Determine"
    if occurrence and interaction:
        return "Supported"
    if occurrence or interaction:
        return "Partially Supported"
    return "Not Supported"


def _summary_markdown(result: dict[str, object]) -> str:
    distribution = result["signal_distribution"]
    fdr = result["interaction_fdr"]
    annual = result["annual_robustness_summary"]
    controlled = result["turnover_controlled"]
    clusters = result["event_clusters"]
    buy = distribution[distribution.family.eq("margin_buy")]
    sell = distribution[distribution.family.eq("margin_sell")]
    buy_up = buy["Up N"].sum() > buy["Down N"].sum() if len(buy) else None
    sell_down = sell["Down N"].sum() > sell["Up N"].sum() if len(sell) else None
    any_a = bool(fdr.evidence_level.eq("Level A").any()) if len(fdr) else False
    buy_interaction = bool((
        fdr.family.eq("margin_buy")
        & fdr.evidence_level.eq("Level A")
        & fdr.beta_up.gt(0)
        & fdr.interaction_beta.lt(0)
    ).any()) if len(fdr) else False
    sell_interaction = bool(
        (fdr.family.eq("margin_sell") & fdr.evidence_level.eq("Level A")).any()
    ) if len(fdr) else False
    keys = ["family", "signal", "pr_group", "k", "rolling_window", "outcome"]
    primary_a = fdr.loc[
        fdr.evidence_level.eq("Level A"), keys + ["interaction_beta"]
    ].rename(columns={"interaction_beta": "primary_interaction_beta"}) if len(fdr) else pd.DataFrame()
    estimated_control = controlled[controlled.status.eq("estimated")] if len(controlled) else controlled
    paired_control = (
        estimated_control.merge(primary_a, on=keys, how="inner")
        if len(estimated_control) and len(primary_a) else pd.DataFrame()
    )
    control_survives = bool((
        paired_control.interaction_p.lt(.05)
        & np.sign(paired_control.interaction_beta).eq(np.sign(paired_control.primary_interaction_beta))
    ).any()) if len(paired_control) else False
    annual_robust = bool(annual["annual_robustness"].eq("robust").any()) if len(annual) else False
    cluster_counts = clusters.drop_duplicates(["signal", "outcome"])
    cluster_ratio = (
        cluster_counts.cluster_N.sum() / cluster_counts.daily_N.sum()
        if len(cluster_counts) and cluster_counts.daily_N.sum() else np.nan
    )
    hypothesis_table = (
        "| Hypothesis | Status |\n|---|---|\n"
        f"| H1 Margin Buy concentrated in Up with regime-dependent effect | {_combined_hypothesis(buy_up, buy_interaction)} |\n"
        f"| H2 Margin Sell concentrated in Down with regime-dependent effect | {_combined_hypothesis(sell_down, sell_interaction)} |\n"
        f"| H3 Margin Sell effect differs across regimes | {_hypothesis_label(sell_interaction)} |\n"
        f"| H4 Regime adds information beyond turnover | {_hypothesis_label(control_survives)} |"
    )
    sections = {
        "A. 目前假設": "Margin Buy 與 Margin Sell 可能屬於不同 prior-return regime 下的不同交易行為；本研究不作因果認定。\n\n" + hypothesis_table,
        "B. 市場機制": "Turnover 描述市場有多忙；prior 5-day adjusted return 只用來區分 signal date 當下的 Up/Down regime。",
        "C. Signal occurrence by regime": f"Margin Buy high 較集中於 Up：{_hypothesis_label(buy_up)}；Margin Sell high 較集中於 Down：{_hypothesis_label(sell_down)}。詳見 CSV。",
        "D. Margin Buy：Up vs Down": "比較 signal 與同一 regime 的 non-signal；不得用全體 non-signal 取代。",
        "E. Margin Sell：Up vs Down": "若 Down 後偏多，只能列為 forced deleveraging/capitulation rebound 的可能解釋，不得直接等同去槓桿。",
        "F. Interaction regression": "Primary model 為 Signal + DownRegime + Signal×DownRegime；beta_up=β1，beta_down=β1+β3。",
        "G. FDR": f"Interaction global Level A 是否存在：{_hypothesis_label(any_a)}；scope=`{FDR_SCOPE}`，未併入 baseline/turnover FDR。",
        "H. Annual robustness": f"至少一項候選跨年度通過門檻：{_hypothesis_label(annual_robust)}。另列 2020、2022、2025 與所有年度。",
        "I. Event clustering": f"連續同 regime signal 已合併；aggregate cluster/daily diagnostic ratio={cluster_ratio if np.isfinite(cluster_ratio) else 'NA'}。",
        "J. Turnover control": f"控制 amount-ratio Margin Turnover 後至少一項 interaction raw p<.05：{_hypothesis_label(control_survives)}。",
        "K. 主要風險": "Overlapping outcomes、tail 小樣本、年度/事件集中、制度變動、非因果解釋與 frozen-output selection 都可能限制結論。",
        "L. 反對者觀點": "Buy/Sell 可能都只是信用市場 activity；regime occurrence 差異不等於 future-return interaction。",
        "M. 最終判定": "整體：保留" if any_a and control_survives and annual_robust else "整體：修改後再測" if len(fdr) else "整體：淘汰",
    }
    questions = [
        f"1. Margin Buy high 主要在 Up？{_hypothesis_label(buy_up)}",
        f"2. Margin Sell high 主要在 Down？{_hypothesis_label(sell_down)}",
        "3. Margin Buy 在 Up 是否更強？見 within-regime effects 與 interaction；未通過 FDR 時為 Unable to Determine。",
        "4. Margin Sell 在 Down 是否不同？以 interaction FDR 判定。",
        "5. Down 後偏多是否 capitulation rebound？只能列可能機制，不能由本研究證明。",
        "6. Up regime 偏多是否正常 turnover/profit taking？只能列可能機制。",
        f"7. Interaction 通過 FDR？{_hypothesis_label(any_a)}",
        f"8. 控制 Turnover 後仍存在？{_hypothesis_label(control_survives)}（robustness raw p，不另擴大 primary FDR）。",
        f"9. 跨年度穩定？{_hypothesis_label(annual_robust)}",
        "10. 是否由少數 clusters 驅動？需合併 daily/cluster N 與年度結果判讀。",
        "11. 支持 Buy=bull、Sell=deleveraging？只有 occurrence 與 interaction 同時有意義才可支持。",
        "12. 或兩者只是信用 activity？若 interaction 無證據，保留此反對解釋。",
    ]
    lines = ["# Margin Buy / Sell × Prior-return Regime", ""]
    for heading, body in sections.items():
        lines.extend([f"## {heading}", "", body, ""])
    lines.extend(["## 必答問題", "", *questions, ""])
    return "\n".join(lines)


def run_margin_regime_interaction_study(
    baseline_run_dir: str | Path,
    turnover_run_dir: str | Path,
    composition_run_dir: str | Path,
    config: MarginRegimeInteractionConfig | None = None,
    provider=None,
    export: bool = True,
    datasets: dict[str, pd.DataFrame] | None = None,
) -> dict[str, object]:
    """Run only this increment; do not invoke any earlier full study."""
    config = config or MarginRegimeInteractionConfig()
    baseline = load_frozen_baseline(baseline_run_dir, config.baseline_commit)
    turnover = load_frozen_turnover(turnover_run_dir, config.turnover_commit, config.baseline_commit)
    composition = load_frozen_composition(
        composition_run_dir, config.composition_commit, config.turnover_commit, config.baseline_commit
    )
    data = load_turnover_data(provider) if datasets is None else datasets
    universe, primary_symbols, limitation = build_universe_diagnostics(
        data["margin_balance"], data["market_value"]
    )
    specs = select_primary_signal_specs(baseline["fdr_results"], config.primary_tail)
    signals = build_signal_frame(data, specs)
    close = data["close"][config.target_symbol]
    prior = prior_5d_return(close)
    regime = binary_regime(prior)
    outcomes = build_outcomes(
        data["open"][config.target_symbol], close, config.outcome_horizons
    )
    price_diagnostics = validate_split_window(data["open"][config.target_symbol], close, outcomes)
    distribution = regime_signal_distribution(signals, regime, specs)
    primary = regime_primary_results(signals, regime, outcomes, specs)
    interaction = run_interaction_models(signals, regime, outcomes, specs, config)
    fdr, fdr_diag = apply_interaction_fdr(interaction)
    candidates = fdr[fdr.evidence_level.isin(["Level A", "Level B"])]
    annual, annual_summary = annual_regime_results(candidates, signals, regime, outcomes, config)
    clusters = regime_event_clusters(signals, regime, outcomes, specs)
    controls = build_turnover_controls(data, primary_symbols, specs)
    turnover_controlled = run_interaction_models(
        signals, regime, outcomes, specs, config, controls
    )
    secondary = continuous_prior_robustness(signals, prior, outcomes, specs, config)
    result: dict[str, object] = {
        "baseline": baseline, "turnover": turnover, "composition": composition,
        "universe": universe, "UNIVERSE_LIMITATION": limitation,
        "signal_specs": specs, "signals": signals, "prior_5d_return": prior,
        "regime": regime, "outcomes": outcomes,
        "outcome_price_diagnostics": price_diagnostics,
        "signal_distribution": distribution, "primary_results": primary,
        "interaction_results": interaction, "interaction_fdr": fdr,
        "interaction_fdr_diagnostics": fdr_diag, "annual_results": annual,
        "annual_robustness_summary": annual_summary, "event_clusters": clusters,
        "turnover_controlled": turnover_controlled,
        "secondary_continuous": secondary,
    }
    if export:
        run_dir = _create_run_directory(config)
        result["run_dir"] = run_dir
        outputs = {
            "regime_signal_distribution.csv": distribution,
            "regime_primary_results.csv": primary,
            "regime_interaction_results.csv": interaction,
            "regime_interaction_fdr_results.csv": fdr,
            "regime_interaction_fdr_diagnostics.csv": fdr_diag,
            "regime_annual_results.csv": annual,
            "regime_annual_robustness_summary.csv": annual_summary,
            "regime_event_clusters.csv": clusters,
            "regime_turnover_controlled_results.csv": turnover_controlled,
            "regime_secondary_continuous_results.csv": secondary,
            "outcome_price_diagnostics.csv": price_diagnostics,
        }
        for filename, frame in outputs.items():
            frame.to_csv(run_dir / filename, index=False)
        (run_dir / "regime_summary.md").write_text(_summary_markdown(result), encoding="utf-8")
        _write_run_info(
            run_dir, Path(baseline_run_dir), Path(turnover_run_dir), Path(composition_run_dir), config
        )
    return result
