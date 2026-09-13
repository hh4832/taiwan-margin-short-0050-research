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

from .features import rolling_ratio, trailing_percentile
from .margin_composition import TURNOVER_BASE_COMMIT
from .margin_regime_interaction import (
    COMPOSITION_COMMIT,
    _bh_adjust,
    _hc3_ols,
    build_turnover_controls,
    prior_5d_return,
    select_primary_signal_specs,
    validate_regime_input_runs,
)
from .margin_turnover import (
    BASELINE_COMMIT,
    _read_run_info,
    _sum_columns,
    load_turnover_data,
)
from .outcomes import build_outcomes
from .price_validation import validate_split_window
from .universe import build_universe_diagnostics


REGIME_COMMIT = "a5ca1e980980e92428d757578a5e7d9027ff5d63"
INTERACTION_FDR_SCOPE = "joint_flow_prior_return_primary_interactions"
MAIN_EFFECT_FDR_SCOPE = "joint_flow_prior_return_incremental_main_effects"
COLAB_REPO_DIR = Path("/content/taiwan-margin-short-0050-research")
FORMAL_BASELINE_RUN_DIR = Path(
    "/content/drive/MyDrive/Quant_Research/taiwan-margin-short-0050-research/"
    "20260912_220859_88d9f26_adjusted_price"
)
FORMAL_TURNOVER_RUN_DIR = Path(
    "/content/drive/MyDrive/Quant_Research/taiwan-margin-short-0050-research/"
    "margin_turnover/20260913_075344_e613b5e_margin_turnover_adjusted"
)
FORMAL_COMPOSITION_RUN_DIR = Path(
    "/content/drive/MyDrive/Quant_Research/taiwan-margin-short-0050-research/"
    "margin_composition/20260913_080358_cf16e4a_margin_composition_adjusted"
)
FORMAL_REGIME_RUN_DIR = Path(
    "/content/drive/MyDrive/Quant_Research/taiwan-margin-short-0050-research/"
    "margin_regime_interaction/20260913_093038_a5ca1e9_margin_regime_interaction_adjusted"
)
FORMAL_DRIVE_OUTPUT_ROOT = Path(
    "/content/drive/MyDrive/Quant_Research/taiwan-margin-short-0050-research/"
    "margin_buy_sell_prior_return_joint"
)
OUTCOMES = tuple(f"O1_C{h}" for h in (1, 2, 3, 5, 10, 20))
PRIMARY_RESULT_COLUMNS = (
    "k", "rolling_window", "outcome", "N_total", "BuyHigh_N", "SellHigh_N",
    "BothHigh_N", "BuyOnly_N", "SellOnly_N", "Neither_N", "buy_beta", "buy_p",
    "sell_beta", "sell_p", "prior_beta", "prior_p", "buy_prior_beta",
    "buy_prior_p", "sell_prior_beta", "sell_prior_p", "condition_number",
    "max_vif", "max_leverage", "status",
)
ANNUAL_COLUMNS = (
    "hypothesis", "k", "rolling_window", "outcome", "year", "sample_N",
    "interaction_beta", "effect_direction", "status",
)
ANNUAL_SUMMARY_COLUMNS = (
    "hypothesis", "k", "rolling_window", "outcome", "years_available",
    "same_direction_years", "direction_consistency", "largest_year_sample_share",
    "annual_robustness",
)


@dataclass(frozen=True)
class JointFlowPriorReturnConfig:
    repository: str = "taiwan-margin-short-0050-research"
    timezone: str = "Asia/Taipei"
    target_symbol: str = "0050"
    baseline_commit: str = BASELINE_COMMIT
    turnover_commit: str = TURNOVER_BASE_COMMIT
    composition_commit: str = COMPOSITION_COMMIT
    regime_commit: str = REGIME_COMMIT
    k_values: tuple[int, ...] = (1, 3, 5, 10)
    rolling_windows: tuple[int, ...] = (126, 252, 504, 756)
    outcome_horizons: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    primary_tail: str = "PR95-100"
    prior_return_days: int = 5
    min_joint_state_n: int = 20
    max_condition_number: float = 1e12
    max_vif: float = 10.0
    leverage_tolerance: float = 1e-10
    robustness_min_years: int = 3
    robustness_min_direction_ratio: float = 0.60
    robustness_max_year_sample_share: float = 0.50
    absorption_partial_threshold: float = 0.30
    absorption_large_threshold: float = 0.70
    significance_threshold: float = 0.05
    beta_epsilon: float = 1e-12
    output_root: Path = Path("outputs_joint_flow_prior_return")


def _git_value(args: list[str], default: str = "unknown") -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return default


def _require_values(layer: str, info: dict[str, str], expected: dict[str, str]) -> None:
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(
                f"{layer} {key} mismatch: expected {value!r}, got {info.get(key)!r}"
            )


def validate_joint_input_runs(
    baseline_run_dir: str | Path,
    turnover_run_dir: str | Path,
    composition_run_dir: str | Path,
    regime_run_dir: str | Path,
    config: JointFlowPriorReturnConfig | None = None,
) -> dict[str, object]:
    """Validate the exact frozen four-layer adjusted-price dependency chain."""
    config = config or JointFlowPriorReturnConfig()
    first_three = validate_regime_input_runs(
        baseline_run_dir, turnover_run_dir, composition_run_dir
    )
    regime_dir = Path(regime_run_dir).expanduser()
    required = (
        "run_info_regime_interaction.txt",
        "regime_interaction_fdr_results.csv",
        "regime_secondary_continuous_results.csv",
        "regime_turnover_controlled_results.csv",
    )
    if not regime_dir.is_dir():
        raise FileNotFoundError(f"Regime run directory does not exist: {regime_dir}")
    missing = [name for name in required if not (regime_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Regime run directory is missing required files: {missing}; path={regime_dir}"
        )
    info = _read_run_info(regime_dir / "run_info_regime_interaction.txt")
    _require_values("Regime", info, {
        "repository": config.repository,
        "git_commit": config.regime_commit,
        "baseline_commit": config.baseline_commit,
        "turnover_commit": config.turnover_commit,
        "composition_commit": config.composition_commit,
        "price_source_open": "etl:adj_open",
        "price_source_close": "etl:adj_close",
        "outcome_price_adjusted": "True",
    })
    diagnostics = pd.concat([
        first_three["diagnostics"],
        pd.DataFrame([{
            "layer": "Regime", "run_dir": str(regime_dir),
            "repository": info["repository"], "commit": info["git_commit"],
            "baseline_commit": info["baseline_commit"],
            "turnover_commit": info["turnover_commit"],
            "composition_commit": info["composition_commit"],
            "price_source_open": info["price_source_open"],
            "price_source_close": info["price_source_close"],
            "outcome_price_adjusted": info["outcome_price_adjusted"], "status": "PASS",
        }]),
    ], ignore_index=True, sort=False)
    return {
        **first_three,
        "regime": {
            "run_dir": regime_dir, "run_info": info,
            "interaction_fdr": pd.read_csv(regime_dir / required[1]),
            "secondary_continuous": pd.read_csv(regime_dir / required[2]),
            "turnover_controlled": pd.read_csv(regime_dir / required[3]),
        },
        "diagnostics": diagnostics,
    }


def select_joint_signal_pairs(
    baseline_fdr: pd.DataFrame,
    config: JointFlowPriorReturnConfig | None = None,
) -> pd.DataFrame:
    """Match frozen Buy and Sell PR95 signals on exactly the same k/window."""
    config = config or JointFlowPriorReturnConfig()
    specs = select_primary_signal_specs(baseline_fdr, config.primary_tail)
    specs = specs[
        specs.k.isin(config.k_values) & specs.rolling_window.isin(config.rolling_windows)
    ]
    buy = specs[specs.family.eq("margin_buy")].rename(columns={"predictor": "buy_signal"})
    sell = specs[specs.family.eq("margin_sell")].rename(columns={"predictor": "sell_signal"})
    keys = ["k", "rolling_window", "pr_group"]
    pairs = buy[keys + ["buy_signal"]].merge(
        sell[keys + ["sell_signal"]], on=keys, how="outer", indicator=True
    )
    unmatched = pairs[pairs["_merge"].ne("both")]
    if not unmatched.empty:
        raise ValueError(
            "Margin Buy/Sell signals must match on identical k/window: "
            f"{unmatched[keys + ['_merge']].to_dict('records')}"
        )
    pairs = pairs.drop(columns="_merge").drop_duplicates(keys)
    expected = {(k, w) for k in config.k_values for w in config.rolling_windows}
    actual = set(pairs[["k", "rolling_window"]].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError(
            f"Frozen signal grid mismatch: expected={sorted(expected)}, actual={sorted(actual)}"
        )
    return pairs.sort_values(["k", "rolling_window"]).reset_index(drop=True)


def build_joint_signals_and_ratios(
    datasets: dict[str, pd.DataFrame], pairs: pd.DataFrame
) -> tuple[pd.DataFrame, dict[tuple[int, int], tuple[pd.Series, pd.Series]]]:
    market_amount = _sum_columns(datasets["market_amount"], ("TAIEX", "OTC"))
    columns = ("上市融資交易金額", "上櫃融資交易金額")
    buy_amount = _sum_columns(datasets["aggregate_buy"], columns)
    sell_amount = _sum_columns(datasets["aggregate_sell"], columns)
    signals: dict[str, pd.Series] = {}
    raw_ratios: dict[tuple[int, int], tuple[pd.Series, pd.Series]] = {}
    for row in pairs.itertuples():
        buy_raw = rolling_ratio(buy_amount, market_amount, int(row.k))
        sell_raw = rolling_ratio(sell_amount, market_amount, int(row.k))
        buy_pr = trailing_percentile(buy_raw, int(row.rolling_window))
        sell_pr = trailing_percentile(sell_raw, int(row.rolling_window))
        signals[row.buy_signal] = buy_pr.ge(95).astype(float).where(buy_pr.notna())
        signals[row.sell_signal] = sell_pr.ge(95).astype(float).where(sell_pr.notna())
        raw_ratios[(int(row.k), int(row.rolling_window))] = (buy_raw, sell_raw)
    return pd.DataFrame(signals).sort_index(), raw_ratios


def signal_overlap_diagnostics(
    signals: pd.DataFrame,
    pairs: pd.DataFrame,
    raw_ratios: dict[tuple[int, int], tuple[pd.Series, pd.Series]],
) -> pd.DataFrame:
    rows = []
    for row in pairs.itertuples():
        pair = pd.concat({
            "buy": signals[row.buy_signal], "sell": signals[row.sell_signal],
            "buy_ratio": raw_ratios[(int(row.k), int(row.rolling_window))][0],
            "sell_ratio": raw_ratios[(int(row.k), int(row.rolling_window))][1],
        }, axis=1)
        binary = pair[["buy", "sell"]].dropna()
        raw = pair[["buy_ratio", "sell_ratio"]].dropna()
        buy_n = int(binary.buy.eq(1).sum())
        sell_n = int(binary.sell.eq(1).sum())
        both = int((binary.buy.eq(1) & binary.sell.eq(1)).sum())
        rows.append({
            "k": int(row.k), "rolling_window": int(row.rolling_window),
            "Buy_signal_N": buy_n, "Sell_signal_N": sell_n, "Both_high_N": both,
            "Neither_high_N": int((binary.buy.eq(0) & binary.sell.eq(0)).sum()),
            "P_Sell_high_given_Buy_high": both / buy_n if buy_n else np.nan,
            "P_Buy_high_given_Sell_high": both / sell_n if sell_n else np.nan,
            "phi_correlation": binary.buy.corr(binary.sell) if len(binary) > 1 else np.nan,
            "raw_amount_ratio_pearson": raw.buy_ratio.corr(raw.sell_ratio, method="pearson") if len(raw) > 1 else np.nan,
            "raw_amount_ratio_spearman": raw.buy_ratio.corr(raw.sell_ratio, method="spearman") if len(raw) > 1 else np.nan,
            "complete_binary_N": len(binary), "complete_raw_ratio_N": len(raw),
        })
    return pd.DataFrame(rows)


def joint_state_descriptive(
    signals: pd.DataFrame, prior: pd.Series, outcomes: pd.DataFrame, pairs: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    states = ((0, 0, "BuyLow_SellLow"), (1, 0, "BuyHigh_SellLow"),
              (0, 1, "BuyLow_SellHigh"), (1, 1, "BuyHigh_SellHigh"))
    for spec in pairs.itertuples():
        base = pd.concat({"buy": signals[spec.buy_signal], "sell": signals[spec.sell_signal],
                          "prior": prior}, axis=1)
        for outcome in [c for c in OUTCOMES if c in outcomes]:
            frame = base.join(outcomes[[outcome]]).dropna()
            for buy, sell, state in states:
                group = frame[frame.buy.eq(buy) & frame.sell.eq(sell)]
                values = group[outcome]
                rows.append({
                    "k": int(spec.k), "rolling_window": int(spec.rolling_window),
                    "outcome": outcome, "state": state, "BuyHigh": buy, "SellHigh": sell,
                    "N": len(group), "mean_future_return": values.mean(),
                    "median_future_return": values.median(),
                    "win_rate": values.gt(0).mean() if len(values) else np.nan,
                    "prior_5d_return_mean": group.prior.mean(),
                    "prior_5d_return_median": group.prior.median(),
                })
    return pd.DataFrame(rows)


def _state_counts(frame: pd.DataFrame) -> dict[str, int]:
    buy, sell = frame["buy"].eq(1), frame["sell"].eq(1)
    return {
        "BuyHigh_N": int(buy.sum()), "SellHigh_N": int(sell.sum()),
        "BothHigh_N": int((buy & sell).sum()), "BuyOnly_N": int((buy & ~sell).sum()),
        "SellOnly_N": int((~buy & sell).sum()), "Neither_N": int((~buy & ~sell).sum()),
    }


def _vif(design: pd.DataFrame) -> dict[str, float]:
    result = {}
    for column in [c for c in design if c != "const"]:
        target = design[column].to_numpy(float)
        other = design.drop(columns=column).to_numpy(float)
        fitted = other @ np.linalg.lstsq(other, target, rcond=None)[0]
        total = np.square(target - target.mean()).sum()
        residual = np.square(target - fitted).sum()
        r2 = 1 - residual / total if total > 0 else 1.0
        result[column] = np.inf if r2 >= 1 else 1 / (1 - r2)
    return result


def fit_joint_model(
    buy: pd.Series, sell: pd.Series, prior: pd.Series, future: pd.Series,
    config: JointFlowPriorReturnConfig | None = None,
    turnover: pd.Series | None = None,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Fit the predeclared joint model using complete cases and HC3 covariance."""
    config = config or JointFlowPriorReturnConfig()
    values = {"future": future, "buy": buy, "sell": sell, "prior": prior}
    if turnover is not None:
        values["turnover"] = turnover
    frame = pd.concat(values, axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    counts = _state_counts(frame)
    base = {
        "N_total": len(frame), **counts, "buy_beta": np.nan, "buy_p": np.nan,
        "sell_beta": np.nan, "sell_p": np.nan, "prior_beta": np.nan,
        "prior_p": np.nan, "buy_prior_beta": np.nan, "buy_prior_p": np.nan,
        "sell_prior_beta": np.nan, "sell_prior_p": np.nan,
        "condition_number": np.nan, "max_vif": np.nan, "max_leverage": np.nan,
    }
    empty_marginal = pd.DataFrame(columns=[
        "family", "prior_level", "prior_value", "effect", "ci_low", "ci_high"
    ])
    empty_collinearity = pd.DataFrame(columns=["term", "vif"])
    if len(frame) < 4 * config.min_joint_state_n or min(
        counts[name] for name in ("BothHigh_N", "BuyOnly_N", "SellOnly_N", "Neither_N")
    ) < config.min_joint_state_n:
        return base | {"status": "insufficient_group_sample"}, empty_marginal, empty_collinearity
    frame["buy_prior"] = frame.buy * frame.prior
    frame["sell_prior"] = frame.sell * frame.prior
    design = pd.DataFrame({
        "const": 1.0, "buy": frame.buy, "sell": frame.sell, "prior": frame.prior,
        "buy_prior": frame.buy_prior, "sell_prior": frame.sell_prior,
    }, index=frame.index)
    if turnover is not None:
        design["turnover"] = frame.turnover
    matrix = design.to_numpy(float)
    condition = float(np.linalg.cond(matrix))
    vifs = _vif(design)
    max_vif = max(vifs.values()) if vifs else np.nan
    collinearity = pd.DataFrame([
        {"term": term, "vif": value} for term, value in vifs.items()
    ])
    if (np.linalg.matrix_rank(matrix) < design.shape[1] or not np.isfinite(condition)
            or condition >= config.max_condition_number or not np.isfinite(max_vif)
            or max_vif >= config.max_vif):
        return base | {"condition_number": condition, "max_vif": max_vif,
                       "status": "unstable_collinearity"}, empty_marginal, collinearity
    fit = _hc3_ols(frame.future, design)
    leverage = np.asarray(fit["leverage"], float)
    max_leverage = float(leverage.max())
    if (not np.isfinite(leverage).all()
            or np.any(leverage >= 1 - config.leverage_tolerance)):
        return base | {"condition_number": condition, "max_vif": max_vif,
                       "max_leverage": max_leverage,
                       "status": "high_leverage_unstable"}, empty_marginal, collinearity
    params, pvalues, covariance = fit["params"], fit["pvalues"], fit["covariance"]
    if not np.isfinite(np.r_[params, pvalues, covariance.to_numpy().ravel()]).all():
        return base | {"condition_number": condition, "max_vif": max_vif,
                       "max_leverage": max_leverage,
                       "status": "hc3_nonfinite"}, empty_marginal, collinearity
    result = base | {
        "buy_beta": float(params.buy), "buy_p": float(pvalues.buy),
        "sell_beta": float(params.sell), "sell_p": float(pvalues.sell),
        "prior_beta": float(params.prior), "prior_p": float(pvalues.prior),
        "buy_prior_beta": float(params.buy_prior), "buy_prior_p": float(pvalues.buy_prior),
        "sell_prior_beta": float(params.sell_prior), "sell_prior_p": float(pvalues.sell_prior),
        "condition_number": condition, "max_vif": max_vif,
        "max_leverage": max_leverage, "status": "estimated",
    }
    if turnover is not None:
        result.update(turnover_beta=float(params.turnover), turnover_p=float(pvalues.turnover))
    critical = stats.t.ppf(.975, float(fit["df_resid"]))
    marginal_rows = []
    for label, value in zip(("p25", "p50", "p75"), frame.prior.quantile([.25, .5, .75])):
        for family, main, interaction in (
            ("margin_buy", "buy", "buy_prior"),
            ("margin_sell", "sell", "sell_prior"),
        ):
            effect = params[main] + value * params[interaction]
            variance = (covariance.loc[main, main]
                        + value ** 2 * covariance.loc[interaction, interaction]
                        + 2 * value * covariance.loc[main, interaction])
            se = np.sqrt(variance) if variance >= 0 else np.nan
            marginal_rows.append({
                "family": family, "prior_level": label, "prior_value": value,
                "effect": effect, "ci_low": effect - critical * se,
                "ci_high": effect + critical * se,
            })
    return result, pd.DataFrame(marginal_rows), collinearity


def run_joint_models(
    signals: pd.DataFrame, prior: pd.Series, outcomes: pd.DataFrame,
    pairs: pd.DataFrame, config: JointFlowPriorReturnConfig | None = None,
    turnover_controls: dict[tuple[int, int], pd.Series] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = config or JointFlowPriorReturnConfig()
    rows, marginals, collinearity = [], [], []
    for spec in pairs.itertuples():
        control = (turnover_controls or {}).get((int(spec.k), int(spec.rolling_window)))
        for outcome in [c for c in OUTCOMES if c in outcomes]:
            fit, marginal, col = fit_joint_model(
                signals[spec.buy_signal], signals[spec.sell_signal], prior,
                outcomes[outcome], config, control,
            )
            key = {"k": int(spec.k), "rolling_window": int(spec.rolling_window),
                   "outcome": outcome}
            rows.append(key | fit)
            if not marginal.empty:
                wide = dict(key)
                for value in marginal.itertuples():
                    prefix = "buy" if value.family == "margin_buy" else "sell"
                    wide[f"prior_return_{value.prior_level}"] = value.prior_value
                    wide[f"{prefix}_effect_{value.prior_level}"] = value.effect
                    wide[f"{prefix}_effect_{value.prior_level}_ci_low"] = value.ci_low
                    wide[f"{prefix}_effect_{value.prior_level}_ci_high"] = value.ci_high
                marginals.append(pd.DataFrame([wide]))
            if not col.empty:
                collinearity.append(col.assign(**key, condition_number=fit["condition_number"],
                                               status=fit["status"]))
    primary = pd.DataFrame(rows).reindex(columns=PRIMARY_RESULT_COLUMNS + (("turnover_beta", "turnover_p") if turnover_controls is not None else ()))
    return (primary,
            pd.concat(marginals, ignore_index=True) if marginals else pd.DataFrame(),
            pd.concat(collinearity, ignore_index=True) if collinearity else pd.DataFrame())


def _apply_separate_fdr(
    primary: pd.DataFrame, hypotheses: tuple[tuple[str, str, str], ...], scope: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for hypothesis, beta_col, p_col in hypotheses:
        part = primary[["k", "rolling_window", "outcome", "status", beta_col, p_col]].copy()
        part = part.rename(columns={beta_col: "beta", p_col: "raw_p_value"})
        part["hypothesis"] = hypothesis
        rows.append(part)
    fdr = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    fdr["family_fdr_q_value"] = np.nan
    for _, index in fdr.groupby("hypothesis").groups.items():
        use = fdr.loc[index].index[fdr.loc[index, "raw_p_value"].notna()]
        if len(use):
            fdr.loc[use, "family_fdr_q_value"] = _bh_adjust(fdr.loc[use, "raw_p_value"])
    fdr["global_fdr_q_value"] = np.nan
    use = fdr.index[fdr.raw_p_value.notna()]
    if len(use):
        fdr.loc[use, "global_fdr_q_value"] = _bh_adjust(fdr.loc[use, "raw_p_value"])
    fdr["evidence_level"] = "No Evidence"
    fdr.loc[fdr.raw_p_value.lt(.05), "evidence_level"] = "Level C"
    fdr.loc[fdr.family_fdr_q_value.lt(.05), "evidence_level"] = "Level B"
    fdr.loc[fdr.global_fdr_q_value.lt(.05), "evidence_level"] = "Level A"
    fdr.loc[fdr.raw_p_value.isna(), "evidence_level"] = "Not Testable"
    fdr["fdr_scope"] = scope
    levels = ("Level A", "Level B", "Level C", "No Evidence", "Not Testable")
    diagnostics = pd.DataFrame(
        [{"metric": "number_of_tests_total", "value": len(fdr)},
         {"metric": "number_of_tests_estimable", "value": int(fdr.raw_p_value.notna().sum())},
         {"metric": "number_of_tests_not_testable", "value": int(fdr.raw_p_value.isna().sum())}]
        + [{"metric": level, "value": int(fdr.evidence_level.eq(level).sum())} for level in levels]
    )
    diagnostics["fdr_scope"] = scope
    return fdr, diagnostics


def apply_joint_fdr(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interaction, interaction_diag = _apply_separate_fdr(primary, (
        ("buy_x_prior", "buy_prior_beta", "buy_prior_p"),
        ("sell_x_prior", "sell_prior_beta", "sell_prior_p"),
    ), INTERACTION_FDR_SCOPE)
    main, main_diag = _apply_separate_fdr(primary, (
        ("margin_buy", "buy_beta", "buy_p"),
        ("margin_sell", "sell_beta", "sell_p"),
    ), MAIN_EFFECT_FDR_SCOPE)
    return interaction, interaction_diag, main, main_diag


def _fit_nested(frame: pd.DataFrame, terms: list[str], config: JointFlowPriorReturnConfig) -> dict:
    design = pd.DataFrame({"const": 1.0}, index=frame.index)
    for term in terms:
        design[term] = frame[term]
    matrix = design.to_numpy(float)
    condition = np.linalg.cond(matrix)
    vifs = _vif(design)
    if (np.linalg.matrix_rank(matrix) < design.shape[1] or not np.isfinite(condition)
            or condition >= config.max_condition_number or max(vifs.values()) >= config.max_vif):
        return {"status": "unstable_collinearity"}
    fit = _hc3_ols(frame.future, design)
    leverage = np.asarray(fit["leverage"])
    if not np.isfinite(leverage).all() or np.any(leverage >= 1 - config.leverage_tolerance):
        return {"status": "unstable_collinearity"}
    return {"status": "estimated", "params": fit["params"], "pvalues": fit["pvalues"]}


def _absorption_classification(before: dict, after: dict, attenuation: float,
                               term: str, config: JointFlowPriorReturnConfig) -> str:
    if before.get("status") != "estimated" or after.get("status") != "estimated":
        return "unstable_collinearity"
    b, a = before["params"][term], after["params"][term]
    if np.sign(b) != np.sign(a):
        return "sign_reversal"
    if attenuation > config.absorption_large_threshold or (
        before["pvalues"][term] < config.significance_threshold
        and after["pvalues"][term] >= config.significance_threshold
    ):
        return "largely_absorbed"
    if attenuation >= config.absorption_partial_threshold:
        return "partially_absorbed"
    return "survives_joint_control"


def nested_model_comparison(
    signals: pd.DataFrame, prior: pd.Series, outcomes: pd.DataFrame,
    pairs: pd.DataFrame, config: JointFlowPriorReturnConfig | None = None,
) -> pd.DataFrame:
    config = config or JointFlowPriorReturnConfig()
    rows = []
    for spec in pairs.itertuples():
        for outcome in [c for c in OUTCOMES if c in outcomes]:
            frame = pd.concat({"future": outcomes[outcome], "buy": signals[spec.buy_signal],
                               "sell": signals[spec.sell_signal], "prior": prior}, axis=1).dropna()
            frame["buy_prior"] = frame.buy * frame.prior
            frame["sell_prior"] = frame.sell * frame.prior
            counts = _state_counts(frame)
            adequate = len(frame) >= 4 * config.min_joint_state_n and min(
                counts[n] for n in ("BothHigh_N", "BuyOnly_N", "SellOnly_N", "Neither_N")
            ) >= config.min_joint_state_n
            if not adequate:
                fits = {"A": {"status": "insufficient_group_sample"},
                        "B": {"status": "insufficient_group_sample"},
                        "C": {"status": "insufficient_group_sample"}}
            else:
                fits = {
                    "A": _fit_nested(frame, ["buy", "prior", "buy_prior"], config),
                    "B": _fit_nested(frame, ["sell", "prior", "sell_prior"], config),
                    "C": _fit_nested(frame, ["buy", "sell", "prior", "buy_prior", "sell_prior"], config),
                }
            for family, before_model, terms in (
                ("margin_buy", "A", ("buy", "buy_prior")),
                ("margin_sell", "B", ("sell", "sell_prior")),
            ):
                for term in terms:
                    before, after = fits[before_model], fits["C"]
                    b = before.get("params", {}).get(term, np.nan)
                    a = after.get("params", {}).get(term, np.nan)
                    attenuation = (1 - abs(a) / abs(b) if np.isfinite(a) and np.isfinite(b)
                                   and abs(b) > config.beta_epsilon else np.nan)
                    rows.append({
                        "family": family, "term": "main_effect" if term in ("buy", "sell") else "prior_interaction",
                        "k": int(spec.k), "rolling_window": int(spec.rolling_window),
                        "outcome": outcome, "before_model": before_model, "after_model": "C",
                        "coefficient_before": b,
                        "coefficient_after": a,
                        "p_before": before.get("pvalues", {}).get(term, np.nan),
                        "p_after": after.get("pvalues", {}).get(term, np.nan),
                        "attenuation_ratio": attenuation,
                        "sign_flip": bool(np.isfinite(a) and np.isfinite(b) and np.sign(a) != np.sign(b)),
                        "classification": _absorption_classification(before, after, attenuation, term, config),
                    })
    return pd.DataFrame(rows)


def turnover_comparison(primary: pd.DataFrame, controlled: pd.DataFrame,
                        config: JointFlowPriorReturnConfig | None = None) -> pd.DataFrame:
    config = config or JointFlowPriorReturnConfig()
    keys = ["k", "rolling_window", "outcome"]
    joined = primary.merge(controlled, on=keys, suffixes=("_before", "_after"))
    rows = []
    for row in joined.itertuples():
        for family, terms in (("margin_buy", ("buy_beta", "buy_prior_beta")),
                              ("margin_sell", ("sell_beta", "sell_prior_beta"))):
            for term in terms:
                before = getattr(row, f"{term}_before")
                after = getattr(row, f"{term}_after")
                p_name = term.replace("beta", "p")
                p_before, p_after = getattr(row, f"{p_name}_before"), getattr(row, f"{p_name}_after")
                attenuation = (1 - abs(after) / abs(before) if np.isfinite(before) and np.isfinite(after)
                               and abs(before) > config.beta_epsilon else np.nan)
                if getattr(row, "status_before") != "estimated" or getattr(row, "status_after") != "estimated":
                    classification = "unstable_collinearity"
                elif np.sign(before) != np.sign(after):
                    classification = "sign_flip"
                elif attenuation > config.absorption_large_threshold or (p_before < .05 <= p_after):
                    classification = "lose_significance" if p_before < .05 <= p_after else "attenuate"
                elif attenuation >= config.absorption_partial_threshold:
                    classification = "attenuate"
                else:
                    classification = "survive"
                rows.append({
                    "family": family, "term": term, "k": row.k,
                    "rolling_window": row.rolling_window, "outcome": row.outcome,
                    "coefficient_before": before, "coefficient_after": after,
                    "p_before": p_before, "p_after": p_after,
                    "attenuation_ratio": attenuation,
                    "sign_flip": bool(np.isfinite(before) and np.isfinite(after) and np.sign(before) != np.sign(after)),
                    "classification": classification,
                })
    return pd.DataFrame(rows)


def joint_event_clusters(signals: pd.DataFrame, prior: pd.Series,
                         pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in pairs.itertuples():
        for family, column in (("margin_buy", spec.buy_signal), ("margin_sell", spec.sell_signal)):
            active = signals[column].eq(1)
            positions = np.flatnonzero(active.to_numpy())
            clusters: list[list[int]] = []
            for position in positions:
                if not clusters or position != clusters[-1][-1] + 1:
                    clusters.append([int(position)])
                else:
                    clusters[-1].append(int(position))
            for members in clusters:
                first, last = members[0], members[-1]
                rows.append({
                    "family": family, "k": int(spec.k), "window": int(spec.rolling_window),
                    "signal_date": signals.index[first], "cluster_start": signals.index[first],
                    "cluster_end": signals.index[last], "cluster_length": len(members),
                    "BuyHigh": signals[spec.buy_signal].iloc[first],
                    "SellHigh": signals[spec.sell_signal].iloc[first],
                    "PriorReturn": prior.reindex(signals.index).iloc[first],
                    "daily_signal_N": len(positions), "independent_cluster_N": len(clusters),
                })
    return pd.DataFrame(rows)


def annual_joint_results(candidates: pd.DataFrame, signals: pd.DataFrame,
                         prior: pd.Series, outcomes: pd.DataFrame, pairs: pd.DataFrame,
                         config: JointFlowPriorReturnConfig | None = None
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or JointFlowPriorReturnConfig()
    if candidates.empty:
        return pd.DataFrame(columns=ANNUAL_COLUMNS), pd.DataFrame(columns=ANNUAL_SUMMARY_COLUMNS)
    rows = []
    pair_lookup = pairs.set_index(["k", "rolling_window"])
    for candidate in candidates[["hypothesis", "k", "rolling_window", "outcome"]].drop_duplicates().itertuples():
        spec = pair_lookup.loc[(candidate.k, candidate.rolling_window)]
        for year in sorted(set(outcomes.index.year)):
            mask = outcomes.index.year == year
            fit, _, _ = fit_joint_model(
                signals[spec.buy_signal].where(mask), signals[spec.sell_signal].where(mask),
                prior.where(mask), outcomes[candidate.outcome].where(mask), config,
            )
            beta = fit["buy_prior_beta" if candidate.hypothesis == "buy_x_prior" else "sell_prior_beta"]
            rows.append({
                "hypothesis": candidate.hypothesis, "k": candidate.k,
                "rolling_window": candidate.rolling_window, "outcome": candidate.outcome,
                "year": year, "sample_N": fit["N_total"], "interaction_beta": beta,
                "effect_direction": "positive" if beta > 0 else "negative" if beta < 0 else "missing",
                "status": fit["status"],
            })
    annual = pd.DataFrame(rows, columns=ANNUAL_COLUMNS)
    summaries = []
    keys = ["hypothesis", "k", "rolling_window", "outcome"]
    for key, group in annual.groupby(keys, dropna=False):
        usable = group[group.status.eq("estimated") & group.interaction_beta.notna()]
        target_sign = 1 if key[0] == "buy_x_prior" else -1
        same = int((np.sign(usable.interaction_beta) == target_sign).sum())
        years = len(usable)
        total = group.sample_N.sum()
        largest = group.sample_N.max() / total if total else np.nan
        consistency = same / years if years else np.nan
        robust = (years >= config.robustness_min_years
                  and consistency >= config.robustness_min_direction_ratio
                  and pd.notna(largest) and largest <= config.robustness_max_year_sample_share)
        summaries.append(dict(zip(keys, key)) | {
            "years_available": years, "same_direction_years": same,
            "direction_consistency": consistency, "largest_year_sample_share": largest,
            "annual_robustness": "robust" if robust else "insufficient_or_mixed",
        })
    return annual, pd.DataFrame(summaries, columns=ANNUAL_SUMMARY_COLUMNS)


def _create_run_directory(config: JointFlowPriorReturnConfig) -> Path:
    stamp = datetime.now(ZoneInfo(config.timezone)).strftime("%Y%m%d_%H%M%S")
    short = _git_value(["rev-parse", "--short", "HEAD"])
    run_dir = config.output_root / f"{stamp}_{short}_margin_buy_sell_prior_return_joint"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_run_info(run_dir: Path, directories: tuple[Path, Path, Path, Path],
                    config: JointFlowPriorReturnConfig) -> None:
    try:
        finlab_version = importlib.metadata.version("finlab")
    except importlib.metadata.PackageNotFoundError:
        finlab_version = "unknown"
    baseline, turnover, composition, regime = directories
    values = {
        "repository": config.repository, "branch": _git_value(["branch", "--show-current"]),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "baseline_run_dir": baseline, "baseline_commit": config.baseline_commit,
        "turnover_run_dir": turnover, "turnover_commit": config.turnover_commit,
        "composition_run_dir": composition, "composition_commit": config.composition_commit,
        "regime_run_dir": regime, "regime_commit": config.regime_commit,
        "price_source_open": "etl:adj_open", "price_source_close": "etl:adj_close",
        "outcome_price_adjusted": True,
        "prior_return_definition": "adjusted_close[t] / adjusted_close[t-5] - 1",
        "buy_signal_definition": "margin_buy amount_ratio PR95-100",
        "sell_signal_definition": "margin_sell amount_ratio PR95-100",
        "k": config.k_values, "rolling_windows": config.rolling_windows,
        "outcome_horizons": config.outcome_horizons,
        "interaction_fdr_scope": INTERACTION_FDR_SCOPE,
        "main_effect_fdr_scope": MAIN_EFFECT_FDR_SCOPE,
        "min_joint_state_n": config.min_joint_state_n,
        "max_condition_number": config.max_condition_number, "max_vif": config.max_vif,
        "timezone": config.timezone,
        "run_timestamp": datetime.now(ZoneInfo(config.timezone)).isoformat(),
        "python_version": platform.python_version(), "finlab_version": finlab_version,
    }
    (run_dir / "run_info_joint_flow_prior_return.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in values.items()) + "\n", encoding="utf-8"
    )


def _summary(result: dict[str, object]) -> str:
    interaction = result["interaction_fdr"]
    main = result["main_effect_fdr"]
    overlap = result["signal_overlap"]
    nested = result["nested_model_comparison"]
    turnover = result["turnover_controlled_results"]
    annual = result["annual_robustness_summary"]
    def evidence(frame: pd.DataFrame, hypothesis: str, sign: int | None = None) -> bool:
        chosen = frame[frame.hypothesis.eq(hypothesis) & frame.evidence_level.isin(["Level A", "Level B"])]
        return bool(len(chosen) and (sign is None or (np.sign(chosen.beta) == sign).any()))
    def hypothesis_status(frame: pd.DataFrame, hypothesis: str,
                          expected_sign: int | None = None) -> str:
        chosen = frame[frame.hypothesis.eq(hypothesis)]
        estimable = chosen[chosen.raw_p_value.notna()]
        if estimable.empty:
            return "Unable to Determine"
        direction = pd.Series(True, index=chosen.index) if expected_sign is None else np.sign(chosen.beta).eq(expected_sign)
        if (chosen.evidence_level.isin(["Level A", "Level B"]) & direction).any():
            return "Supported"
        if (chosen.evidence_level.eq("Level C") & direction).any():
            return "Partially Supported"
        return "Not Supported"
    buy_main, sell_main = evidence(main, "margin_buy"), evidence(main, "margin_sell")
    buy_int, sell_int = evidence(interaction, "buy_x_prior", 1), evidence(interaction, "sell_x_prior", -1)
    high_collinearity = bool(result["collinearity_diagnostics"].status.eq("unstable_collinearity").any()) if len(result["collinearity_diagnostics"]) else False
    both_share = overlap.Both_high_N.sum() / overlap.complete_binary_N.sum() if overlap.complete_binary_N.sum() else np.nan
    annual_note = ("No Level A/B interaction candidates; annual confirmatory analysis not applicable."
                   if not interaction.evidence_level.isin(["Level A", "Level B"]).any()
                   else f"Robust candidates: {int(annual.annual_robustness.eq('robust').sum())}.")
    h1 = hypothesis_status(main, "margin_buy")
    h2 = hypothesis_status(main, "margin_sell")
    h3 = hypothesis_status(interaction, "buy_x_prior", 1)
    h4 = hypothesis_status(interaction, "sell_x_prior", -1)
    if h1 == h2 == "Supported":
        h5 = "Supported"
    elif "Unable to Determine" in (h1, h2):
        h5 = "Unable to Determine"
    elif "Supported" in (h1, h2) or "Partially Supported" in (h1, h2):
        h5 = "Partially Supported"
    else:
        h5 = "Not Supported"
    hypothesis_table = (
        "| Hypothesis | Status |\n|---|---|\n"
        f"| H1 Buy conditional incremental information | {h1} |\n"
        f"| H2 Sell conditional incremental information | {h2} |\n"
        f"| H3 Buy × Prior Return > 0 | {h3} |\n"
        f"| H4 Sell × Prior Return < 0 | {h4} |\n"
        f"| H5 Buy/Sell are distinct flow dimensions | {h5} |"
    )
    any_estimable = bool(primary["status"].eq("estimated").any()) if len(primary := result["primary_results"]) else False
    overall = "保留" if all(value == "Supported" for value in (h1, h2, h3, h4)) else "修改後再測" if any_estimable else "淘汰"
    sections = {
        "A. 目前假設": "Buy 與 Sell 可能是兩個 flow dimensions；本研究不檢驗因果。\n\n" + hypothesis_table,
        "B. 市場機制": "Buy 可反映 leverage/risk-on，Sell 可反映 deleveraging/turnover；兩者也可能只是同一 latent Credit Activity。",
        "C. Input validation": "四層 frozen dependency chain 與 adjusted prices 必須全部 PASS，否則研究停止。",
        "D. Buy/Sell overlap": f"BothHigh aggregate share={both_share if np.isfinite(both_share) else 'NA'}；完整分窗結果見 CSV。",
        "E. 2×2 joint states": f"每個 state 預先要求至少 {result['config'].min_joint_state_n} 筆；不足即 Not Testable。",
        "F. Margin Buy incremental information": h1,
        "G. Margin Sell incremental information": h2,
        "H. Buy × Prior Return interaction": h3,
        "I. Sell × Prior Return interaction": h4,
        "J. Nested model absorption": "Model A/B 與 C 使用相同 complete cases；分類見 joint_nested_model_comparison.csv。",
        "K. Marginal effects": "只在 prior-return p25/p50/p75 呈現固定 marginal effects 與 95% CI，不搜尋門檻。",
        "L. FDR": f"Interaction scope={INTERACTION_FDR_SCOPE}；main-effect scope={MAIN_EFFECT_FDR_SCOPE}，兩者不混合。",
        "M. Collinearity": "存在不穩定模型。" if high_collinearity else "未觸發預先設定的 condition-number/VIF 門檻。",
        "N. Turnover robustness": "加入相同 k/window amount-ratio turnover 後的係數變化另列，不擴大 primary FDR。",
        "O. Event clustering": "連續 signal trading rows 合併為 cluster，並列 daily N 與 independent cluster N。",
        "P. Annual robustness": annual_note,
        "Q. 主要風險": "Look-ahead、selection/data snooping、overlapping outcomes、serial dependence、tail size、rolling-window dependence、collinearity、regime/turnover confounding、年度集中及缺乏真正 out-of-sample validation。",
        "R. 反對者觀點": "若 Buy/Sell 在 Model C 被彼此吸收，較支持同一 Credit Activity latent factor。",
        "S. 最終判定": overall,
    }
    questions = [
        f"1. Buy 控制 Sell/prior 後有 incremental information？{h1}",
        f"2. Sell 控制 Buy/prior 後有 incremental information？{h2}",
        f"3. Buy×Prior 穩定 >0？{h3}",
        f"4. Sell×Prior 穩定 <0？{h4}",
        f"5. Interaction 通過 FDR？{'Supported' if buy_int or sell_int else 'Not Supported' if interaction.raw_p_value.notna().any() else 'Unable to Determine'}",
        "6. 是否高度共線？見 overlap 與 collinearity CSV。",
        f"7. Buy/Sell 同時 high 比例？aggregate={both_share if np.isfinite(both_share) else 'NA'}。",
        "8. BothHigh 常見或罕見？依各 k/window 的完整樣本比例判讀。",
        "9. Model A→C Buy 被吸收多少？見 nested CSV。",
        "10. Model B→C Sell 被吸收多少？見 nested CSV。",
        "11. 加 Turnover 後是否仍存在？見 turnover-controlled CSV；不以 raw p 擴大 primary 結論。",
        "12. Prior return 解釋 occurrence 或改變 effect？只有 interaction FDR 可支持後者。",
        "13. 是否支持 Buy=risk-on、Sell=deleveraging/turnover？需要雙方 conditional evidence 與方向證據。",
        "14. 是否較支持同一 latent factor？若 joint control 吸收係數或共線性過高，該反論保留。",
        "15. 是否值得三階 interaction？僅雙方 survive 且 BothHigh 樣本充分時再考慮；本研究未估三階 interaction。",
    ]
    lines = ["# Margin Buy + Margin Sell + Prior 5-day Return Joint Model", ""]
    for heading, body in sections.items():
        lines += [f"## {heading}", "", body, ""]
    lines += ["## 最終必答問題", "", *questions, ""]
    return "\n".join(lines)


def run_margin_buy_sell_prior_return_joint_study(
    baseline_run_dir: str | Path, turnover_run_dir: str | Path,
    composition_run_dir: str | Path, regime_run_dir: str | Path,
    config: JointFlowPriorReturnConfig | None = None, provider=None,
    export: bool = True, datasets: dict[str, pd.DataFrame] | None = None,
) -> dict[str, object]:
    """Run only the frozen joint-flow increment; never rerun earlier studies."""
    config = config or JointFlowPriorReturnConfig()
    validated = validate_joint_input_runs(
        baseline_run_dir, turnover_run_dir, composition_run_dir, regime_run_dir, config
    )
    data = load_turnover_data(provider) if datasets is None else datasets
    universe, primary_symbols, limitation = build_universe_diagnostics(
        data["margin_balance"], data["market_value"]
    )
    pairs = select_joint_signal_pairs(validated["baseline"]["fdr_results"], config)
    signals, raw_ratios = build_joint_signals_and_ratios(data, pairs)
    prices = pd.concat({"open": data["open"][config.target_symbol],
                        "close": data["close"][config.target_symbol]}, axis=1).dropna().sort_index()
    signals = signals.reindex(prices.index)
    prior = prior_5d_return(prices.close)
    outcomes = build_outcomes(prices.open, prices.close, config.outcome_horizons)
    price_diagnostics = validate_split_window(prices.open, prices.close, outcomes)
    overlap = signal_overlap_diagnostics(signals, pairs, raw_ratios)
    states = joint_state_descriptive(signals, prior, outcomes, pairs)
    primary, marginal, collinearity = run_joint_models(signals, prior, outcomes, pairs, config)
    interaction_fdr, interaction_diag, main_fdr, main_diag = apply_joint_fdr(primary)
    nested = nested_model_comparison(signals, prior, outcomes, pairs, config)
    controls = build_turnover_controls(data, primary_symbols, pairs)
    controlled, _, controlled_collinearity = run_joint_models(
        signals, prior, outcomes, pairs, config, controls
    )
    turnover_results = turnover_comparison(primary, controlled, config)
    clusters = joint_event_clusters(signals, prior, pairs)
    candidates = interaction_fdr[interaction_fdr.evidence_level.isin(["Level A", "Level B"])]
    annual, annual_summary = annual_joint_results(
        candidates, signals, prior, outcomes, pairs, config
    )
    result: dict[str, object] = {
        "config": config, "input_validation": validated["diagnostics"],
        "baseline": validated["baseline"], "turnover": validated["turnover"],
        "composition": validated["composition"], "regime_input": validated["regime"],
        "universe": universe, "UNIVERSE_LIMITATION": limitation, "signal_pairs": pairs,
        "signals": signals, "prior_5d_return": prior, "outcomes": outcomes,
        "outcome_price_diagnostics": price_diagnostics, "signal_overlap": overlap,
        "state_descriptive": states, "primary_results": primary,
        "interaction_fdr": interaction_fdr, "interaction_fdr_diagnostics": interaction_diag,
        "main_effect_fdr": main_fdr, "main_effect_fdr_diagnostics": main_diag,
        "nested_model_comparison": nested, "marginal_effects": marginal,
        "collinearity_diagnostics": collinearity,
        "turnover_controlled_models": controlled,
        "turnover_controlled_results": turnover_results,
        "turnover_collinearity_diagnostics": controlled_collinearity,
        "event_clusters": clusters, "annual_results": annual,
        "annual_robustness_summary": annual_summary,
    }
    if export:
        run_dir = _create_run_directory(config)
        result["run_dir"] = run_dir
        outputs = {
            "joint_input_validation.csv": validated["diagnostics"],
            "joint_signal_overlap.csv": overlap,
            "joint_state_descriptive.csv": states,
            "joint_primary_results.csv": primary,
            "joint_interaction_fdr_results.csv": interaction_fdr,
            "joint_interaction_fdr_diagnostics.csv": interaction_diag,
            "joint_main_effect_fdr_results.csv": main_fdr,
            "joint_main_effect_fdr_diagnostics.csv": main_diag,
            "joint_nested_model_comparison.csv": nested,
            "joint_marginal_effects.csv": marginal,
            "joint_collinearity_diagnostics.csv": collinearity,
            "joint_turnover_controlled_results.csv": turnover_results,
            "joint_event_clusters.csv": clusters,
            "joint_annual_results.csv": annual,
            "joint_annual_robustness_summary.csv": annual_summary,
            "outcome_price_diagnostics.csv": price_diagnostics,
        }
        for filename, frame in outputs.items():
            frame.to_csv(run_dir / filename, index=False)
        (run_dir / "joint_summary.md").write_text(_summary(result), encoding="utf-8")
        _write_run_info(
            run_dir,
            tuple(map(Path, (baseline_run_dir, turnover_run_dir, composition_run_dir, regime_run_dir))),
            config,
        )
    return result
