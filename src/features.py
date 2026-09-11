from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_ratio(numerator: pd.Series, denominator: pd.Series, k: int) -> pd.Series:
    num = numerator.rolling(k, min_periods=k).sum()
    den = denominator.rolling(k, min_periods=k).sum().replace(0, np.nan)
    return num / den


def trailing_percentile(series: pd.Series, window: int) -> pd.Series:
    """Rank each observed value against the last ``window`` valid observations.

    Missing dates remain missing.  They are neither forward-filled nor counted as
    observations, which prevents an intermittent missing value from invalidating
    every later row while preserving the full-window requirement.
    """
    observed = series.replace([np.inf, -np.inf], np.nan).dropna()
    ranked = observed.rolling(window, min_periods=window).apply(
        lambda x: 100.0 * (np.count_nonzero(x[:-1] <= x[-1]) + 1) / len(x), raw=True
    )
    return ranked.reindex(series.index)


def pr_group(percentile: pd.Series) -> pd.Categorical:
    """Seven non-overlapping descriptive bins.

    Bins are left-closed/right-open, except that 100 belongs to PR95-100.
    Primary inference remains the separately defined <=5 and >=95 tails.
    """
    values = pd.to_numeric(percentile, errors="coerce").astype(float).clip(lower=0, upper=100)
    values = values.mask(values.eq(100), np.nextafter(100.0, 0.0))
    return pd.cut(
        values,
        bins=[0, 5, 20, 40, 60, 80, 95, 100],
        labels=["PR0-5", "PR5-20", "PR20-40", "PR40-60", "PR60-80", "PR80-95", "PR95-100"],
        right=False,
        include_lowest=True,
    )


def kday_change(series: pd.Series, k: int) -> pd.Series:
    return series / series.shift(k) - 1


def position_market_value(balance_lots: pd.DataFrame, close: pd.DataFrame, symbols: set[str] | None = None) -> pd.Series:
    cols = sorted((set(balance_lots.columns) & set(close.columns)) if symbols is None else (symbols & set(balance_lots.columns) & set(close.columns)))
    if not cols:
        raise ValueError("No common symbols for position market value")
    balances, prices = balance_lots[cols].align(close[cols], join="inner", axis=0)
    return (balances * 1000.0 * prices).sum(axis=1, min_count=1)


def adjust_short_for_suspensions(frames: dict[str, pd.DataFrame], suspension: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Mask observed event intervals inclusively on the existing trading calendar."""
    affected = pd.DataFrame(False, index=frames["short_balance"].index, columns=frames["short_balance"].columns)
    for symbol, start, end in suspension[["symbol", "停券起日(最後回補日)", "停券迄日"]].itertuples(index=False, name=None):
        if symbol in affected.columns:
            end = start if pd.isna(end) else end
            affected.loc[(affected.index >= start) & (affected.index <= end), symbol] = True
    adjusted = {}
    for key in ("short_sell", "short_cover", "short_stock_repayment", "short_balance"):
        adjusted[key] = frames[key].mask(affected)
    return adjusted, affected


def adjusted_short_change(balance: pd.DataFrame, affected: pd.DataFrame, k: int) -> pd.Series:
    """Use the same eligible securities at both endpoints to avoid mask turnover."""
    eligible = affected.astype(int).rolling(k + 1, min_periods=k + 1).sum().eq(0)
    previous = balance.shift(k)
    eligible &= balance.notna() & previous.notna()
    now = balance.where(eligible).sum(axis=1, min_count=1)
    before = previous.where(eligible).sum(axis=1, min_count=1)
    return now / before.replace(0, np.nan) - 1


LEVEL_FAMILIES = (
    "margin_position_level",
    "margin_credit_level",
    "approx_margin_maintenance",
    "short_position_level",
    "short_position_level_adjusted",
    "short_margin_ratio",
)


def level_feature_diagnostics(
    base_series: dict[str, pd.Series],
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Trace level predictors from raw coverage through tails and outcomes."""
    joined = features.join(outcomes, how="inner")
    outcome_cols = [column for column in outcomes if column.startswith("O1_C")]
    rows = []
    catalog = features.attrs.get("catalog", {})
    components = {
        "margin_position_level": ("margin_position_market_value", "market_cap"),
        "margin_credit_level": ("margin_credit_amount", "market_cap"),
        "approx_margin_maintenance": ("margin_position_market_value", "margin_credit_amount"),
        "short_position_level": ("short_position_market_value", "market_cap"),
        "short_position_level_adjusted": ("short_position_market_value_adjusted", "market_cap"),
        "short_margin_ratio": ("short_position_market_value", "margin_position_market_value"),
    }
    for feature, meta in catalog.items():
        family = next((name for name in LEVEL_FAMILIES if feature.startswith(f"{name}__")), None)
        if family is None:
            continue
        raw = base_series[family].replace([np.inf, -np.inf], np.nan)
        numerator_name, denominator_name = components[family]
        numerator = base_series[numerator_name].replace([np.inf, -np.inf], np.nan)
        denominator = base_series[denominator_name].replace([np.inf, -np.inf], np.nan)
        numerator, denominator = numerator.align(denominator, join="inner")
        percentile = features[feature].replace([np.inf, -np.inf], np.nan)
        common = joined[[feature, *outcome_cols]].dropna(subset=[feature])
        common_dates = common.index[common[outcome_cols].notna().any(axis=1)]
        rows.append({
            "feature": feature,
            "family": meta.get("family", family),
            "numerator_first_date": numerator.first_valid_index(),
            "numerator_last_date": numerator.last_valid_index(),
            "numerator_non_null_n": int(numerator.notna().sum()),
            "denominator_first_date": denominator.first_valid_index(),
            "denominator_last_date": denominator.last_valid_index(),
            "denominator_non_null_n": int(denominator.notna().sum()),
            "numerator_denominator_common_n": int((numerator.notna() & denominator.notna()).sum()),
            "raw_first_date": raw.first_valid_index(),
            "raw_last_date": raw.last_valid_index(),
            "raw_non_null_n": int(raw.notna().sum()),
            "raw_finite_n": int(np.isfinite(raw.dropna()).sum()),
            "percentile_first_date": percentile.first_valid_index(),
            "percentile_last_date": percentile.last_valid_index(),
            "percentile_non_null_n": int(percentile.notna().sum()),
            "outcome_common_date_n": int(len(common_dates)),
            "pr_low_n": int(joined.loc[common_dates, feature].le(5).sum()),
            "pr_high_n": int(joined.loc[common_dates, feature].ge(95).sum()),
            "status": "ok" if len(common_dates) and joined.loc[common_dates, feature].le(5).any() and joined.loc[common_dates, feature].ge(95).any() else "insufficient_tail_coverage",
        })
    return pd.DataFrame(rows)


def build_feature_catalog() -> pd.DataFrame:
    families = {
        "margin_buy": "raw, amount ratio, volume ratio", "margin_sell": "raw, amount ratio, volume ratio",
        "margin_cash_repayment": "raw, volume ratio", "margin_balance_change": "percentage change",
        "margin_position_level": "position market value / market cap", "margin_credit_level": "credit / market cap",
        "approx_margin_maintenance": "position market value / credit", "short_sell": "raw and adjusted",
        "short_cover": "raw and adjusted", "short_repayment": "raw and adjusted",
        "short_balance_change": "percentage change; raw and adjusted", "short_position_level": "market value / market cap",
        "short_margin_ratio": "short position market value / margin position market value",
    }
    return pd.DataFrame([{"family": k, "definition": v} for k, v in families.items()])


def build_level_features(base_series: dict[str, pd.Series], config) -> pd.DataFrame:
    """Build only level-family percentiles for universe sensitivity work."""
    values: dict[str, pd.Series] = {}
    catalog: dict[str, dict] = {}
    for family in LEVEL_FAMILIES:
        for window in config.rolling_windows:
            col = f"{family}__w{window}_percentile"
            values[col] = trailing_percentile(base_series[family], window)
            catalog[col] = {
                "family": family.replace("_adjusted", ""),
                "k": 1,
                "window": window,
                "variant": family,
            }
    frame = pd.DataFrame(values).sort_index()
    frame.attrs["catalog"] = catalog
    return frame
