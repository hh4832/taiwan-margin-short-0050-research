from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_ratio(numerator: pd.Series, denominator: pd.Series, k: int) -> pd.Series:
    num = numerator.rolling(k, min_periods=k).sum()
    den = denominator.rolling(k, min_periods=k).sum().replace(0, np.nan)
    return num / den


def trailing_percentile(series: pd.Series, window: int) -> pd.Series:
    """Percentile rank of current value within trailing window, including current."""
    return series.rolling(window, min_periods=window).apply(
        lambda x: 100.0 * (np.count_nonzero(x[:-1] <= x[-1]) + 1) / len(x), raw=True
    )


def pr_group(percentile: pd.Series) -> pd.Categorical:
    return pd.cut(percentile, bins=[0, 5, 20, 40, 60, 80, 95, 100], labels=["PR0-5", "PR5-20", "PR20-40", "PR40-60", "PR60-80", "PR80-95", "PR95-100"], include_lowest=True)


def kday_change(series: pd.Series, k: int) -> pd.Series:
    return series / series.shift(k) - 1


def position_market_value(balance_lots: pd.DataFrame, close: pd.DataFrame, symbols: set[str] | None = None) -> pd.Series:
    cols = sorted((set(balance_lots.columns) & set(close.columns)) if symbols is None else (symbols & set(balance_lots.columns) & set(close.columns)))
    if not cols:
        raise ValueError("No common symbols for position market value")
    balances, prices = balance_lots[cols].align(close[cols], join="inner", axis=0)
    return (balances * 1000.0 * prices).sum(axis=1, min_count=1)


def adjust_short_for_suspensions(frames: dict[str, pd.DataFrame], suspension: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Mask known final-cover dates only; never infer an unobserved suspension interval."""
    affected = pd.DataFrame(False, index=frames["short_balance"].index, columns=frames["short_balance"].columns)
    if isinstance(suspension, pd.DataFrame):
        if set(suspension.columns) & set(affected.columns):
            marker = suspension.reindex(index=affected.index, columns=affected.columns).notna()
            affected |= marker
        else:
            for dt in suspension.index.intersection(affected.index):
                values = suspension.loc[dt].dropna().astype(str)
                for symbol in values:
                    if symbol in affected.columns:
                        affected.loc[dt, symbol] = True
    adjusted = {}
    for key in ("short_sell", "short_cover", "short_stock_repayment", "short_balance"):
        adjusted[key] = frames[key].mask(affected)
    return adjusted, affected


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
