from __future__ import annotations

import re
import pandas as pd


COMMON_EQUITY_PATTERN = re.compile(r"^[1-9][0-9]{3}$")


def build_universe_diagnostics(margin: pd.DataFrame, market_value: pd.DataFrame) -> tuple[pd.DataFrame, set[str], bool]:
    margin_symbols = set(map(str, margin.columns))
    mv_symbols = set(map(str, market_value.columns))
    common = margin_symbols & mv_symbols
    primary = {s for s in common if COMMON_EQUITY_PATTERN.fullmatch(s)}
    excluded = {s for s in margin_symbols if not COMMON_EQUITY_PATTERN.fullmatch(s)}
    rows = []
    for category, symbols in {
        "margin_symbols": margin_symbols,
        "market_value_symbols": mv_symbols,
        "common_symbols": common,
        "intersection": common,
        "primary_common_equity": primary,
        "all_available_margin_securities": common,
        "only_in_margin": margin_symbols - mv_symbols,
        "only_in_market_value": mv_symbols - margin_symbols,
        "excluded_etf_like_or_non_common": excluded,
        "excluded_from_primary": margin_symbols - primary,
    }.items():
        rows.append({"category": category, "count": len(symbols), "symbols": ",".join(sorted(symbols))})
    # A ticker-pattern filter is not a point-in-time security master.
    return pd.DataFrame(rows), primary, True
