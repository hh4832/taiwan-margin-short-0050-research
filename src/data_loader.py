from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd
from pandas.core.frame import DataFrame as NativeDataFrame

from .config import FINLAB_FIELDS


SUSPENSION_START = "停券起日(最後回補日)"
SUSPENSION_END = "停券迄日"


@dataclass(frozen=True)
class SuspensionParseResult:
    valid_events: pd.DataFrame
    invalid_events: pd.DataFrame
    diagnostics: pd.DataFrame


def _as_datetime_frame(value: object, name: str) -> pd.DataFrame:
    if not isinstance(value, (pd.DataFrame, pd.Series)):
        raise TypeError(f"{name} must be a pandas object, got {type(value).__name__}")
    # FinlabDataFrame.copy() intentionally preserves its subclass and blocks
    # direct index assignment. These inputs are daily market datasets, so make
    # the conversion to ordinary pandas semantics explicit before normalising
    # their already date-like index. Do not use this helper for financial
    # statement period indexes, which require FinLab deadline semantics.
    source = value.to_frame() if isinstance(value, pd.Series) else value
    # Construct from an ndarray rather than from the FinLab object/manager.
    # `pd.DataFrame(source)` is not sufficient with current FinLab on Python
    # 3.13 because the protected subclass can survive that constructor.
    values = source.to_numpy(copy=True)
    index = pd.to_datetime(source.index.to_numpy(copy=True))
    columns = source.columns.copy()
    order = index.argsort()
    frame = NativeDataFrame(values[order], index=index.take(order), columns=columns)
    if type(frame) is not NativeDataFrame:
        raise TypeError(
            f"{name} could not be detached from {type(value).__name__}; "
            f"rebuilt type is {type(frame).__name__}"
        )
    if frame.index.has_duplicates:
        raise ValueError(f"{name} contains duplicate dates")
    return frame


def load_finlab_data(provider=None) -> dict[str, pd.DataFrame]:
    """Load only the exact FinLab field names declared in the protocol."""
    if provider is None:
        from finlab import data as provider
    loaded: dict[str, pd.DataFrame] = {}
    suspension_result: SuspensionParseResult | None = None
    for alias, field in FINLAB_FIELDS.items():
        try:
            raw = provider.get(field)
            if alias == "short_suspension":
                suspension_result = parse_suspension_events(raw)
                loaded[alias] = suspension_result.valid_events
            else:
                loaded[alias] = _as_datetime_frame(raw, field)
        except Exception as exc:
            raise RuntimeError(f"Failed loading exact FinLab dataset {field!r}: {exc}") from exc
    validate_required_columns(loaded)
    if suspension_result is not None:
        loaded["short_suspension_invalid"] = suspension_result.invalid_events
        loaded["short_suspension_validation"] = suspension_result.diagnostics
    return loaded


def parse_suspension_events(value: pd.DataFrame) -> SuspensionParseResult:
    """Split FinLab suspension event rows into usable events and a quarantine."""
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"Suspension events must be a DataFrame, got {type(value).__name__}")
    required = {"symbol", SUSPENSION_START, SUSPENSION_END}
    if not required.issubset(value.columns):
        raise ValueError(f"Suspension event columns missing: {required - set(value.columns)}")

    # Rebuild column by column so a FinlabDataFrame subclass cannot retain its
    # protected index behaviour. Keep an untouched copy for the quarantine.
    raw = NativeDataFrame({c: value[c].array.copy() for c in value.columns})
    raw.insert(0, "source_index", value.index.to_numpy(copy=True))
    raw.insert(0, "source_row", range(len(value)))
    symbol = raw["symbol"].astype("string").str.strip()
    start = pd.to_datetime(raw[SUSPENSION_START], errors="coerce").dt.normalize()
    end = pd.to_datetime(raw[SUSPENSION_END], errors="coerce").dt.normalize()

    source_start_missing = raw[SUSPENSION_START].isna()
    source_end_missing = raw[SUSPENSION_END].isna()
    missing_symbol = symbol.isna() | symbol.eq("")
    missing_start = source_start_missing
    invalid_start_format = ~source_start_missing & start.isna()
    invalid_end_format = ~source_end_missing & end.isna()
    end_before_start = start.notna() & end.notna() & end.lt(start)
    reason_masks = {
        "missing_symbol": missing_symbol,
        "missing_start": missing_start,
        "invalid_start_format": invalid_start_format,
        "invalid_end_format": invalid_end_format,
        "end_before_start": end_before_start,
    }
    invalid = pd.concat(reason_masks, axis=1).any(axis=1)

    invalid_events = raw.loc[invalid].copy()
    invalid_events["parsed_start"] = start.loc[invalid].to_numpy()
    invalid_events["parsed_end"] = end.loc[invalid].to_numpy()
    invalid_events["validation_error"] = [
        "|".join(name for name, mask in reason_masks.items() if bool(mask.iloc[i]))
        for i in invalid_events.index
    ]

    valid_events = raw.loc[~invalid].drop(columns=["source_index"]).copy()
    valid_events["symbol"] = symbol.loc[~invalid].to_numpy()
    valid_events[SUSPENSION_START] = start.loc[~invalid].to_numpy()
    valid_events[SUSPENSION_END] = end.loc[~invalid].to_numpy()
    # Duplicate dates across symbols/events are legitimate.
    valid_events.index = pd.DatetimeIndex(valid_events[SUSPENSION_START], name="event_date")

    invalid_dates = pd.concat([start.loc[invalid], end.loc[invalid]]).dropna()
    affected = sorted(set(symbol.loc[invalid].dropna()) - {""})
    diagnostics = NativeDataFrame([{
        "total_rows": int(len(raw)),
        "valid_rows": int((~invalid).sum()),
        "invalid_rows": int(invalid.sum()),
        "invalid_rate": float(invalid.mean()) if len(raw) else 0.0,
        "missing_start": int(missing_start.sum()),
        "end_before_start": int(end_before_start.sum()),
        "missing_symbol": int(missing_symbol.sum()),
        "invalid_start_format": int(invalid_start_format.sum()),
        "invalid_end_format": int(invalid_end_format.sum()),
        "affected_symbol_count": len(affected),
        "affected_symbols": ",".join(affected),
        "earliest_valid_date": start.loc[~invalid].min(),
        "latest_valid_date": start.loc[~invalid].max(),
        "earliest_invalid_date": invalid_dates.min() if len(invalid_dates) else pd.NaT,
        "latest_invalid_date": invalid_dates.max() if len(invalid_dates) else pd.NaT,
        "SUSPENSION_COVERAGE_LIMITATION": bool(invalid.any()),
    }])
    limitation = (
        "Retrospective sensitivity; key_date is not a verified historical announcement date. "
        f"Quarantined {int(invalid.sum())} invalid source rows; adjusted coverage is incomplete."
        if invalid.any()
        else "Retrospective sensitivity; key_date is not a verified historical announcement date."
    )
    valid_events.attrs["limitation"] = limitation
    valid_events.attrs["quarantined_rows"] = int(invalid.sum())
    return SuspensionParseResult(valid_events.sort_index(), invalid_events, diagnostics)


def validate_required_columns(datasets: Mapping[str, pd.DataFrame]) -> None:
    for key in ("open", "close"):
        if "0050" not in datasets[key].columns.astype(str):
            raise ValueError(f"0050 missing from {FINLAB_FIELDS[key]}; stopping")
    for key in ("market_volume", "market_amount", "market_count", "market_index"):
        missing = {"TAIEX", "OTC"} - set(datasets[key].columns)
        if missing:
            raise ValueError(f"{FINLAB_FIELDS[key]} missing columns: {sorted(missing)}")
    required_aggregate = {
        "aggregate_buy": {"上市融資交易張數", "上市融資交易金額", "上櫃融資交易張數", "上櫃融資交易金額"},
        "aggregate_sell": {"上市融資交易張數", "上市融資交易金額", "上櫃融資交易張數", "上櫃融資交易金額"},
        "aggregate_balance": {"上市融券交易張數", "上市融資交易張數", "上市融資交易金額", "上櫃融券交易張數", "上櫃融資交易張數", "上櫃融資交易金額"},
    }
    for key, expected in required_aggregate.items():
        missing = expected - set(datasets[key].columns)
        if missing:
            raise ValueError(f"{FINLAB_FIELDS[key]} missing known columns: {sorted(missing)}")
