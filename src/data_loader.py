from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from pandas.core.frame import DataFrame as NativeDataFrame

from .config import FINLAB_FIELDS


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
    for alias, field in FINLAB_FIELDS.items():
        try:
            loaded[alias] = _as_datetime_frame(provider.get(field), field)
        except Exception as exc:
            raise RuntimeError(f"Failed loading exact FinLab dataset {field!r}") from exc
    validate_required_columns(loaded)
    return loaded


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
