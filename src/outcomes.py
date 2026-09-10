from __future__ import annotations

import pandas as pd


def build_outcomes(open_0050: pd.Series, close_0050: pd.Series, horizons=(1, 2, 3, 5, 10, 20)) -> pd.DataFrame:
    prices = pd.concat({"open": open_0050, "close": close_0050}, axis=1).dropna().sort_index()
    # Row indexed by d0: shift(-1) is O1, shift(-h) is C_h.
    out = pd.DataFrame(index=prices.index)
    out["entry_date"] = prices.index.to_series().shift(-1)
    entry = prices["open"].shift(-1)
    for h in horizons:
        out[f"O1_C{h}"] = prices["close"].shift(-h) / entry - 1
    out["C0_O1_descriptive_nontradable"] = entry / prices["close"] - 1
    return out
