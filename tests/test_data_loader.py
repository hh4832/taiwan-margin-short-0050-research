import unittest

import pandas as pd

from src.data_loader import _as_datetime_frame


class ProtectedIndexFrame(pd.DataFrame):
    """Minimal stand-in for FinlabDataFrame's protected index behaviour."""

    @property
    def _constructor(self):
        return ProtectedIndexFrame

    def __setattr__(self, name, value):
        if name == "index":
            raise AttributeError("direct index assignment is protected")
        super().__setattr__(name, value)


class DataLoaderTests(unittest.TestCase):
    def test_converts_dataframe_subclass_before_index_normalization(self):
        source = ProtectedIndexFrame({"0050": [1.0, 2.0]}, index=["2026-09-09", "2026-09-10"])
        result = _as_datetime_frame(source, "price:開盤價")

        self.assertIs(type(result), pd.DataFrame)
        self.assertIsInstance(result.index, pd.DatetimeIndex)
        self.assertEqual(result.index[-1], pd.Timestamp("2026-09-10"))
        self.assertIsInstance(source, ProtectedIndexFrame)
