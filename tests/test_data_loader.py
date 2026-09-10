import unittest

import pandas as pd
from pandas.core.frame import DataFrame as NativeDataFrame

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

    def to_numpy(self, *args, **kwargs):
        object.__setattr__(self, "to_numpy_called", True)
        return super().to_numpy(*args, **kwargs)


class DataLoaderTests(unittest.TestCase):
    def test_converts_dataframe_subclass_before_index_normalization(self):
        source = ProtectedIndexFrame({"0050": [1.0, 2.0]}, index=["2026-09-09", "2026-09-10"])
        result = _as_datetime_frame(source, "price:開盤價")

        self.assertIs(type(result), NativeDataFrame)
        self.assertIsInstance(result.index, pd.DatetimeIndex)
        self.assertEqual(result.index[-1], pd.Timestamp("2026-09-10"))
        self.assertIsInstance(source, ProtectedIndexFrame)
        self.assertTrue(source.to_numpy_called)

    def test_sorts_during_native_reconstruction_without_index_assignment(self):
        source = ProtectedIndexFrame({"0050": [2.0, 1.0]}, index=["2026-09-10", "2026-09-09"])
        result = _as_datetime_frame(source, "price:開盤價")

        self.assertEqual(result.index.tolist(), [pd.Timestamp("2026-09-09"), pd.Timestamp("2026-09-10")])
        self.assertEqual(result["0050"].tolist(), [1.0, 2.0])
