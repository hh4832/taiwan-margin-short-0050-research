import unittest
from unittest.mock import patch
import sys
import types

import pandas as pd

from src.data_loader import load_finlab_data, parse_suspension_events
from src.diagnostics import dataset_coverage
from src.features import adjust_short_for_suspensions, adjusted_short_change
from src.statistics import controlled_regression
from src.universe import build_universe_diagnostics
from src.pipeline import run, run_stage_zero


def suspension_rows():
    """Minimal reproduction of FinLab's event-table schema and invalid classes."""
    return pd.DataFrame({
        "symbol": pd.Series(["0050", "2330", "1101"], dtype="category"),
        "停券起日(最後回補日)": pd.to_datetime(["2026-09-04", None, "2026-09-07"]),
        "證券名稱": ["元大台灣50", "台積電", "台泥"],
        "停券迄日": pd.to_datetime(["2026-09-07", "2026-09-08", "2026-09-04"]),
        "原因": ["分配收益", "股東常會", "除息"],
        "key_date": pd.to_datetime(["2026-01-01"] * 3),
        "stock_id": pd.Series(["0050", "2330", "1101"], dtype="category"),
    })


class SuspensionTests(unittest.TestCase):
    def test_schema_dtype_index_and_valid_event(self):
        raw = suspension_rows().iloc[[0]].reset_index(drop=True)
        parsed = parse_suspension_events(raw)
        self.assertIsInstance(raw.index, pd.RangeIndex)
        self.assertEqual(raw.columns.tolist(), [
            "symbol", "停券起日(最後回補日)", "證券名稱", "停券迄日", "原因", "key_date", "stock_id",
        ])
        self.assertEqual(str(raw["symbol"].dtype), "category")
        self.assertEqual(str(raw["停券起日(最後回補日)"].dtype), "datetime64[ns]")
        self.assertEqual(len(parsed.valid_events), 1)
        self.assertTrue(parsed.invalid_events.empty)
        self.assertEqual(parsed.valid_events.index[0], pd.Timestamp("2026-09-04"))
        self.assertFalse(parsed.diagnostics.iloc[0]["SUSPENSION_COVERAGE_LIMITATION"])

    def test_reported_39169_row_distribution_is_quarantined(self):
        n = 39169
        raw = pd.DataFrame({
            "symbol": pd.Series([f"S{i:05d}" for i in range(n)], dtype="category"),
            "停券起日(最後回補日)": pd.to_datetime(["2020-01-01"] * n),
            "證券名稱": ["測試"] * n,
            "停券迄日": pd.to_datetime(["2020-01-02"] * n),
            "原因": ["測試"] * n,
            "key_date": pd.to_datetime(["2026-01-01"] * n),
            "stock_id": pd.Series([f"S{i:05d}" for i in range(n)], dtype="category"),
        })
        raw.loc[:27, "停券起日(最後回補日)"] = pd.NaT
        raw.loc[28:29, "停券起日(最後回補日)"] = pd.Timestamp("2020-01-03")
        parsed = parse_suspension_events(raw)
        diag = parsed.diagnostics.iloc[0]
        self.assertEqual(diag["total_rows"], 39169)
        self.assertEqual(diag["valid_rows"], 39139)
        self.assertEqual(diag["invalid_rows"], 30)
        self.assertEqual(diag["missing_start"], 28)
        self.assertEqual(diag["end_before_start"], 2)
        self.assertAlmostEqual(diag["invalid_rate"], 30 / 39169)
        self.assertEqual(len(parsed.invalid_events), 30)
        self.assertTrue(diag["SUSPENSION_COVERAGE_LIMITATION"])

    def test_missing_start_and_reversed_interval_keep_original_rows(self):
        raw = suspension_rows()
        parsed = parse_suspension_events(raw)
        self.assertEqual(parsed.invalid_events["source_row"].tolist(), [1, 2])
        self.assertEqual(parsed.invalid_events["validation_error"].tolist(), ["missing_start", "end_before_start"])
        self.assertTrue(pd.isna(parsed.invalid_events.iloc[0]["停券起日(最後回補日)"]))
        self.assertEqual(parsed.invalid_events.iloc[1]["停券起日(最後回補日)"], pd.Timestamp("2026-09-07"))
        self.assertEqual(str(parsed.invalid_events["symbol"].dtype), "category")

    def test_additional_unusable_values_are_quarantined(self):
        raw = pd.DataFrame({
            "symbol": ["", "2330", "1101"],
            "停券起日(最後回補日)": ["2026-01-01", "not-a-date", "2026-01-01"],
            "停券迄日": ["2026-01-02", "2026-01-02", "not-a-date"],
        })
        parsed = parse_suspension_events(raw)
        diag = parsed.diagnostics.iloc[0]
        self.assertEqual(diag["missing_symbol"], 1)
        self.assertEqual(diag["invalid_start_format"], 1)
        self.assertEqual(diag["invalid_end_format"], 1)
        self.assertEqual(diag["invalid_rows"], 3)

    def test_invalid_rows_do_not_crash_data_loading(self):
        class Provider:
            @staticmethod
            def get(_):
                return suspension_rows()
        with (
            patch("src.data_loader.FINLAB_FIELDS", {"short_suspension": "margin_short_sale_suspension"}),
            patch("src.data_loader.validate_required_columns"),
        ):
            loaded = load_finlab_data(Provider())
        self.assertEqual(len(loaded["short_suspension"]), 1)
        self.assertEqual(len(loaded["short_suspension_invalid"]), 2)
        self.assertEqual(loaded["short_suspension_validation"].iloc[0]["invalid_rows"], 2)

    def test_quarantine_does_not_crash_stage_zero_or_change_raw_short(self):
        idx = pd.date_range("2024-01-01", periods=2)
        security = lambda values: pd.DataFrame({"2330": values}, index=idx)
        market = pd.DataFrame({"TAIEX": [100.0, 100.0], "OTC": [50.0, 50.0]}, index=idx)
        parsed = parse_suspension_events(suspension_rows())
        parsed.valid_events.attrs["quarantined_rows"] = 2
        aggregate_balance = pd.DataFrame({
            "上市融資交易張數": [12.0, 9.0], "上櫃融資交易張數": [0.0, 0.0],
            "上市融資交易金額": [100.0, 100.0], "上櫃融資交易金額": [0.0, 0.0],
            "上市融券交易張數": [7.0, 4.0], "上櫃融券交易張數": [0.0, 0.0],
        }, index=idx)
        datasets = {
            "open": pd.DataFrame({"0050": [10.0, 11.0]}, index=idx),
            "close": pd.DataFrame({"0050": [10.5, 11.5], "2330": [100.0, 101.0]}, index=idx),
            "margin_prev_balance": security([10.0, 12.0]),
            "margin_balance": security([12.0, 9.0]),
            "margin_buy": security([4.0, 1.0]),
            "margin_sell": security([1.0, 3.0]),
            "margin_cash_repayment": security([1.0, 1.0]),
            "short_prev_balance": security([5.0, 7.0]),
            "short_balance": security([7.0, 4.0]),
            "short_sell": security([4.0, 1.0]),
            "short_cover": security([1.0, 3.0]),
            "short_stock_repayment": security([1.0, 1.0]),
            "market_amount": market,
            "market_volume": market,
            "market_value": security([1_000_000.0, 1_010_000.0]),
            "aggregate_balance": aggregate_balance,
            "short_suspension": parsed.valid_events,
            "short_suspension_invalid": parsed.invalid_events,
            "short_suspension_validation": parsed.diagnostics,
        }
        with patch("src.pipeline.load_finlab_data", return_value=datasets):
            stage = run_stage_zero()
        self.assertEqual(stage["base_series"]["short_sell_raw"].tolist(), [4.0, 1.0])
        self.assertEqual(len(stage["suspension_invalid_events"]), 2)
        self.assertTrue(stage["suspension"]["SUSPENSION_COVERAGE_LIMITATION"].all())
        self.assertTrue((stage["reconciliation"].iloc[:2]["exact_match_ratio"] == 1.0).all())

        fake_fdr = types.ModuleType("src.fdr")
        fake_fdr.apply_fdr = lambda frame: frame
        fake_fdr.fdr_diagnostics = lambda frame: pd.DataFrame()
        empty = pd.DataFrame()
        with (
            patch.dict(sys.modules, {"src.fdr": fake_fdr}),
            patch("src.pipeline.run_stage_zero", return_value=stage),
            patch("src.pipeline.build_features", return_value=pd.DataFrame(index=idx)),
            patch("src.pipeline.build_outcomes", return_value=pd.DataFrame(index=idx)),
            patch("src.pipeline.run_primary_tests", return_value=empty),
            patch("src.pipeline.run_controlled_tests", return_value=empty),
            patch("src.pipeline.annual_results", return_value=empty),
            patch("src.pipeline.neighborhood_consistency", return_value=empty),
            patch("src.pipeline.regime_results", return_value=empty),
        ):
            result = run(export=False)
        self.assertEqual(len(result["suspension_invalid_events"]), 2)

    def test_invalid_rows_never_enter_mask_and_raw_data_is_unchanged(self):
        parsed = parse_suspension_events(suspension_rows())
        idx = pd.to_datetime(["2026-09-03", "2026-09-04", "2026-09-07", "2026-09-08"])
        frame = pd.DataFrame(10.0, index=idx, columns=["0050", "2330", "1101"])
        frames = {key: frame.copy() for key in ("short_balance", "short_cover", "short_sell", "short_stock_repayment")}
        raw_short = frames["short_sell"].sum(axis=1).copy()
        adjusted, mask = adjust_short_for_suspensions(frames, parsed.valid_events)
        self.assertEqual(mask["0050"].tolist(), [False, True, True, False])
        self.assertFalse(mask[["2330", "1101"]].to_numpy().any())
        pd.testing.assert_series_equal(frames["short_sell"].sum(axis=1), raw_short)
        self.assertTrue(pd.isna(adjusted["short_sell"].loc[idx[1], "0050"]))

    def test_valid_events_still_support_coverage_and_adjusted_change(self):
        parsed = parse_suspension_events(suspension_rows().iloc[[0]].reset_index(drop=True))
        coverage = dataset_coverage({"short_suspension": parsed.valid_events})
        self.assertEqual(coverage.iloc[0]["first_date"], pd.Timestamp("2026-09-04"))
        idx = pd.to_datetime(["2026-09-03", "2026-09-04", "2026-09-07", "2026-09-08"])
        balance = pd.DataFrame(10.0, index=idx, columns=["0050"])
        frames = {key: balance.copy() for key in ("short_balance", "short_cover", "short_sell", "short_stock_repayment")}
        _, mask = adjust_short_for_suspensions(frames, parsed.valid_events)
        change = adjusted_short_change(balance, mask, 1)
        self.assertTrue(change.dropna().eq(0).all())

    def test_universe_counts_etfs_outside_intersection(self):
        diag, _, limitation = build_universe_diagnostics(pd.DataFrame(columns=["2330", "0050"]), pd.DataFrame(columns=["2330"]))
        indexed = diag.set_index("category")
        self.assertEqual(indexed.loc["excluded_etf_like_or_non_common", "count"], 1)
        self.assertEqual(indexed.loc["margin_symbols", "count"], 2)
        self.assertEqual(indexed.loc["market_value_symbols", "count"], 1)
        self.assertEqual(indexed.loc["intersection", "count"], 1)
        self.assertEqual(indexed.loc["primary_common_equity", "count"], 1)
        self.assertTrue(limitation)

    def test_rank_deficient_model_is_explicitly_skipped(self):
        result = controlled_regression(pd.Series([0.0] * 30), pd.Series(range(30)), pd.Series(range(30)))
        self.assertEqual(result["status"], "rank_deficient")
        self.assertTrue(pd.isna(result["signal_p_value"]))
