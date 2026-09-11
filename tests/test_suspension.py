import unittest
import pandas as pd
from src.data_loader import parse_suspension_events
from src.features import adjust_short_for_suspensions, adjusted_short_change
from src.diagnostics import dataset_coverage
from src.universe import build_universe_diagnostics
from src.statistics import controlled_regression


class SuspensionTests(unittest.TestCase):
    def events(self):
        return parse_suspension_events(pd.DataFrame({
            "symbol": ["0050", "2330"],
            "停券起日(最後回補日)": ["2026-09-04", "2026-09-04"],
            "停券迄日": ["2026-09-07", None],
            "key_date": ["2026-01-01", "2026-01-01"],
        }))

    def test_range_index_and_duplicate_event_dates(self):
        events = self.events()
        self.assertEqual(events.index.min(), pd.Timestamp("2026-09-04"))
        self.assertEqual(len(events), 2)
        self.assertEqual(events.iloc[0]["symbol"], "0050")
        coverage = dataset_coverage({"short_suspension": events})
        self.assertEqual(coverage.iloc[0]["first_date"], pd.Timestamp("2026-09-04"))

    def test_interval_mask_only_existing_sessions_and_symbol(self):
        idx = pd.to_datetime(["2026-09-03", "2026-09-04", "2026-09-07", "2026-09-08"])
        frame = pd.DataFrame(10., index=idx, columns=["0050", "2330", "1101"])
        frames = {k: frame.copy() for k in ("short_balance", "short_cover", "short_sell", "short_stock_repayment")}
        adjusted, mask = adjust_short_for_suspensions(frames, self.events())
        self.assertEqual(mask["0050"].tolist(), [False, True, True, False])
        self.assertEqual(mask["2330"].tolist(), [False, True, False, False])
        self.assertTrue(adjusted["short_cover"].loc[idx[1], ["0050", "2330"]].isna().all())
        self.assertTrue(frames["short_balance"].eq(10).all().all())
        change = adjusted_short_change(frame, mask, 1)
        self.assertTrue(change.iloc[1:].eq(0).all())

    def test_invalid_interval_stops(self):
        with self.assertRaises(ValueError):
            parse_suspension_events(pd.DataFrame({"symbol": ["0050"], "停券起日(最後回補日)": ["2026-09-07"], "停券迄日": ["2026-09-04"]}))

    def test_invalid_rows_include_raw_values_and_positions(self):
        raw = pd.DataFrame({
            "symbol": ["0050", "2330", "1101"],
            "停券起日(最後回補日)": ["2026-09-07", None, "2026-09-01"],
            "停券迄日": ["2026-09-04", "2026-09-08", None],
            "原因": ["除息", "股東常會", "除息"],
        })
        with self.assertRaises(ValueError) as caught:
            parse_suspension_events(raw)
        error = caught.exception
        self.assertIn("invalid_rows=2", str(error))
        self.assertIn("missing_start=1", str(error))
        self.assertIn("end_before_start=1", str(error))
        self.assertEqual(error.invalid_events["source_row"].tolist(), [0, 1])
        self.assertEqual(error.invalid_events["symbol"].tolist(), ["0050", "2330"])
        self.assertEqual(raw.iloc[0]["停券起日(最後回補日)"], "2026-09-07")

    def test_universe_counts_etfs_outside_intersection(self):
        diag, _, _ = build_universe_diagnostics(pd.DataFrame(columns=["2330", "0050"]), pd.DataFrame(columns=["2330"]))
        self.assertEqual(diag.set_index("category").loc["excluded_etf_like_or_non_common", "count"], 1)

    def test_rank_deficient_model_is_explicitly_skipped(self):
        result = controlled_regression(pd.Series([0.] * 30), pd.Series(range(30)), pd.Series(range(30)))
        self.assertEqual(result["status"], "rank_deficient")
        self.assertTrue(pd.isna(result["signal_p_value"]))
