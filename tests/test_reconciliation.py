import pandas as pd
import unittest
from src.diagnostics import run_reconciliation


def frames():
    idx = pd.date_range("2024-01-01", periods=2)
    f = lambda x: pd.DataFrame({"2330": x}, index=idx)
    return {"margin_prev_balance": f([10, 12]), "margin_balance": f([12, 9]), "margin_buy": f([4, 1]), "margin_sell": f([1, 3]), "margin_cash_repayment": f([1, 1]), "short_prev_balance": f([5, 7]), "short_balance": f([7, 4]), "short_sell": f([4, 1]), "short_cover": f([1, 3]), "short_stock_repayment": f([1, 1])}


class ReconciliationTests(unittest.TestCase):
    def test_accounting_identities_pass(self):
        result = run_reconciliation(frames(), 0, 1)
        self.assertTrue((result.exact_match_ratio == 1).all())

    def test_bad_identity_stops_stage_zero(self):
        d = frames(); d["margin_balance"].iloc[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "Stage 0"):
            run_reconciliation(d, 0, 1)
