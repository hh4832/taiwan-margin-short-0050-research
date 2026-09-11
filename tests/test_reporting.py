import unittest
import warnings

import pandas as pd

from src.reporting import assess_signals, get_latest_tradable_signal_date, signal_summary, thermometer_table


def result_rows():
    common = {
        "family": "short_cover", "k": 1, "rolling_window": 126,
        "pr_group": "PR95-100", "outcome_horizon": "O1_C1", "N": 50,
        "mean_return": .01, "effect_vs_unconditional": .005, "win_rate": .6,
        "evidence_level": "Level A",
    }
    return pd.DataFrame([
        common | {"predictor": "short_cover_raw__k1__raw__w126_percentile"},
        common | {"predictor": "short_cover_adjusted__k1__raw__w126_percentile"},
    ])


class ReportingTests(unittest.TestCase):
    def test_future_auxiliary_date_cannot_advance_signal_date(self):
        close = pd.Series([100.0, 101.0], index=pd.to_datetime(["2026-09-10", "2026-09-11"]))
        suspension_latest = pd.Timestamp("2026-10-30")
        signal_date = get_latest_tradable_signal_date(close)
        table = thermometer_table(self._assessed(), signal_date)
        self.assertEqual(signal_date, pd.Timestamp("2026-09-11"))
        self.assertTrue(table["signal_date"].eq(pd.Timestamp("2026-09-11")).all())
        self.assertLess(table["signal_date"].max(), suspension_latest)

    def _assessed(self):
        annual = pd.DataFrame([{
            "predictor": row.predictor, "family": row.family, "k": row.k,
            "rolling_window": row.rolling_window, "pr_group": row.pr_group,
            "outcome_horizon": row.outcome_horizon, "years_with_samples": 4,
            "positive_year_ratio": .75, "largest_year_sample_share": .3,
            "few_year_concentration_flag": False,
        } for row in result_rows().itertuples()])
        neighborhood = pd.DataFrame([{
            "family": "short_cover", "pr_group": "PR95-100", "consistency_type": "fixed_horizon",
            "fixed_parameter": "O1_C1", "consistency_score": .8,
        }])
        return assess_signals(result_rows(), annual, neighborhood, pd.DataFrame(), True)

    def test_adjusted_suspension_signal_is_retrospective_and_not_live(self):
        assessed = self._assessed().set_index("predictor")
        adjusted = assessed.loc["short_cover_adjusted__k1__raw__w126_percentile"]
        self.assertTrue(adjusted["retrospective_sensitivity"])
        self.assertFalse(adjusted["publication_time_verified"])
        self.assertTrue(adjusted["SUSPENSION_COVERAGE_LIMITATION"])
        self.assertFalse(adjusted["live_eligibility"])
        self.assertEqual(adjusted["research_decision"], "修改後再測")

    def test_level_a_does_not_automatically_mean_keep(self):
        results = result_rows().iloc[[0]]
        assessed = assess_signals(results)
        self.assertEqual(assessed.iloc[0]["robustness_status"], "insufficient")
        self.assertEqual(assessed.iloc[0]["research_decision"], "修改後再測")
        text = signal_summary(assessed)
        self.assertIn("robustness=insufficient", text)

    def test_nullable_concentration_flag_has_explicit_boolean_handling(self):
        results = result_rows().iloc[[0]]
        row = results.iloc[0]
        annual = pd.DataFrame([{
            "predictor": row.predictor, "family": row.family, "k": row.k,
            "rolling_window": row.rolling_window, "pr_group": row.pr_group,
            "outcome_horizon": row.outcome_horizon, "years_with_samples": 4,
            "positive_year_ratio": .75, "largest_year_sample_share": .3,
            "few_year_concentration_flag": None,
        }])
        neighborhood = pd.DataFrame([{
            "family": row.family, "pr_group": row.pr_group,
            "consistency_type": "fixed_horizon", "fixed_parameter": row.outcome_horizon,
            "consistency_score": .8,
        }])
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            assessed = assess_signals(results, annual, neighborhood)
        self.assertEqual(str(annual["few_year_concentration_flag"].astype("boolean").dtype), "boolean")
        self.assertEqual(assessed.iloc[0]["robustness_status"], "mixed")
        self.assertEqual(assessed.iloc[0]["research_decision"], "修改後再測")


if __name__ == "__main__":
    unittest.main()
