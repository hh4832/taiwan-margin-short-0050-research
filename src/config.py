from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResearchConfig:
    repository: str = "taiwan-margin-short-0050-research"
    timezone: str = "Asia/Taipei"
    target_symbol: str = "0050"
    k_values: tuple[int, ...] = (1, 3, 5, 10)
    rolling_windows: tuple[int, ...] = (126, 252, 504, 756)
    outcome_horizons: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    pr_edges: tuple[int, ...] = (0, 5, 20, 40, 60, 80, 95, 100)
    reconciliation_tolerance: float = 1e-9
    min_reconciliation_ratio: float = 0.99999
    min_group_n: int = 20
    output_root: Path = Path("outputs")
    primary_universe: str = "historically identifiable Taiwan common equities"

    def serializable(self) -> dict:
        out = asdict(self)
        out["output_root"] = str(self.output_root)
        return out


FINLAB_FIELDS = {
    "open": "price:開盤價",
    "close": "price:收盤價",
    "margin_buy": "margin_transactions:融資買進",
    "margin_sell": "margin_transactions:融資賣出",
    "margin_cash_repayment": "margin_transactions:融資現金償還",
    "margin_prev_balance": "margin_transactions:融資前日餘額",
    "margin_balance": "margin_transactions:融資今日餘額",
    "margin_limit": "margin_transactions:融資限額",
    "margin_utilization": "margin_transactions:融資使用率",
    "short_cover": "margin_transactions:融券買進",
    "short_sell": "margin_transactions:融券賣出",
    "short_stock_repayment": "margin_transactions:融券現券償還",
    "short_prev_balance": "margin_transactions:融券前日餘額",
    "short_balance": "margin_transactions:融券今日餘額",
    "short_limit": "margin_transactions:融券限額",
    "short_utilization": "margin_transactions:融券使用率",
    "offset": "margin_transactions:資券互抵",
    "aggregate_buy": "margin_balance:融資券總買進",
    "aggregate_sell": "margin_balance:融資券總賣出",
    "aggregate_repayment": "margin_balance:現金(券)總償還",
    "aggregate_balance": "margin_balance:融資券總餘額",
    "market_volume": "market_transaction_info:成交股數",
    "market_amount": "market_transaction_info:成交金額",
    "market_count": "market_transaction_info:成交筆數",
    "market_index": "market_transaction_info:收盤指數",
    "market_value": "etl:market_value",
    "short_suspension": "margin_short_sale_suspension",
}
