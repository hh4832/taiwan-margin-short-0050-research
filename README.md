# Taiwan Margin & Short → 0050 Research

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hh4832/taiwan-margin-short-0050-research/blob/main/notebooks/margin_short_0050_research_colab.ipynb)

研究臺股整體市場融資融券行為，是否能解釋或預測 0050 下一交易日與未來 1/2/3/5/10/20 個交易日報酬。這是 signal predictive study，不是完整投資組合回測；O1 是理論進場 benchmark，實務仍有開盤滑價與成交偏離。

## 假設與市場機制

- 融資 flow/level 可能反映散戶槓桿追價、風險偏好或被迫去槓桿。
- 融券 flow/level 可能反映方向性看空、避險、套利或制度性回補。
- 這些訊號也可能只是近期價格的結果，因此 secondary model 必須控制 0050 prior 1/3/5/10D return。

假設、訊號定義、統計結果、結果解釋與尚未證實推測必須分開呈現。不得以最佳單點直接宣稱策略有效。

## Exact FinLab datasets

程式只呼叫下列名稱：

- `price:開盤價`、`price:收盤價`（只取 `0050`）
- `margin_transactions:融資買進/融資賣出/融資現金償還/融資前日餘額/融資今日餘額/融資限額/融資使用率`
- `margin_transactions:融券買進/融券賣出/融券現券償還/融券前日餘額/融券今日餘額/融券限額/融券使用率/資券互抵`
- `margin_balance:融資券總買進/融資券總賣出/現金(券)總償還/融資券總餘額`
- `market_transaction_info:成交股數/成交金額/成交筆數/收盤指數`
- `etl:market_value`
- `margin_short_sale_suspension`

每次執行會輸出各 dataset 實際 coverage，不把研究硬切成同一起日。

## Signal timing 與 outcomes

融資券資料在 d0 收盤後才完整可知，故 d0 對應下一個有 0050 正常 OHLC 的交易日 O1，且不 forward-fill 假日。

`return_h = C_h / O1 - 1`，h = 1/2/3/5/10/20。`C0 → O1` 只作 descriptive/non-tradable 統計。

## Predictor catalog

- Gross flow：融資買進、賣出、現金償還；融券賣出（新增空單）、買進（回補）、現券償還。
- k = 1/3/5/10 個交易日，raw 使用 sum。
- Amount ratio（融資買進/賣出 primary normalized）：先加總分子與市場成交金額分母，再相除。
- Volume ratio：融資券張數先乘 1000 換成股；同樣使用 ratio of sums。它是 robustness，不能與 amount ratio 重複計為獨立證據。
- Position change：balance 相對 k 日前的百分比變化；absolute change 留於 diagnostics。
- Level：融資/融券部位市值 ÷ market cap、融資信用金額 ÷ market cap、`approx_margin_maintenance_ratio`、市值化券資比。

`approx_margin_maintenance_ratio` 是市場 aggregate approximation，不是券商整戶擔保維持率。

## Rolling percentile 與固定 PR groups

只使用當時及之前的 126/252/504/756 個**有效交易觀測**；缺值日保持缺值，不 forward-fill，也不降低完整窗口要求。固定 descriptive bins 採左閉右開：`[0,5)`、`[5,20)`、`[20,40)`、`[40,60)`、`[60,80)`、`[80,95)`、`[95,100]`。主要 inferential comparison 仍獨立定義為 PR≤5 或 PR≥95 vs non-group，不可事後新增門檻追結果；七個 descriptive bins 不進 family/global FDR。

## Stage 0 diagnostics

Run All 在統計前檢查 dataset coverage、融資/融券 accounting identity、individual aggregate vs market aggregate、security universe 與停券影響。若 reconciliation 不接近完全一致即停止。

普通股以保守 ticker 規則建立初步 primary universe，但這不是 point-in-time security master，因此固定輸出 `UNIVERSE_LIMITATION=True`，並要求保留 all-security sensitivity；不得用今日名單回頭過濾歷史。

## 停券處理

融券同時保留 raw 與 adjusted signals。停券表按事件讀取，以 symbol、停券起日(最後回補日)、停券迄日對應個股；起訖日均包含，僅遮罩既有交易日。缺少迄日只遮罩起日，不推測區間。缺少起日、迄日早於起日或無法解析的事件列會完整保留在 quarantine，不加入 adjusted mask，也不阻斷 raw analysis。餘額變化以前後同一組股票計算，排除整個比較窗口內受停券影響者，避免遮罩進出造成假變化。所有 adjusted predictor 固定標示 `retrospective_sensitivity=True`、`publication_time_verified=False`、`is_live_eligible=False`；quarantine 非空時另標示 `SUSPENSION_COVERAGE_LIMITATION=True`。

這是事後敏感度分析：key_date 未證實為歷史公告日，不能據此宣稱停券資料在當時已可取得；起日前提前回補與歷史事件完整性仍有限制。dataset_coverage 的停券起訖指有效事件起日範圍，不是下載或公告時間。輸出 `suspension_events.csv`、`suspension_invalid_events.csv`、`suspension_validation_summary.csv` 與 `suspension_diagnostics.csv`，分別保存有效事件、原始 quarantine、Stage 0 coverage 限制與每日遮罩數。制度性停券與強制回補不得解讀為主動市場訊號。

報酬計算使用 fill_method=None，缺值不自動補價；缺少 predictor 的日期不進對照組或控制迴歸。controlled_results.csv 的 status 區分 estimated、insufficient_sample、rank_deficient，跳過無法唯一估計的模型。以上修正會改變歷史統計及 FDR，舊結果需重跑；本研究尚未完成完整實證驗證。

## Statistics、FDR 與 robustness

每個 predictor × k × rolling window × PR group × outcome 輸出 N、平均/中位數、勝率、標準差、Q25/Q75、vs zero、vs unconditional、raw p、family/global BH-FDR q。

- Level A：global FDR < .05
- Level B：family FDR < .05，但 global 未通過
- Level C：raw p < .05，但 family/global 未通過
- No Evidence：raw p ≥ .05

候選訊號另外依 fixed horizon、fixed k、fixed rolling window 檢查方向一致性；consistency 只作 robustness，不能取代 FDR。年度驗證完整納入所有 Level A/B，不依 raw p-value 任意截取前 30 名。short-cover raw/normalized/adjusted variants 另行標記方向衝突；`approx_margin_maintenance` 以七 bins 描述性分類單調、U 型或 mixed，不額外加入新模型。robust 預設要求至少 3 個有樣本年度、年度與 neighborhood 方向一致率均至少 60%、單一年樣本占比不超過 50%，門檻明列於 `ResearchConfig` 與 `run_info.txt`。最終分開輸出 statistical evidence、robustness status、live eligibility 與「保留／修改後再測／淘汰／無法判定」。

## Colab Run All

Colab 的 Python runtime 可能由平台升級；notebook 接受 Python 3.11 以上並在 `run_info.txt` 記錄實際版本，不再要求版本必須剛好等於 3.11。

1. 在 Colab Secrets 新增 `GITHUB_TOKEN`（private repo 讀取權限），以及需要時的 FinLab credential；不要寫進 notebook。
2. 點上方 badge，依序 Run All。
3. Notebook 安裝依賴、clone 或更新 repo、執行 FinLab login、呼叫 `src.pipeline`、顯示 diagnostics/results。
4. Google Drive 掛載後，成功 output 會複製到 `MyDrive/Quant_Research/taiwan-margin-short-0050-research/<timestamp_commit>/`；不覆蓋舊資料。

資料修正後可先執行 `from src.pipeline import run_stage_zero; stage0 = run_stage_zero()`，只檢查 coverage、reconciliation、universe 與 suspension quarantine；通過後才執行完整 `run()`。

## Outputs 與追溯性

每次建立 `outputs/YYYYMMDD_HHMMSS_<git_commit>/`，包含 `run_info.txt`、coverage/reconciliation/universe/feature catalog、level coverage、primary/FDR diagnostics、controlled low/high tails、七個 PR bins、annual summary、neighborhood、variant conflict、shape、universe sensitivity、`signal_summary.md`、`thermometer_signals.csv`、bias checklist 與可選 parquet。thermometer 的 `signal_date` 只取最新有效 0050 收盤日，不受未來 auxiliary/suspension event 日期影響。大型 outputs 不進 Git。`run_info.txt` 記錄 commit、branch、時間、Python/FinLab 版本、各資料起訖與研究設定。

`thermometer_signals.csv` 保留未來接入 `taiwan-market-thermometer` 的 schema；本 repo 不修改該專案。

## Bias checklist

每次正式分析必須逐項覆核 look-ahead、survivorship、data snooping、selection bias、成本、滑價、流動性、樣本數、少數年份/股票集中與 suspension distortion。

## 反對者觀點

1. 融資增加可能只是上漲後追價。
2. 融券可能是 hedge，而非 outright bearish view。
3. 強制停券/回補可能製造假訊號。
4. 全市場訊號可能由中小型股主導，與 0050 不一致。
5. 不同年代制度與市場結構可能改變。
6. 大量檢定會產生漂亮但不可重現的單點。
7. 市值 normalization 仍受價格變化機械性影響。

## Local validation

```bash
python -m unittest discover -s tests -v
```

沒有實際 FinLab credential/data 的測試只驗證 accounting、日期對齊、公式與防止 future leakage；不代表已完成實證研究。
