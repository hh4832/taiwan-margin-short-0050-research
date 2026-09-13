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

報酬計算使用 fill_method=None，缺值不自動補價；缺少 predictor 的日期不進對照組或控制迴歸。Controlled regression 保留 HC3，但只在總樣本、tail/control 組別、signal/prior variation、design rank/condition number 與 hat leverage 均通過後計算；兩組最低樣本沿用 `min_group_n=20`，condition number ≥ `1e12` 視為數值 rank-deficient，`max_leverage >= 1 - 1e-10` 時不進 HC3。這些數值門檻明列於 `ResearchConfig` 與 `run_info.txt`。status 區分 estimated、insufficient_sample、insufficient_group_sample、no_signal_variation、no_prior_variation、rank_deficient、high_leverage_unstable、hc3_nonfinite；只有 estimated 且 p-value finite 才可解讀，其餘均為「無法判定」。Primary/FDR 不受 controlled diagnostics 反向修改。

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

每次建立 `outputs/YYYYMMDD_HHMMSS_<git_commit>/`，包含 `run_info.txt`、coverage/reconciliation/universe/feature catalog、level coverage、primary/FDR diagnostics、controlled low/high tails、`controlled_regression_diagnostics.csv`、七個 PR bins、annual summary、neighborhood、variant conflict、shape、universe sensitivity、`signal_summary.md`、`thermometer_signals.csv`、bias checklist 與可選 parquet。controlled diagnostics 同時列出整體與 family × PR tail × prior k × outcome 的 status 分布。thermometer 的 `signal_date` 只取最新有效 0050 收盤日，不受未來 auxiliary/suspension event 日期影響。大型 outputs 不進 Git。`run_info.txt` 記錄 commit、branch、時間、Python/FinLab 版本、各資料起訖與研究設定。

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

## Margin Turnover incremental branch

`research/margin-turnover` 將 adjusted baseline `88d9f267ddfa4b5e80cadcb1709d34ab0d9ab2ce`
研究視為 frozen baseline，只新增一個 economic variable：`margin_buy + margin_sell`。
獨立入口是 `src.margin_turnover.run_margin_turnover_study()`；它不呼叫
`src.pipeline.run()`，也不重算原本 4,884 個 tests、short/level/suspension、完整年度、
neighborhood、universe sensitivity 或 thermometer export。

三個 variant 使用同一組 k=1/3/5/10、rolling window=126/252/504/756 與固定 PR bins：

- `amount_ratio`：融資買進金額＋融資賣出金額 ÷ TAIEX＋OTC 成交金額，是 primary normalized version。
- `volume_ratio`：primary-universe 融資買進／賣出張數乘 1000 ÷ TAIEX＋OTC 成交股數，是 robustness。
- `raw_lots`：primary-universe 融資買進＋賣出張數，是 diagnostics/robustness。

k-day ratios 都是 ratio of rolling sums，缺值不 forward-fill，percentile 沿用既有
`trailing_percentile()`。Turnover inferential universe 最大 576 cells，BH-FDR scope 固定為
`margin_turnover_incremental_study`。因此本 branch 的 Level A 只代表 turnover incremental
experiment global FDR，不是 baseline 4,884 tests 與 turnover 合併後的 global FDR。

Absorption test 只讀 frozen `fdr_results.csv` 中 retained margin_buy/margin_sell 訊號。
Primary control 是相同 k/window 的 continuous `amount_ratio percentile / 100`，volume ratio
只作 robustness。HC3 前沿用總樣本、tail/control 樣本、variation、rank、condition number、
leverage 與 finite covariance safeguards。`attenuation_ratio = 1 - |beta_after|/|beta_before|`
是 primary continuous quantity；beta before 接近 0 時保持 undefined。分類僅作 descriptive：

- `survives_turnover_control`：方向相同、至少保留 70% beta，且 after p<0.05。
- `partially_absorbed`：方向相同且 attenuation 介於 30%–70%。
- `largely_absorbed`：attenuation >70% 或 after 明顯失去顯著性。
- `sign_reversal`、`unstable_collinearity`、`insufficient_sample`：分別標記反轉、數值不穩及樣本不足。

Colab 使用 `notebooks/margin_turnover_incremental_colab.ipynb`，必須明確設定
`BASELINE_RUN_DIR`。該資料夾必須包含 baseline 的 run info、FDR、controlled、annual 與
neighborhood outputs，且 `run_info.txt` 的完整 git commit 必須等於指定 baseline，否則停止。
增量結果輸出到 `outputs_turnover/<timestamp>_<commit>_margin_turnover_incremental/`，不覆寫 baseline。

## Margin Composition incremental branch

`research/margin-composition` 以 turnover commit
`e613b5e7176c998bd1e8abe191f0e7a4caf2abd3` 為 adjusted frozen base，只新增
`buy_share = BuyAmount_k / (BuyAmount_k + SellAmount_k)`。分子與分母先各自加總 k 日，
非正分母維持缺值，不補值或推測。`imbalance = 2 * buy_share - 1` 只作數值、排序、
percentile 與 PR bin 等價性驗證，不形成第二個 FDR family。

推論範圍固定為 4 個 k、4 個 rolling windows、2 個 tails 與 6 個 outcomes，共 192
個 cells；scope 是 `margin_composition_incremental_study`。NaN p-value 明確標為
`Not Testable`。固定低 turnover 診斷只檢查 amount-ratio k10/W504/PR0-5，並把
composition 分為 PR0-20、PR20-80、PR80-100 三組。Model B→C absorption 使用相同
complete-case rows；rank、condition number、leverage 或 HC3 失敗一律標為
`unstable_collinearity`。

Colab 使用 `notebooks/margin_composition_incremental_colab.ipynb`，必須指定 frozen
`BASELINE_RUN_DIR` 與 `TURNOVER_RUN_DIR`。入口
`src.margin_composition.run_margin_composition_study()` 不呼叫完整 pipeline 或既有兩項研究，
結果只寫入 timestamped `outputs_composition/`。

## Adjusted 0050 outcome prices

0050 forward outcomes use `etl:adj_open` and `etl:adj_close`. This removes the
June 2025 1:4 split discontinuity while leaving the O1→Ch formula, horizons,
predictors, thresholds, and FDR universe unchanged. Each full run writes
`outcome_price_diagnostics.csv`, uses an `_adjusted_price` output suffix, and
records the price sources and corporate-action correction in `run_info.txt`.

## Margin Buy / Sell × prior-return regime incremental study

`research/margin-regime-interaction` 以 adjusted-price research chain 的 baseline
`88d9f267ddfa4b5e80cadcb1709d34ab0d9ab2ce`、turnover
`e613b5e7176c998bd1e8abe191f0e7a4caf2abd3`、composition
`cf16e4aa7056c4c2c54e1d6248312bee6acac9da` 為 frozen dependencies。獨立入口
`src.margin_regime_interaction.run_margin_regime_interaction_study()` 不呼叫或覆寫前三項研究。

本研究只新增 `prior_5d_return = adjusted_close[t] / adjusted_close[t-5] - 1`：大於零為
Up，否則為 Down；缺值不分類。Primary signals 僅使用 baseline 已定義的 Margin Buy 與
Margin Sell `amount_ratio` high tail（PR95-100），不新增 normalization、threshold 或 regime
horizon。每個 outcome 先輸出 A/B/C/D 四個 state，再以 HC3 估計
`FutureReturn ~ Signal + DownRegime + Signal×DownRegime`。`beta_up` 是 Signal coefficient，
`beta_down` 是 Signal 與 interaction coefficients 的和。

新的 BH-FDR universe 只包含 binary primary model 的 interaction p-values；不與 baseline
4,884 tests、turnover 或 composition 合併。Turnover-controlled model加入相同 k/window 的
continuous amount-ratio turnover percentile。Continuous prior-return interaction 另存為
secondary robustness，不進 primary FDR。只有 Level A/B interaction candidates 進年度
robustness；連續 signal 日按交易列相鄰且 regime 相同合併為 event cluster。

Colab 使用 `notebooks/margin_regime_interaction_incremental_colab.ipynb`，必須明確指定三個
frozen run directories。結果寫入 timestamped `outputs_regime_interaction/`，且
`run_info_regime_interaction.txt` 記錄三個 dependency commits、current commit 與
`etl:adj_open` / `etl:adj_close`。Signal occurrence 差異不能單獨證明預測效果差異；
Margin Sell 也不得未經 occurrence 與 interaction 證據直接解讀成去槓桿。

正式 Colab run 固定讀取下列 adjusted-price artifacts，不使用 glob 或「最新資料夾」fallback：

- baseline：`20260912_220859_88d9f26_adjusted_price`
- turnover：`margin_turnover/20260913_075344_e613b5e_margin_turnover_adjusted`
- composition：`margin_composition/20260913_080358_cf16e4a_margin_composition_adjusted`

`validate_regime_input_runs()` 會在 FinLab 資料載入前驗證三層必要檔案、repository、完整
commit dependency chain、`etl:adj_open`、`etl:adj_close` 與
`outcome_price_adjusted=True`；任何不符均停止。Regime 與 signal index 會明確 reindex 並輸出
`regime_index_alignment_diagnostics.csv`。只有 prior-5D 前五個 warm-up 缺值可被記錄後排除；
其餘 active signal 無 regime 時停止。若 interaction 沒有 Level A/B，年度 confirmatory
輸出仍保留正式空 schema，不以 Level C 取代。

## Margin Buy + Sell + prior-return joint incremental study

`research/margin-buy-sell-prior-return-joint` 以 baseline `88d9f267...`、turnover
`e613b5e7...`、composition `cf16e4aa...` 與 regime interaction `a5ca1e98...` 為四層 frozen
adjusted-price dependencies。獨立入口
`src.margin_buy_sell_prior_return_joint.run_margin_buy_sell_prior_return_joint_study()` 只估計
相同 k、相同 rolling window 的 Margin Buy/Sell amount-ratio PR95-100 joint model：

`FutureReturn ~ BuyHigh + SellHigh + prior_5d_return + BuyHigh×prior + SellHigh×prior`

不加入 Buy×Sell 或三階 interaction。`prior_5d_return` 維持 adjusted close[t] / adjusted
close[t-5] - 1。Interaction（Buy×Prior、Sell×Prior）與 conditional main effects（Buy、Sell）
使用兩個互不混合的 BH-FDR universe。2×2 joint states 只作描述，並預先要求每個 state 至少
20 筆才能估計正式 joint interaction；condition number ≥1e12 或任一 VIF ≥10 的模型標為
`unstable_collinearity`，不得硬解讀。

Nested Model A/B/C 使用相同 complete cases 比較被另一 flow 吸收的幅度。Turnover
amount-ratio 只在第二階段 robustness 加入，不擴大 primary FDR。只有 interaction Level A/B
候選進年度 confirmatory analysis，不以 Level C 代替；連續 signal trading rows 另作 cluster
diagnostics。Colab runner 是
`notebooks/margin_buy_sell_prior_return_joint_colab.ipynb`，固定讀取 prompt 指定的四個 Drive
run directories，不用 glob 或最新資料夾 fallback。輸出寫入 timestamped
`outputs_joint_flow_prior_return/`，不覆寫既有研究或舊 run。
