# Trader-Pro: Quantitative Strategy Research & Audit Pipeline

A quantitative research pipeline built to test whether machine learning and rule-based strategies on Bitcoin (`BTCUSDT`) maintain a measurable edge after realistic exchange fees, taxes, and execution friction.

---

## 1. Executive Summary & Core Results

The initial prototype of this system claimed a **77.7% win rate** on 15-minute Bitcoin scalping. A thorough audit revealed that this metric was an artifact of counting breakeven trades as wins while omitting exchange fees. When evaluated with honest transaction costs, lookahead-free data alignment, and pre-registered hold-out testing:

- **No short-term directional edge was detectable at this sample size after realistic costs.**
- **High win-rate claims were either accounting illusions (breakevens counted as wins) or low-yielding structural carry (~2% on capital).**
- **Trend following reduced peak-to-trough drawdowns during market declines, but sample sizes were small ($N=13$) and the confidence interval remains wide.**

### Summary of Evaluated Strategies

| Strategy / Setup | Timeframe | Evaluation Window | Trade Count | Gross Win Rate / Metric | Net Result After Realistic Costs | 95% Monthly Block Bootstrap CI | Status / Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **15M ML Scalp (Backtest)** | 15M | 2020–2024 | 4,218 | 77.7% gross (62.0% True WR) | **−0.301 R** net / trade | [−0.33, −0.27] R | Negative expectancy after fees. |
| **15M ML Scalp (Live Engine)** | 15M | Oct 2026 | 31 | 80.6% gross (9 TP, 16 BE, 6 SL) | **−₹64.04** net PnL (₹80.09 fees on ₹16.05 gross) | N/A ($N=31$) | Net win rate was 29.0% (9/31). Negative after fees. |
| **Price-Only ML Breakouts** | 1H / 4H | 2024–2026 Hold-out | ~70–140 | 44%–55% True WR | **−0.06 R to +0.05 R** net / trade | Spanned across zero | Inconclusive; no edge detectable beyond noise. |
| **Derivatives Confluence (G2)** | 4H | 2025–2026 Hold-out | 343 | 48.6% True WR | **−0.054 R** net / trade (+0.042 R gross) | [−0.109, −0.006] R | Entire 95% CI is below zero after fees. |
| **Funding-Rate Carry (A3)** | 8H | 2025–2026 Final | 4 turns | 86.9% positive funding periods | **+1.96%** annual return on 2x capital | [+1.4%, +2.5%] | Below 5% risk-free cash reference. Yield compressed over time. |
| **Daily Trend Following (B1)** | 1D | 2025–2026 Final | 13 | 23.1% True WR (4.33 W/L ratio) | **+2.42%** CAGR (vs −10.16% Buy & Hold) | [−23.5%, +36.5%] | Inconclusive ($N=13$). Cut max DD from −53% to −24%. |

---

## 2. Origin: The 77.7% "Win Rate" & What Was Actually Happening

The initial version of the live trading dashboard reported a 77.7% win rate for the 15-minute BTC scalping model. 

### What the Code Was Doing
In the original simulation and reporting code:
```python
# Naive metric in earlier notebook
win_rate = (R >= 0).mean()
```
The strategy employed a breakeven ratchet: once price moved $+0.30\text{R}$ in favor of the trade, the stop loss was moved to the entry price ($R = 0$). The reporting code categorized every breakeven exit as a "win".

### What the Numbers Looked Like After Auditing
When trades were separated by actual exit label and exchange friction was deducted:
1. **True Win Rate ($\text{TP} / (\text{TP} + \text{SL})$):** Only **62.0%** in the 5-year backtest.
2. **Net Win Rate (% of trades where Net PnL $> 0$):** Fell to **29.0%** in live reconciliation.
3. **Reconciled Live Session (31 trades):**
   - Stored Exits: 9 Take Profits, 16 Breakevens, 6 Stop Losses.
   - Gross PnL: **+₹16.05**.
   - Fees & Slippage Paid: **₹80.09**.
   - Net Realized PnL: **−₹64.04**.

The high win rate was an accounting artifact of grouping zero-gain and fee-losing exits into the win column.

---

## 3. Honest Fee & Execution Friction Model

All strategies in this repository deduct exchange fees and slippage on every simulated fill:

$$\text{cost\_R} = \frac{\text{Round-Trip Friction Pct}}{\text{Stop Distance Pct}}$$

- **Exchange Fee (Taker):** 0.040% per side.
- **GST on Trading Fees (India):** 18% ($0.040\% \times 1.18 = 0.0472\%$ per side).
- **Execution Slippage:** 0.010% per side on taker orders.
- **Total Round-Trip Friction:** $2 \times 0.0472\% + 2 \times 0.010\% = \mathbf{0.1144\%}$ of notional.
- **Volatility Filter:** Skip trades where $\text{cost\_R} > 0.15$ (protects capital when stop distance is narrower than $\approx 0.76\%$, where fees consume $>15\%$ of risk).

---

## 4. Leakage & Methodological Issues Found and Fixed

Four distinct methodological flaws were identified and corrected during the audit:

1. **Full-Period Median Cutoff:** Early versions computed prediction threshold quantiles over the entire dataset (including future bars).  
   *Fix:* Converted to an expanding historical window: `expanding(min_periods=100).quantile(0.80)`.
2. **Higher-Timeframe Alignment:** 4H and 1D features were previously aligned using bar open times, allowing data from bars that had not yet closed to leak into intraday signals.  
   *Fix:* Aligned derivative and HTF metrics strictly at completed close times ($T_{\text{pub}} \le \text{bar.close\_time}$). Verified with an automated test suite (`tests/test_derivatives_lookahead.py`) that successfully flagged 100/100 deliberate synthetic leaks.
3. **ATR Formula Lookahead:** True Range calculations originally included the current bar's close.  
   *Fix:* Shifted the True Range input by 1 bar (`shift(1)`).
4. **Overlapping Trades Distortion:** Evaluating independent bars on 4H timeframes (with a 72-hour holding window) caused 44% to 50% of signals to fire while an existing position was already open. These overlapping trades were near-copies of each other, artificially multiplying sample sizes and producing falsely narrow confidence intervals.  
   *Fix:* Built a sequential, one-trade-at-a-time simulator that ignores new entry signals until the current position has formally closed.

---

## 5. Pre-Registration Protocol & Git Commit Hashes

To eliminate p-hacking and selective reporting, every study was committed to git **before** the hold-out data was accessed:

- **Derivatives Feature Confluence Study:**  
  *Git Commit:* `d32217f68b936a8149db18d924ad6c07fcb3c0b5`  
  *File:* [`notebooks/preregistration_derivatives_study.md`](file:///D:/Projects_Main/Trade-Pro/Trader-Pro/notebooks/preregistration_derivatives_study.md)
- **Structural Edge Study (Carry & Trend Following):**  
  *Git Commit:* `287dd23a0a21ef6c5e1f528e88f42ce30c4c1837`  
  *File:* [`notebooks/preregistration_carry_trend_study.md`](file:///D:/Projects_Main/Trade-Pro/Trader-Pro/notebooks/preregistration_carry_trend_study.md)
- **Final Candidate Selection Protocol:**  
  *Git Commit:* `9c882a846fac7b5cc778ff6dcef5c202d708c2c2`  
  *File:* [`notebooks/final_selection_carry_trend.md`](file:///D:/Projects_Main/Trade-Pro/Trader-Pro/notebooks/final_selection_carry_trend.md)

---

## 6. Detailed Findings Across Major Studies

### A. Derivatives Features Confluence Study (4H BTCUSDT)
Tested whether adding funding rates, open interest, and long/short sentiment ratios to price action features could overcome exchange fees on 4H bars.
- **DEV (2020–2023):** In-sample fit showed $+0.1486\text{ R}$, but 5-fold purged walk-forward cross-validation dropped to $+0.0046\text{ R}$ ($95\%\text{ CI: } [−0.044, +0.061]\text{ R}$).
- **Validation (2024):** G2 candidate achieved $+0.0218\text{ R}$ over 290 trades ($95\%\text{ CI: } [−0.057, +0.096]\text{ R}$).
- **Hold-Out (2025–2026):** Over 343 one-at-a-time trades, G2 produced **$-0.0542\text{ R}$ net per trade** (Net PF 0.77, Max DD $-23.4\text{R}$, 95% CI $[−0.109, −0.006]\text{ R}$).
- **Verdict:** No edge detectable after costs. Adding derivatives narrowed the drag relative to price-only ($-0.054\text{R}$ vs $-0.113\text{R}$), but the net expectancy remains negative.

### B. Delta-Neutral Funding Carry (Strategy A3)
Tested a cash-and-carry position (Long 1.0 Spot BTC + Short 1.0 BTC Perpetual Futures) with a 30-day trailing mean funding filter ($>0$).
- **Consistency:** Funding payments were positive in **85% to 91%** of 8-hour periods across all windows.
- **Yield Path on Total Capital (2x Collateral Model):**
  - `2019-09` to `2022-12` (DEV): **~8.15%** annualized net return ($95\%\text{ CI: } [+5.2\%, +11.5\%]$).
  - `2023-01` to `2024-12` (VALIDATION): **~4.77%** annualized net return ($95\%\text{ CI: } [+3.4\%, +6.4\%]$).
  - `2025-01` to `2026-10` (FINAL): **~1.96%** annualized net return ($95\%\text{ CI: } [+1.4\%, +2.5\%]$).
- **Yield Compression:** In H1 2026, annualized net yield on total capital fell to under **1%** ($+0.42\%$ in Q1 2026, $+0.35\%$ in Q2 2026) before rebounding to $+3.25\%$ in Q3 2026.
- **Verdict:** While consistently positive, a net return of ~2% fails to beat a 5% risk-free cash reference rate while carrying exchange counterparty, liquidation, and tax risks.

### C. Daily Trend Following (Strategy B1: 100-Day SMA)
Tested a long/flat trend rule on closed daily bars, executed at the next day's open, accounting for taker fees and cumulative perpetual funding payments.
- **DEV (2020–2022, Bear/Cycle):** CAGR **+39.3%** vs $+32.0\%$ Buy & Hold; Max DD **−42.4%** vs $−76.7\%$ Buy & Hold.
- **VALIDATION (2023–2024, Bull Run):** CAGR **+59.7%** vs **+137.5%** Buy & Hold; Max DD **−38.0%** vs $−26.3\%$ Buy & Hold. (Lagged during the strong bull market).
- **FINAL (2025–2026, Choppy/Decline):**
  - **CAGR:** **+2.42%** vs **−10.16%** for Buy & Hold BTC.
  - **Max Drawdown:** **−24.43%** vs **−52.98%** for Buy & Hold BTC.
  - **Trade Sample Size:** **13 completed trades** over 22 months (True WR 23.1%, Avg Win/Loss 4.33, Profit Factor 1.30).
  - **Bootstrap CI:** 95% Monthly Block CI is **[−23.5%, +36.5%]**.
  - **Concentration:** Most of the net gain came from two trades (+15.5% and +10.8%), while 10 of the 13 trades were small losses.
- **Parameter Neighborhood Stability:** 80-day SMA produced $+5.22\%$ CAGR (Max DD $-28.9\%$, 11 trades); 120-day SMA produced $-0.34\%$ CAGR (Max DD $-28.5\%$, 12 trades).
- **Verdict:** Statistically inconclusive due to small sample size ($N=13$). The rule cut drawdowns in half relative to buy-and-hold during down markets, but returns are concentrated in a few trades and lag in strong bull markets.

---

## 7. Limitations & Disclosures

1. **Asset Scope:** All tests were conducted strictly on `BTCUSDT`.
2. **Sample Size Constraints:** Daily trend following generated only 13 completed trades in the 22-month final window. At this sample size, confidence intervals are wide ($60\%$ spread), and results cannot be distinguished from sample variance.
3. **No Real-Money Performance:** All results reflect backtested simulations and paper execution.
4. **Paper Trading Status:** The paper executor for setup 1H-A is deployed and verified to float64 machine epsilon ($10^{-16}$) parity against Python LightGBM, running via hourly cron. It currently logs **0 live trades**.
5. **Taxation & Legal Realities (India):** Under Section 115BBH, Virtual Digital Assets (VDAs) face a 30% flat tax on gross gains with no deduction for losses, along with a 1% TDS on every transfer. High-turnover intraday strategies and low-margin carry trades are economically unviable under this tax structure.

---

## 8. How to Reproduce

### Prerequisites
- Python 3.11+
- Virtual environment with dependencies: `pandas`, `numpy`, `lightgbm`, `joblib`, `pyarrow`, `scikit-learn`

### Running the Studies
```bash
# 1. Verify Lookahead Absence & Data Pipeline
python tests/test_derivatives_lookahead.py

# 2. Run Python vs TypeScript Parity Test
python scripts/test_parity.py

# 3. Re-run Derivatives Confluence Study (Pre-Checks & Holdout)
python scripts/run_study_v2.py

# 4. Re-run Structural Study (Carry & Trend on DEV/VAL)
python scripts/run_structural_study.py

# 5. Re-run Final Evaluation on Sealed Window
python scripts/evaluate_final.py
```

---

## 9. What I Learned

1. **A high win rate without fee modeling is meaningless.** What looked like a 77.7% winning algorithm was losing money on live fills because breakeven exits were being counted as wins while transaction fees steadily depleted equity.
2. **Execution friction dominates low-timeframe trading.** When target stop distances are under 1%, taker fees, exchange GST, and slippage account for 15% to 30% of total risk. A strategy needs an extraordinarily strong raw predictive signal just to break even.
3. **Pre-registration is essential.** It is easy to find combinations of moving average lengths or ML hyperparameters that look profitable in hindsight. Locking configurations into git commits before reading test data is the only way to avoid tuning on noise.
4. **Overlapping trades distort confidence intervals.** Allowing an algorithm to enter multiple positions concurrently during multi-day holding periods generates clustered, correlated trades that make confidence intervals appear artificially tight.
5. **Negative results are valuable.** Discovering that an edge does not exist through systematic testing is far better than discovering it with real capital in a live account.
