# Final Config Selection Protocol: Funding-Rate Carry & Daily Trend Following

**Date of Selection:** 2026-10-31 23:45:00 IST  
**Author:** Anshul Parate & AI Quantitative Research Engine  
**Target Asset:** `BTCUSDT`  
**Status:** COMMITTED TO GIT PRIOR TO READING FINAL WINDOW DATA  

---

## 1. Selected Configurations (Chosen Strictly on DEV + VALIDATION)

Based solely on the performance in the DEV (`2019-09` / `2020-01` to `2022-12-31`) and VALIDATION (`2023-01-01` to `2024-12-31`) windows, the following single configurations are selected for evaluation on the sealed FINAL window:

### Strategy A (Funding-Rate Carry): **Config A3**
- **Description:** Long 1.0 BTC Spot + Short 1.0 BTCUSDT Perpetual Futures (delta-neutral).
- **Rule:** Active only when the trailing 30-day mean funding rate $> 0.000\%$ (trailing 90 8h periods mean $F_t > 0$). Otherwise flat.
- **Selection Rationale:** In DEV and VALIDATION, A3 captured virtually identical net return to A1 (+7.90% DEV, +4.77% VAL on 2x capital) with negligible turnover (only 4–8 trades total), while providing a structural off-switch to protect capital during sustained negative funding regimes.
- **Reference Benchmark:** **5.0% Annual Risk-Free Reference Rate** (assumption representing short-term USD cash / Treasury yield / bank rate).
- **Capital Model:** Full collateral model (1.0x spot + 1.0x cash margin = 2.0x capital deployed).

### Strategy B (Daily Trend Following): **Config B1**
- **Description:** Long BTC perpetual futures when daily close $>$ 100-day SMA; Flat when daily close $\le$ 100-day SMA.
- **Rule:** Signals evaluated at closed daily bar $t$ (`close_time`); executed at next day's open $t+1$ (`open_time`). Includes $0.0572\%$ per side transaction costs and daily funding paid/received while in position.
- **Selection Rationale:** Delivered the strongest risk-adjusted performance across both the bear market (DEV: +39.3% CAGR, Sharpe 0.84, cutting max drawdown from −76.7% to −42.4%) and the bull market (VAL: +59.7% CAGR, Sharpe 1.38). Exhibited classic trend-following behavior: 23–31% true win rate with 9x–17x win/loss ratio.
- **Reference Benchmark:** **BTC Buy-and-Hold** over the identical period.

---

## 2. Robustness Check (Stability Only — Not for Model Selection)
To assess parameter sensitivity without p-hacking or re-selection, B1 will also be reported with:
- **B1-SMA80:** 80-day SMA
- **B1-SMA120:** 120-day SMA

*Rule:* These parameter variants are reported strictly for parameter neighborhood stability analysis and cannot replace B1 as the primary candidate.

---

## 3. FINAL Sealed Window & Decision Criteria
- **Window Range:** `2025-01-01 00:00:00 UTC` to `2026-10-31 23:59:59 UTC` (22 calendar months).
- **Evaluation Count:** Strictly evaluated **once**. No post-hoc tuning or modifications permitted.
- **Sample Size Constraint:** If completed trades $< 15$ or the 95% Confidence Interval is very wide, the result must be explicitly classified as **"inconclusive"**.
- **Edge Decision Rule:** Positive evidence requires:
  1. Net return after all realistic costs on FINAL $> 0$.
  2. Monthly block bootstrap 95% CI lower bound $\ge 0$.
  3. For Strategy B: Max drawdown is substantially better than or competitive with buy-and-hold BTC at comparable returns.
