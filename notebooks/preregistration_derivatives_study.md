# Pre-Registration: BTCUSDT Derivatives Features Study (1H & 4H)

**Date of Pre-Registration:** 2026-10-31  
**Author:** Anshul Parate & AI Quantitative Research Engine  
**Target Asset:** `BTCUSDT` (USDT-M Perpetual Futures)  
**Status:** PRE-REGISTERED (Frozen prior to training or evaluating feature groups)  

---

## 1. Research Hypothesis & Objective
To determine whether adding derivative market microstructure features (funding rates, open interest, top-trader long/short ratios, global account long/short ratios, and taker volume ratios) to price action provides a statistically significant, fee-clearing edge in predicting Bitcoin returns on 1H and 4H horizons.

---

## 2. Feature Groups & Pre-Registered Configurations
We pre-register exactly **8 configurations** (4 feature groups $\times$ 2 timeframes), which is strictly $\le 10$ max configs:

1. **G0 (Price-Only Baseline):** 53 price action and volatility features (RSI, ATR, Donchian, MACD, EMAs, return momentum, Parkinson volatility).
2. **G1 (G0 + Funding Rate):** G0 features plus:
   - Latest completed funding rate ($F_t$)
   - Cumulative 3-period funding rate ($F_{3d}$)
   - Funding rate deviation from 30-day moving average
   - Funding rate percentile rank (expanding window)
3. **G2 (G0 + Open Interest & Sentiment Ratios):** G0 features plus:
   - 1H / 4H Open Interest percentage change ($\Delta \text{OI}_{1h}$, $\Delta \text{OI}_{24h}$)
   - Open interest to volume ratio ($\text{OI} / \text{Vol}$)
   - Global long/short account ratio ($LS_{\text{global}}$)
   - Top-trader long/short account ratio ($LS_{\text{top}}$)
   - Taker buy/sell volume ratio ($V_{\text{taker\_ratio}}$)
4. **G3 (G0 + Full Derivatives Confluence):** G0 + Funding Rates + Open Interest + Sentiment & Taker Ratios (all features combined).

- **Timeframes Evaluated:** 1H and 4H.
- **Total Configurations Tried:** **8 total** (4 feature groups $\times$ 2 timeframes). No other variants will be evaluated.

---

## 3. Data Splits & Leakage Prevention Protocol
To guarantee complete independence and prevent p-hacking:

1. **DEV Period:** Inception (`2020-09-01`) to `2023-12-31 23:59:59` UTC.
   - Used exclusively for model training and walk-forward cross-validation.
2. **VALIDATION Period:** `2024-01-01` to `2024-12-31 23:59:59` UTC (Calendar year 2024).
   - Used exclusively to select the best feature group (by Avg Net R).
3. **HOLD-OUT Period:** `2025-01-01` to `2026-08-31 23:59:59` UTC (`today - 60 days`).
   - *Disclosure:* Data from 2025+ has been touched by earlier studies; therefore it is treated as a semi-independent validation window.
4. **SEALED FINAL WINDOW:** `2026-09-01` to `2026-10-31` (Newest 60 days).
   - **Strictly sealed and untouched.** Neither features, models, nor validation code may read this window until explicit human approval is granted after reviewing B4/B5.

---

## 4. Point-in-Time Causality & Timestamp Alignment
- A derivative feature stamped at publish time $T_{\text{pub}}$ is usable **only on bars opening at or after $T_{\text{pub}}$** (i.e. $T_{\text{pub}} \le \text{bar.close\_time}$).
- Daily metrics are aggregated strictly to closed 5m bins.
- No forward-filling across data gaps $> 24\text{ hours}$.
- Zero lookahead bias verified by automated tests on 1,000 random bars.

---

## 5. Trade Labeling & Honest Fee Model
- **Label:** Realized Net R per trade after fees and slippage ($R_{\text{net}}$), NOT gross win/loss or breakeven-padded wins.
- **Execution Cost:**
  - Shark Exchange Taker Fee: 0.040% per side
  - GST: 18% on trading fees ($0.040\% \times 1.18 = 0.0472\%$ per side)
  - Slippage: 0.010% per side ($0.020\%$ round-trip)
  - **Total Round-Trip Friction:** **0.1144% of notional**.
- **Exit Logic:** 1:1 TP/SL, Breakeven ratchet triggered at $+0.30\text{R}$ locking exact round-trip friction.
- **Filter:** Skip trade if round-trip fee $\text{cost\_R} > 0.15$.
- **Validation:** Purged walk-forward cross-validation with expanding prediction thresholds.

---

## 6. Pre-Registered Decision Rule
Positive evidence of a genuine tradable edge is declared **ONLY IF ALL** of the following four conditions are met:
1. Taker Avg Net R on both the HOLD-OUT and the SEALED FINAL WINDOW are strictly $> 0$.
2. The pooled monthly block bootstrap 95% Confidence Interval lower bound on (HOLD-OUT + SEALED) is $\ge 0$.
3. The selected model's Avg Net R strictly exceeds the G0 (price-only) baseline on the same hold-out windows.
4. At least 2 of 3 calendar years show positive Avg Net R.

- **Uncertainty & Sample Size Constraint:**
  - Trade counts must be reported in every table.
  - If the 95% Confidence Interval width $> 0.20\text{ R}$, the result must be explicitly stated as **"inconclusive"** due to insufficient sample size, rather than claiming a proven edge.
  - If any condition fails, state plainly: **"no demonstrated edge"** and stop. No re-tuning, no threshold shifting, no feature additions to rescue the result.
