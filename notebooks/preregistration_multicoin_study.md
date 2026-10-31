# Multi-Coin Study Pre-Registration Protocol

**Date & Time of Pre-Registration:** 2026-10-31 21:12:00 IST  
**Status:** SEALED BEFORE RUNNING ANY HOLD-OUT EVALUATION  

---

## 1. Objective & Hypothesis
Test whether the pre-registered 4H and 1H LightGBM + Donchian breakout trading setups have a statistically significant net positive edge after honest exchange fees and execution slippage when pooled across 5 liquid cryptocurrency pairs:
1. `BTCUSDT`
2. `ETHUSDT`
3. `SOLUSDT`
4. `BNBUSDT`
5. `XRPUSDT`

---

## 2. Pre-Registered Configurations (3 Setups Across 5 Coins = 15 Configs Total)

Only the following 3 configurations are pre-registered and will be tested:

1. **4H-A:**
   - Timeframe: 4H
   - Horizon: 18 bars (72 hours)
   - Risk/Reward: 1:1 TP / SL
   - Ratchet: Breakeven trigger at +0.30R, locking exact round-trip cost in R
   - Conviction Filter: Top 20% ML prediction probability ($P \ge 80\text{th percentile}$)
2. **1H-A:**
   - Timeframe: 1H
   - Horizon: 24 bars (24 hours)
   - Risk/Reward: 1:1 TP / SL
   - Ratchet: Breakeven trigger at +0.30R, locking exact round-trip cost in R
   - Conviction Filter: Top 20% ML prediction probability ($P \ge 80\text{th percentile}$)
3. **1H-C:**
   - Timeframe: 1H
   - Horizon: 24 bars (24 hours)
   - Risk/Reward: 1:2 TP / SL
   - Ratchet: Breakeven trigger at +0.50R, locking exact round-trip cost in R
   - Conviction Filter: Top 20% ML prediction probability ($P \ge 80\text{th percentile}$)

*Rule:* Zero parameter tuning per coin. Exactly the same parameters apply to all 5 coins. Total configurations tried = **15** (3 configs $\times$ 5 coins).

---

## 3. Honest Fee Models

All trades subtract fees directly in R units:
$$\text{cost\_R} = \frac{\text{Friction Pct}}{\text{Stop Distance Pct}}$$

- **Scenario 1 (Primary — Taker Entry + Taker Exit):**
  - Taker Fee: 0.040% per side
  - GST on fee: 18% ($0.040\% \times 1.18 = 0.0472\%$ per side)
  - Slippage: 0.010% per side
  - Total Round-Trip Friction: $2 \times 0.0472\% + 2 \times 0.010\% = \mathbf{0.1144\%}$
- **Scenario 2 (Secondary — Maker Entry + Taker Exit):**
  - Maker Fee: 0.016% + 18% GST = $0.01888\%$
  - Taker Fee: 0.040% + 18% GST = $0.04720\%$
  - Slippage: 0.010% (taker side only)
  - Total Round-Trip Friction: $0.01888\% + 0.0472\% + 0.010\% = \mathbf{0.07608\%}$

---

## 4. Execution & Anti-Leakage Filters

1. **Stop Distance Filter:** Skip trade if $\text{cost\_R} > 0.15$ (protects capital from low-volatility fee churn).
2. **Expanding Window Threshold:** The top 20% conviction threshold is computed strictly using past predictions up to bar $t$ (`expanding(min_periods=100).quantile(0.80)`), with zero future information.
3. **Point-in-Time Alignment:** Higher-timeframe predictions and indicators are usable only on bars at or after the bar's close time.

---

## 5. Chronological Train / Test Split

- **DEV Window (In-Sample / Tuning):** All bars with `open_time < 2024-05-01 00:00:00 UTC`.
- **SEALED HOLD-OUT Window (Out-of-Sample):** All bars with `open_time >= 2024-05-01 00:00:00 UTC`.
- *Evaluation Protocol:* Each coin and each config is evaluated on the sealed hold-out **exactly once**. No post-hoc tuning or parameter search is permitted after viewing hold-out results.

---

## 6. Statistical Bootstrap Methods

1. **Trade-Level Bootstrap:** 10,000 resamples with replacement to compute empirical 95% Confidence Interval for Average Net R.
2. **Monthly Block Bootstrap:** Resample trades grouped by calendar month (10,000 resamples) to account for cross-coin correlation and market regime clustering.

---

## 7. Decision Rule (Strict Stop Criteria)

Positive evidence of an edge will be declared **ONLY IF**:
1. Pooled Taker Average Net R on Hold-Out $> 0$, **AND**
2. 95% Confidence Interval lower bound (both trade-level and monthly block) is $\ge 0$, **AND**
3. At least 4 of the 5 coins display positive Average Net R on Hold-Out.

If any of these conditions fails, the report will state plainly: **"No demonstrated edge"** and stop.
