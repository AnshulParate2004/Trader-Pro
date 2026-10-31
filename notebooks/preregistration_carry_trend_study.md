# Pre-Registration: Structural Edge Study — Funding-Rate Carry & Daily Trend Following

**Date of Pre-Registration:** 2026-10-31 23:30:00 IST  
**Author:** Anshul Parate & AI Quantitative Research Engine  
**Target Asset:** `BTCUSDT`  
**Status:** SEALED BEFORE ANY ANALYSIS OR BACKTESTING  

---

## 1. Research Hypothesis & Structural Edge Rationale

Short-term directional ML models failed to clear exchange friction (−0.05R to −0.30R per trade) due to high transaction turnover and microscopic per-trade signal-to-noise ratios. This study tests two structural trading paradigms designed to overcome fee drag:

1. **Strategy A (Funding-Rate Carry / Delta-Neutral Cash & Carry):**
   - **Structural Source:** Perpetual futures trade at an average premium to spot in cryptocurrency markets because of structural long demand for leverage. Market longs continuously pay shorts funding every 8 hours.
   - **Turnover & Fee Impact:** Low turnover (held continuously or rebalanced over weeks), meaning trading friction is negligible relative to cumulative funding payments collected.
2. **Strategy B (Daily Trend Following):**
   - **Structural Source:** Captures fat-tailed, multi-month directional Bitcoin momentum while cutting downside drawdowns early.
   - **Turnover & Fee Impact:** Very low turnover (~3 to 8 trades per year). Daily bars smooth intraday microstructure noise, and winners compound far larger than small transaction costs.

---

## 2. Pre-Registered Configurations (7 Total Configs)

We strictly pre-register exactly **7 configurations** across the two strategies:

### Strategy A: Funding-Rate Carry (3 Configs)
- **Position:** Long 1.0 BTC Spot + Short 1.0 BTCUSDT Perpetual Futures (delta-neutral).
- **Cash Flow:** Receive $+F_t$ when $F_t > 0$; pay $|F_t|$ when $F_t < 0$, applied every 8 hours (00:00, 08:00, 16:00 UTC).
- **Configurations:**
  - **A1 (Always On):** Enter at inception, stay 100% invested continuously across all periods.
  - **A2 (7-Day Regime Filter):** Long spot + short perp only when trailing 7-day mean funding rate $> 0.010\%$ per 8h (trailing 21 periods mean $F_t > 0.00010$). Otherwise flat.
  - **A3 (30-Day Regime Filter):** Long spot + short perp only when trailing 30-day mean funding rate $> 0.000\%$ (trailing 90 periods mean $F_t > 0$). Otherwise flat.
- **Costs & Execution Friction:**
  - Spot Taker Fee: $0.100\%$ per side.
  - Spot Slippage: $0.010\%$ per side (Total spot leg = $0.110\%$ per side).
  - Perpetual Taker Fee: $0.040\% \times 1.18 = 0.0472\%$ per side.
  - Perpetual Slippage: $0.010\%$ per side (Total perp leg = $0.0572\%$ per side).
  - Basis Slippage: $0.050\%$ per round trip ($0.025\%$ on entry, $0.025\%$ on exit).
  - **Total Round-Trip Entry + Exit Friction:** $(0.110\% + 0.0572\%) \times 2 + 0.050\% = \mathbf{0.3844\%}$ of notional.
  - Applied upon each position transition ($0 \to 1$ or $1 \to 0$).
- **Report Metrics:**
  - Annualized Net Return on Notional (1x notional).
  - Annualized Net Return on Total Capital Deployed (assuming 1.0x spot + 1.0x cash collateral = 2.0x capital model).
  - Max Drawdown of the cumulative net equity stream.
  - Number of entries/exits (turnover).
  - Percentage of 8h funding periods positive.
  - Worst 30-day rolling net return.
  - Count of days / periods where funding was negative.
- **Unmodeled Risks Disclosure:**
  - Counterparty and exchange risk (insolvency of the venue).
  - Short-leg liquidation risk during sudden explosive upside wicks.
  - Margin requirements and capital lockup: holding a short perpetual requires cash margin; return on total capital is roughly half the raw notional funding yield.
  - Basis dislocation risk during extreme market dislocations.

### Strategy B: Daily Trend Following (4 Configs)
- **Data:** BTCUSDT daily bars (UTC close).
- **Execution Timing:** Signals generated at closed daily bar $t$ (`close_time`); executed at next day's open $t+1$ (`open_time`). Strictly zero lookahead.
- **Configurations:**
  - **B1 (100-Day SMA):** Long when $\text{Close}_t > \text{SMA}_{100}(t)$; Flat when $\text{Close}_t \le \text{SMA}_{100}(t)$.
  - **B2 (Donchian 50/20 Breakout):** Enter Long when $\text{Close}_t > \max(\text{High}_{t-50..t-1})$; Exit to Flat when $\text{Close}_t < \min(\text{Low}_{t-20..t-1})$.
  - **B3 (50 / 200 SMA Golden Cross):** Long when $\text{SMA}_{50}(t) > \text{SMA}_{200}(t)$; Flat when $\text{SMA}_{50}(t) \le \text{SMA}_{200}(t)$.
  - **B4 (B2 with Volatility Scaling):** B2 entry/exit logic, with position weight scaled to target $20\%$ annualized volatility:  
    $$w_t = \min\left(1.0, \frac{0.20}{\max(\sigma_{\text{ann}, 30d}, 0.05)}\right)$$  
    where $\sigma_{\text{ann}, 30d} = \text{std}(\text{daily returns})_{30d} \times \sqrt{365}$.
- **Costs & Execution Friction:**
  - Taker Fee: $0.040\% \times 1.18 = 0.0472\%$ per side.
  - Slippage: $0.010\%$ per side.
  - **Transaction Cost:** $0.0572\%$ per side ($0.1144\%$ round trip) applied to turnover: $|\Delta w| \times 0.0572\%$.
  - **Funding Cost While in Position:** Perpetual funding payments paid or received while holding long position: $-\sum_{k=1}^3 F_{t, k} \times w_t$.
- **Report Metrics:**
  - CAGR (Compounded Annual Growth Rate).
  - Annualized Volatility ($\text{std} \times \sqrt{365}$).
  - Sharpe Ratio ($\text{CAGR} / \text{Vol}$, assuming $r_f = 0\%$).
  - Max Drawdown (peak-to-trough decline).
  - Trades per year.
  - True Win Rate (% of completed trades with Net PnL $> 0$).
  - Win/Loss Ratio ($\text{Avg Win} / |\text{Avg Loss}|$).
  - Profit Factor ($\sum \text{Gains} / \sum \text{Losses}$).
  - Time in Market (% of days invested).
  - Benchmark comparison: Buy-and-Hold BTC over the same period.

---

## 3. Data Splits & Leakage Protocol

- **DEV Period:** Inception (2019-09 for funding rates, 2020-01 for daily BTC) to `2022-12-31 23:59:59 UTC`.
- **VALIDATION Period:** `2023-01-01 00:00:00` to `2024-12-31 23:59:59 UTC`.
- **FINAL SEALED WINDOW:** `2025-01-01 00:00:00` to `2026-10-31 UTC`.

**Strict Stop Protocol:**
- Only DEV and VALIDATION are evaluated now.
- One configuration per strategy will be selected based solely on DEV + VALIDATION.
- The FINAL window remains untouched until explicit human authorization is received.

---

## 4. Statistical Methods & Decision Rule

1. **Monthly Block Bootstrap:** 10,000 resamples by calendar month of daily net returns to calculate empirical 95% Confidence Intervals for annualized net return and Sharpe ratio.
2. **Pre-Registered Decision Rule for FINAL:**
   Positive evidence of an edge will be declared **ONLY IF ALL** of the following hold:
   - Net return after realistic costs on FINAL is strictly $> 0$.
   - Monthly block bootstrap 95% Confidence Interval lower bound on FINAL is $\ge 0$.
   - For Strategy B: Max drawdown is substantially better than or competitive with buy-and-hold BTC at comparable returns.

If any criterion fails, the study will report: **"no demonstrated edge"** and stop.
