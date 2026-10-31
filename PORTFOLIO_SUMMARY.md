# Quantitative Strategy Audit & Research Pipeline: Bitcoin (BTCUSDT)

I developed an end-to-end quantitative research and automated execution system for Bitcoin (`BTCUSDT`) and conducted a rigorous audit across machine learning and rule-based strategies. 

My initial 15-minute ML scalping model showed an impressive 77.7% gross win rate. However, a deeper audit revealed this was an accounting illusion: the engine was categorizing fee-losing breakeven trades as wins. After modeling realistic Indian exchange friction (0.04% taker fee + 18% GST + 0.01% slippage per side), the true net performance collapsed to −0.30R per trade, confirming negative expectancy.

To test whether any genuine edge existed, I built a leak-tested research pipeline featuring purged walk-forward cross-validation, automated lookahead tests, sequential non-overlapping trade execution, and monthly block bootstrapping. I pre-registered every hypothesis and locked configuration hashes in Git prior to evaluating hold-out data.

Key findings across 5+ years of data:
1. **Intraday ML (15M, 1H, 4H):** Adding derivatives microstructure features (funding rates, open interest, sentiment) reduced fee drag relative to price action, but net expectancy remained negative (−0.05R to −0.30R per trade). No directional edge was detectable at this sample size after realistic costs.
2. **Funding-Rate Carry:** Maintained positive cash flows in 87% of periods, but annualized return on capital compressed from ~8% (2020) to ~2% (2026), underperforming risk-free cash while taking exchange risk.
3. **Daily Trend Following:** A 100-day SMA rule cut max drawdown from −53% to −24% during market downturns (+2.4% CAGR vs −10.2% buy-and-hold), though a 13-trade sample size leaves the result statistically inconclusive.

Building this reinforced that robust data hygiene, honest transaction cost accounting, and pre-registered out-of-sample testing are essential to separating real market behavior from statistical noise.
