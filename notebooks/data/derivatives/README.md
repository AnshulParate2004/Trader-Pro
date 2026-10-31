# BTCUSDT Derivatives Dataset Documentation

## Data Sources & Provenance
1. **Funding Rate History:**
   - Source: Binance Futures REST API (`/fapi/v1/fundingRate?symbol=BTCUSDT`).
   - Granularity: 8-hour settlement intervals (00:00, 08:00, 16:00 UTC).
   - Date Coverage: `2019-09-10 08:00:00 UTC` to `2026-10-31 16:00:00 UTC`.
   - Records: 7,826 settlements.
   - File: `funding_rates_BTCUSDT.parquet`

2. **Daily Metrics (Open Interest, Long/Short Accounts, Top Trader Accounts, Taker Volume):**
   - Source: `data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-*.zip`.
   - Native Granularity: 5-minute sampling interval.
   - Date Coverage: `2020-09-01 00:00:00 UTC` to `2026-10-30 23:55:00 UTC` (2,251 days).
   - Records: 647,654 5-minute intervals.
   - Columns:
     - `sum_open_interest`: Aggregate open interest in BTC.
     - `sum_open_interest_value`: Aggregate open interest in USD notional.
     - `count_long_short_ratio`: Global long/short account ratio.
     - `count_toptrader_long_short_ratio`: Top trader long/short account ratio.
     - `sum_taker_long_short_vol_ratio`: Taker buy volume / Taker sell volume ratio.
   - Files:
     - `metrics_BTCUSDT_5m.parquet` (647,654 bars)
     - `metrics_BTCUSDT_1h.parquet` (54,024 bars)
     - `metrics_BTCUSDT_4h.parquet` (13,506 bars)

## Forward-Fill Policy
- Maximum forward-fill limit: strictly **24 hours** (`limit=24` on 1H, `limit=6` on 4H).
- Gaps exceeding 24 hours remain `NaN` and are never forward-filled.
