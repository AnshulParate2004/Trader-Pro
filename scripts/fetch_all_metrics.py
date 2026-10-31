import requests
import zipfile
import io
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

CACHE_DIR = r"D:\Projects_Main\Trade-Pro\Trader-Pro\notebooks\data\derivatives\raw_cache"
OUT_DIR = r"D:\Projects_Main\Trade-Pro\Trader-Pro\notebooks\data\derivatives"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def fetch_single_day(d_str):
    cache_path = os.path.join(CACHE_DIR, f"{d_str}.parquet")
    if os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass

    url = f"https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-{d_str}.zip"
    try:
        r = requests.get(url, timeout=12)
        if r.status_code == 200:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            df = pd.read_csv(z.open(z.namelist()[0]))
            df["create_time"] = pd.to_datetime(df["create_time"], utc=True)
            for col in ["sum_open_interest", "sum_open_interest_value", "count_toptrader_long_short_ratio",
                        "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df.to_parquet(cache_path, index=False)
            return df
        return None
    except Exception:
        return None

def main():
    dates = pd.date_range("2020-09-01", "2026-10-30").strftime("%Y-%m-%d").tolist()
    print(f"Total days to fetch: {len(dates)}")
    
    t0 = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(fetch_single_day, d): d for d in dates}
        completed = 0
        for f in as_completed(futures):
            df = f.result()
            completed += 1
            if df is not None and not df.empty:
                results.append(df)
            if completed % 250 == 0 or completed == len(dates):
                print(f"Progress: {completed}/{len(dates)} days processed ({len(results)} downloaded) in {time.time()-t0:.1f}s")
                
    if not results:
        print("No metrics downloaded!")
        return

    print("Concatenating all daily metrics...")
    all_df = pd.concat(results, ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["create_time"]).sort_values("create_time").reset_index(drop=True)
    print(f"Total 5m records: {len(all_df)} from {all_df['create_time'].iloc[0]} to {all_df['create_time'].iloc[-1]}")
    
    raw_path = os.path.join(OUT_DIR, "metrics_BTCUSDT_5m.parquet")
    all_df.to_parquet(raw_path, index=False)
    print(f"Saved raw 5m metrics to {raw_path}")
    
    # Resample to 1H without lookahead (5m bars within the hour, last 5m at XX:55 represents bar state)
    print("Resampling to 1H...")
    all_df = all_df.set_index("create_time")
    
    # Each 1H bar opening at T has 5m bars from T+00 to T+55. The bar's close is T+60.
    # Group by floor to 1H
    h_df = pd.DataFrame(index=pd.date_range(all_df.index[0].floor("h"), all_df.index[-1].floor("h"), freq="1h", tz="UTC"))
    
    resampled_1h = all_df.resample("1h").agg({
        "sum_open_interest": "last",
        "sum_open_interest_value": "last",
        "count_toptrader_long_short_ratio": "last",
        "count_long_short_ratio": "last",
        "sum_taker_long_short_vol_ratio": "mean"
    })
    
    # Merge onto full hourly index, forward-fill with max limit of 24 hours (do not forward fill gaps > 24h)
    h_df = h_df.join(resampled_1h)
    h_df = h_df.ffill(limit=24)
    h_df.index.name = "open_time"
    h_df = h_df.reset_index()
    
    out_1h_path = os.path.join(OUT_DIR, "metrics_BTCUSDT_1h.parquet")
    h_df.to_parquet(out_1h_path, index=False)
    print(f"Saved 1H metrics ({len(h_df)} bars) to {out_1h_path}")
    
    # Resample to 4H
    print("Resampling to 4H...")
    resampled_4h = all_df.resample("4h").agg({
        "sum_open_interest": "last",
        "sum_open_interest_value": "last",
        "count_toptrader_long_short_ratio": "last",
        "count_long_short_ratio": "last",
        "sum_taker_long_short_vol_ratio": "mean"
    })
    h4_df = pd.DataFrame(index=pd.date_range(all_df.index[0].floor("4h"), all_df.index[-1].floor("4h"), freq="4h", tz="UTC"))
    h4_df = h4_df.join(resampled_4h).ffill(limit=6) # 6 * 4h = 24h max gap fill
    h4_df.index.name = "open_time"
    h4_df = h4_df.reset_index()
    
    out_4h_path = os.path.join(OUT_DIR, "metrics_BTCUSDT_4h.parquet")
    h4_df.to_parquet(out_4h_path, index=False)
    print(f"Saved 4H metrics ({len(h4_df)} bars) to {out_4h_path}")
    print("Metrics download and alignment complete!")

if __name__ == "__main__":
    main()
