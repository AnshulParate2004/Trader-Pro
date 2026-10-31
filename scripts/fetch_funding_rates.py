import requests
import pandas as pd
import time
import os

def fetch_all_funding_rates(symbol="BTCUSDT"):
    print(f"Fetching full funding rate history for {symbol}...")
    base_url = "https://fapi.binance.com/fapi/v1/fundingRate"
    all_rates = []
    start_time = 1567296000000 # 2019-09-01
    
    while True:
        url = f"{base_url}?symbol={symbol}&limit=1000&startTime={start_time}"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                print(f"Error {r.status_code}: {r.text}")
                break
            data = r.json()
            if not data:
                break
            all_rates.extend(data)
            last_ts = data[-1]["fundingTime"]
            if last_ts == start_time or len(data) < 1000:
                break
            start_time = last_ts + 1
            print(f"Fetched {len(all_rates)} funding rate records (latest: {pd.to_datetime(last_ts, unit='ms')})")
            time.sleep(0.1)
        except Exception as e:
            print(f"Exception: {e}")
            break

    df = pd.DataFrame(all_rates)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    if "markPrice" in df.columns:
        df["markPrice"] = pd.to_numeric(df["markPrice"], errors="coerce")
    df = df.drop_duplicates(subset=["fundingTime"]).sort_values("fundingTime").reset_index(drop=True)
    
    out_dir = r"D:\Projects_Main\Trade-Pro\Trader-Pro\notebooks\data\derivatives"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "funding_rates_BTCUSDT.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} funding rates from {df['fundingTime'].iloc[0]} to {df['fundingTime'].iloc[-1]} to {out_path}")
    return df

if __name__ == "__main__":
    fetch_all_funding_rates()
