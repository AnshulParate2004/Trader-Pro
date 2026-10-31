import requests
import zipfile
import io
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

def download_day(date_str):
    url = f"https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-{date_str}.zip"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            df = pd.read_csv(z.open(z.namelist()[0]))
            return df
        return None
    except Exception:
        return None

if __name__ == "__main__":
    t0 = time.time()
    dates = pd.date_range("2021-01-01", "2021-01-31").strftime("%Y-%m-%d").tolist()
    print(f"Testing download for {len(dates)} days...")
    dfs = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_day, d): d for d in dates}
        for f in as_completed(futures):
            res = f.result()
            if res is not None:
                dfs.append(res)
    print(f"Downloaded {len(dfs)} days in {time.time()-t0:.2f}s")
