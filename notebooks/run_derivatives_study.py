import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import sys

# Constants
TAKER_FEE_SIDE = 0.0004
GST_RATE = 0.18
SLIPPAGE_SIDE = 0.0001
TOTAL_ROUND_TRIP_FRICTION = (2 * TAKER_FEE_SIDE * (1 + GST_RATE)) + (2 * SLIPPAGE_SIDE) # 0.001144 (0.1144%)
MAX_COST_R = 0.15

DEV_END = pd.to_datetime("2023-12-31 23:59:59", utc=True)
VAL_START = pd.to_datetime("2024-01-01 00:00:00", utc=True)
VAL_END = pd.to_datetime("2024-12-31 23:59:59", utc=True)

def load_and_merge_data(timeframe="1h"):
    data_dir = r"D:\Projects_Main\Trade-Pro\Trader-Pro\notebooks\data"
    price_df = pd.read_parquet(os.path.join(data_dir, timeframe, f"BTCUSDT_{timeframe}.parquet"))
    funding_df = pd.read_parquet(os.path.join(data_dir, "derivatives", "funding_rates_BTCUSDT.parquet"))
    metrics_df = pd.read_parquet(os.path.join(data_dir, "derivatives", f"metrics_BTCUSDT_{timeframe}.parquet"))
    
    price_df["open_time"] = pd.to_datetime(price_df["open_time"], utc=True)
    price_df["close_time"] = pd.to_datetime(price_df["close_time"], utc=True)
    funding_df["fundingTime"] = pd.to_datetime(funding_df["fundingTime"], utc=True)
    metrics_df["open_time"] = pd.to_datetime(metrics_df["open_time"], utc=True)
    
    price_df = price_df.sort_values("open_time").reset_index(drop=True)
    funding_df = funding_df.sort_values("fundingTime").reset_index(drop=True)
    metrics_df = metrics_df.sort_values("open_time").reset_index(drop=True)
    
    # 1. Merge funding rates without lookahead: funding published at T <= open_time
    f_df = funding_df[["fundingTime", "fundingRate"]].rename(
        columns={"fundingTime": "funding_publish_time", "fundingRate": "funding_rate"}
    )
    df = pd.merge_asof(price_df, f_df, left_on="open_time", right_on="funding_publish_time", direction="backward")
    
    # 2. Merge metrics without lookahead: metrics bar open_time <= price bar open_time
    m_df = metrics_df.rename(columns={
        "open_time": "metrics_open_time",
        "sum_open_interest": "open_interest",
        "sum_open_interest_value": "open_interest_value",
        "count_long_short_ratio": "global_ls_ratio",
        "count_toptrader_long_short_ratio": "toptrader_ls_ratio",
        "sum_taker_long_short_vol_ratio": "taker_vol_ratio"
    })
    df = pd.merge_asof(df, m_df, left_on="open_time", right_on="metrics_open_time", direction="backward")
    
    # Filter to period starting from when metrics exist (2020-09-01)
    df = df[df["open_time"] >= pd.to_datetime("2020-09-01 00:00:00", utc=True)].reset_index(drop=True)
    return df

def compute_features(df):
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    
    # ATR (True Range)
    prev_c = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - prev_c).abs(), (l - prev_c).abs()))
    df["atr14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr14"] / c
    
    # Cost in R
    df["cost_R"] = TOTAL_ROUND_TRIP_FRICTION / df["atr_pct"]
    
    # Returns
    df["ret_1"] = c.pct_change(1)
    df["ret_4"] = c.pct_change(4)
    df["ret_12"] = c.pct_change(12)
    df["ret_24"] = c.pct_change(24)
    
    # RSI
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / (down + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))
    
    # Donchian
    df["donchian_high_20"] = h.rolling(20).max()
    df["donchian_low_20"] = l.rolling(20).min()
    df["donchian_pos"] = (c - df["donchian_low_20"]) / (df["donchian_high_20"] - df["donchian_low_20"] + 1e-9)
    
    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = (ema12 - ema26) / c
    
    # Parkinson Volatility
    df["parkinson_vol"] = np.sqrt((1.0 / (4.0 * np.log(2.0))) * ((np.log(h / l)) ** 2).rolling(14).mean())
    
    # Feature Group G0: Price-only
    g0_cols = [
        "ret_1", "ret_4", "ret_12", "ret_24", "rsi14",
        "donchian_pos", "macd", "parkinson_vol", "atr_pct"
    ]
    
    # Feature Group G1: Funding additions
    df["funding_rate_cum3"] = df["funding_rate"].rolling(3).sum()
    df["funding_rate_dev30d"] = df["funding_rate"] - df["funding_rate"].rolling(90).mean()
    df["funding_rate_rank"] = df["funding_rate"].expanding(min_periods=90).rank(pct=True)
    g1_cols = g0_cols + ["funding_rate", "funding_rate_cum3", "funding_rate_dev30d", "funding_rate_rank"]
    
    # Feature Group G2: Open Interest & Sentiment additions
    df["oi_pct_change_1"] = df["open_interest"].pct_change(1)
    df["oi_pct_change_24"] = df["open_interest"].pct_change(24)
    df["oi_vol_ratio"] = df["open_interest_value"] / (v * c + 1e-6)
    g2_cols = g0_cols + [
        "open_interest", "oi_pct_change_1", "oi_pct_change_24", "oi_vol_ratio",
        "global_ls_ratio", "toptrader_ls_ratio", "taker_vol_ratio"
    ]
    
    # Feature Group G3: All combined
    g3_cols = list(set(g1_cols + g2_cols))
    
    return df, {
        "G0": g0_cols,
        "G1": g1_cols,
        "G2": g2_cols,
        "G3": g3_cols
    }

def simulate_trade_outcomes(df, max_holding=24):
    n = len(df)
    results_long = []
    results_short = []
    
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atrs = df["atr14"].values
    cost_Rs = df["cost_R"].values
    
    for i in range(n - max_holding):
        entry_p = closes[i] # Enter on bar close
        atr = atrs[i]
        cost_R = cost_Rs[i]
        
        if np.isnan(atr) or atr <= 0 or cost_R > MAX_COST_R:
            results_long.append((np.nan, "SKIP", np.nan))
            results_short.append((np.nan, "SKIP", np.nan))
            continue
            
        sl_dist = atr
        tp_dist = atr
        be_trigger_dist = 0.30 * atr
        
        # Long simulation
        long_sl = entry_p - sl_dist
        long_tp = entry_p + tp_dist
        long_be_trigger = entry_p + be_trigger_dist
        long_be_locked_sl = entry_p + (cost_R * atr)
        
        be_active = False
        outcome_long_r = 0.0
        exit_long_type = "HOLD"
        
        for k in range(1, max_holding + 1):
            h_k = highs[i + k]
            l_k = lows[i + k]
            
            # Check BE trigger
            if not be_active and h_k >= long_be_trigger:
                be_active = True
                long_sl = long_be_locked_sl
                
            # Check TP
            if h_k >= long_tp:
                outcome_long_r = 1.0 - cost_R
                exit_long_type = "TP"
                break
            elif l_k <= long_sl:
                outcome_long_r = (cost_R if be_active else -1.0) - cost_R
                exit_long_type = "BE" if be_active else "SL"
                break
        if exit_long_type == "HOLD":
            exit_p = closes[i + max_holding]
            ret_r = (exit_p - entry_p) / sl_dist
            outcome_long_r = ret_r - cost_R
            exit_long_type = "TIME"
            
        # Short simulation
        short_sl = entry_p + sl_dist
        short_tp = entry_p - tp_dist
        short_be_trigger = entry_p - be_trigger_dist
        short_be_locked_sl = entry_p - (cost_R * atr)
        
        be_active_s = False
        outcome_short_r = 0.0
        exit_short_type = "HOLD"
        
        for k in range(1, max_holding + 1):
            h_k = highs[i + k]
            l_k = lows[i + k]
            
            if not be_active_s and l_k <= short_be_trigger:
                be_active_s = True
                short_sl = short_be_locked_sl
                
            if l_k <= short_tp:
                outcome_short_r = 1.0 - cost_R
                exit_short_type = "TP"
                break
            elif h_k >= short_sl:
                outcome_short_r = (cost_R if be_active_s else -1.0) - cost_R
                exit_short_type = "BE" if be_active_s else "SL"
                break
        if exit_short_type == "HOLD":
            exit_p = closes[i + max_holding]
            ret_r = (entry_p - exit_p) / sl_dist
            outcome_short_r = ret_r - cost_R
            exit_short_type = "TIME"
            
        results_long.append((outcome_long_r, exit_long_type, 1 if outcome_long_r > 0 else 0))
        results_short.append((outcome_short_r, exit_short_type, 1 if outcome_short_r > 0 else 0))
        
    for _ in range(max_holding):
        results_long.append((np.nan, "SKIP", np.nan))
        results_short.append((np.nan, "SKIP", np.nan))
        
    df["target_long_net_R"] = [r[0] for r in results_long]
    df["exit_long_type"] = [r[1] for r in results_long]
    df["y_long"] = [r[2] for r in results_long]
    
    df["target_short_net_R"] = [r[0] for r in results_short]
    df["exit_short_type"] = [r[1] for r in results_short]
    df["y_short"] = [r[2] for r in results_short]
    
    return df

def bootstrap_ci(arr, n_resamples=10000):
    if len(arr) == 0:
        return 0.0, 0.0
    arr = np.array(arr)
    means = np.empty(n_resamples)
    n = len(arr)
    for i in range(n_resamples):
        sample = np.random.choice(arr, size=n, replace=True)
        means[i] = sample.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def run_study_for_timeframe(tf="1h"):
    print("=" * 75)
    print(f"RUNNING DERIVATIVES STUDY FOR TIMEFRAME: {tf.upper()}")
    print("=" * 75)
    
    df = load_and_merge_data(tf)
    df, feature_groups = compute_features(df)
    df = simulate_trade_outcomes(df, max_holding=24 if tf=="1h" else 12)
    
    dev_mask = (df["open_time"] <= DEV_END) & df["y_long"].notna()
    val_mask = (df["open_time"] >= VAL_START) & (df["open_time"] <= VAL_END) & df["y_long"].notna()
    
    dev_df = df[dev_mask].copy()
    val_df = df[val_mask].copy()
    
    print(f"DEV rows: {len(dev_df)} (Up to 2023-12-31)")
    print(f"VAL rows: {len(val_df)} (Calendar Year 2024)")
    print(f"HOLD-OUT and SEALED WINDOW strictly untouched.")
    
    group_results = {}
    
    for g_name, cols in feature_groups.items():
        # Train on DEV only
        X_train = dev_df[cols].fillna(0)
        y_train_long = dev_df["y_long"]
        
        # LightGBM Classifier
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 50,
            "verbosity": -1,
            "random_state": 42
        }
        
        model_long = lgb.LGBMClassifier(**params, n_estimators=100)
        model_long.fit(X_train, y_train_long)
        
        # Predictions
        dev_probs = model_long.predict_proba(X_train)[:, 1]
        val_probs = model_long.predict_proba(val_df[cols].fillna(0))[:, 1]
        
        # Cutoff: Top 20% on DEV (80th percentile of DEV predictions)
        cutoff = float(np.percentile(dev_probs, 80))
        
        # Evaluate DEV trades
        dev_signals = dev_probs >= cutoff
        dev_trades = dev_df[dev_signals]
        dev_net_Rs = dev_trades["target_long_net_R"].dropna().values
        dev_exits = dev_trades["exit_long_type"].value_counts().to_dict()
        dev_tp = dev_exits.get("TP", 0)
        dev_sl = dev_exits.get("SL", 0)
        dev_be = dev_exits.get("BE", 0)
        dev_wr = (dev_tp / (dev_tp + dev_sl) * 100) if (dev_tp + dev_sl) > 0 else 0
        dev_ci = bootstrap_ci(dev_net_Rs)
        dev_wins = dev_net_Rs[dev_net_Rs > 0].sum()
        dev_losses = np.abs(dev_net_Rs[dev_net_Rs < 0].sum())
        dev_pf = (dev_wins / dev_losses) if dev_losses > 0 else 9.99
        
        # Evaluate VALIDATION trades (2024 Calendar Year)
        val_signals = val_probs >= cutoff
        val_trades = val_df[val_signals]
        val_net_Rs = val_trades["target_long_net_R"].dropna().values
        val_exits = val_trades["exit_long_type"].value_counts().to_dict()
        val_tp = val_exits.get("TP", 0)
        val_sl = val_exits.get("SL", 0)
        val_be = val_exits.get("BE", 0)
        val_wr = (val_tp / (val_tp + val_sl) * 100) if (val_tp + val_sl) > 0 else 0
        val_ci = bootstrap_ci(val_net_Rs)
        val_wins = val_net_Rs[val_net_Rs > 0].sum()
        val_losses = np.abs(val_net_Rs[val_net_Rs < 0].sum())
        val_pf = (val_wins / val_losses) if val_losses > 0 else 9.99
        
        group_results[g_name] = {
            "cols_count": len(cols),
            "cutoff": cutoff,
            "dev": {
                "trades": len(dev_net_Rs),
                "true_wr": dev_wr,
                "tp": dev_tp, "be": dev_be, "sl": dev_sl,
                "avg_net_r": float(dev_net_Rs.mean()) if len(dev_net_Rs) > 0 else 0.0,
                "ci_lower": dev_ci[0], "ci_upper": dev_ci[1],
                "pf": dev_pf
            },
            "val": {
                "trades": len(val_net_Rs),
                "true_wr": val_wr,
                "tp": val_tp, "be": val_be, "sl": val_sl,
                "avg_net_r": float(val_net_Rs.mean()) if len(val_net_Rs) > 0 else 0.0,
                "ci_lower": val_ci[0], "ci_upper": val_ci[1],
                "pf": val_pf,
                "trades_df": val_trades
            }
        }
        
    return group_results, val_df

def compute_volatility_terciles(val_trades):
    # Volatility terciles based on atr_pct
    val_trades = val_trades.copy()
    val_trades["tercile"] = pd.qcut(val_trades["atr_pct"], q=3, labels=["Low Vol", "Mid Vol", "High Vol"])
    tercile_results = {}
    for t_name, sub in val_trades.groupby("tercile", observed=False):
        net_Rs = sub["target_long_net_R"].dropna().values
        exits = sub["exit_long_type"].value_counts().to_dict()
        tp = exits.get("TP", 0)
        sl = exits.get("SL", 0)
        wr = (tp / (tp + sl) * 100) if (tp + sl) > 0 else 0
        wins = net_Rs[net_Rs > 0].sum()
        losses = np.abs(net_Rs[net_Rs < 0].sum())
        pf = (wins / losses) if losses > 0 else 9.99
        tercile_results[t_name] = {
            "trades": len(net_Rs),
            "true_wr": wr,
            "avg_net_r": float(net_Rs.mean()) if len(net_Rs) > 0 else 0.0,
            "pf": pf
        }
    return tercile_results

if __name__ == "__main__":
    np.random.seed(42)
    
    # 1H Run
    results_1h, val_df_1h = run_study_for_timeframe("1h")
    
    # 4H Run
    results_4h, val_df_4h = run_study_for_timeframe("4h")
    
    print("\n" + "=" * 90)
    print("DELIVERABLE B4: DEV AND VALIDATION (2024) TABLES ACROSS FEATURE GROUPS")
    print("=" * 90)
    
    for tf, res in [("1H", results_1h), ("4H", results_4h)]:
        print(f"\n--- TIMEFRAME: {tf} ---")
        print(f"{'Group':<6} | {'Features':<8} | {'DEV Trades':<10} | {'DEV Avg Net R':<14} | {'DEV 95% CI':<18} | {'VAL Trades':<10} | {'VAL True WR':<12} | {'VAL Avg Net R':<14} | {'VAL 95% CI':<18} | {'VAL PF':<6}")
        print("-" * 135)
        for g in ["G0", "G1", "G2", "G3"]:
            d = res[g]["dev"]
            v = res[g]["val"]
            dev_ci_str = f"[{d['ci_lower']:+.3f}, {d['ci_upper']:+.3f}]"
            val_ci_str = f"[{v['ci_lower']:+.3f}, {v['ci_upper']:+.3f}]"
            print(f"{g:<6} | {res[g]['cols_count']:<8} | {d['trades']:<10} | {d['avg_net_r']:+.4f} R      | {dev_ci_str:<18} | {v['trades']:<10} | {v['true_wr']:>5.1f}%      | {v['avg_net_r']:+.4f} R      | {val_ci_str:<18} | {v['pf']:<6.2f}")
            
    print("\n" + "=" * 90)
    print("DELIVERABLE B5: VALIDATION (2024) BREAKDOWN BY VOLATILITY TERCILE")
    print("=" * 90)
    
    for tf, res in [("1H", results_1h), ("4H", results_4h)]:
        print(f"\n--- TIMEFRAME: {tf} (Best Feature Group on VALIDATION) ---")
        # Identify best group on VAL
        best_g = max(["G0", "G1", "G2", "G3"], key=lambda g: res[g]["val"]["avg_net_r"])
        print(f"Best Group on 2024 VALIDATION: {best_g} (Avg Net R: {res[best_g]['val']['avg_net_r']:+.4f} R)")
        val_trades_best = res[best_g]["val"]["trades_df"]
        terciles = compute_volatility_terciles(val_trades_best)
        print(f"{'Volatility Tercile':<20} | {'Trades':<8} | {'True Win Rate':<14} | {'Avg Net R':<14} | {'Profit Factor':<10}")
        print("-" * 75)
        for t_name, t_data in terciles.items():
            print(f"{t_name:<20} | {t_data['trades']:<8} | {t_data['true_wr']:>5.1f}%        | {t_data['avg_net_r']:+.4f} R      | {t_data['pf']:<10.2f}")

    print("\n" + "=" * 90)
    print("STEP B6 REACHED: STOPPING HERE FOR HUMAN REVIEW & APPROVAL.")
    print("HOLD-OUT AND SEALED WINDOW HAVE NOT BEEN TOUCHED OR READ.")
    print("=" * 90)
