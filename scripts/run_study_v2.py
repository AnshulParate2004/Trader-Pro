import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib

# Constants
TAKER_FEE_SIDE = 0.0004
GST_RATE = 0.18
SLIPPAGE_SIDE = 0.0001
TOTAL_ROUND_TRIP_FRICTION = (2 * TAKER_FEE_SIDE * (1 + GST_RATE)) + (2 * SLIPPAGE_SIDE) # 0.001144 (0.1144%)
MAX_COST_R = 0.15

DEV_END = pd.to_datetime("2023-12-31 23:59:59", utc=True)
VAL_START = pd.to_datetime("2024-01-01 00:00:00", utc=True)
VAL_END = pd.to_datetime("2024-12-31 23:59:59", utc=True)
HOLDOUT_START = pd.to_datetime("2025-01-01 00:00:00", utc=True)
HOLDOUT_END = pd.to_datetime("2026-08-31 23:59:59", utc=True)
SEALED_START = pd.to_datetime("2026-09-01 00:00:00", utc=True)
SEALED_END = pd.to_datetime("2026-10-31 23:59:59", utc=True)

def load_data():
    data_dir = r"D:\Projects_Main\Trade-Pro\Trader-Pro\notebooks\data"
    price_df = pd.read_parquet(os.path.join(data_dir, "4h", "BTCUSDT_4h.parquet"))
    funding_df = pd.read_parquet(os.path.join(data_dir, "derivatives", "funding_rates_BTCUSDT.parquet"))
    metrics_df = pd.read_parquet(os.path.join(data_dir, "derivatives", "metrics_BTCUSDT_4h.parquet"))
    
    price_df["open_time"] = pd.to_datetime(price_df["open_time"], utc=True)
    price_df["close_time"] = pd.to_datetime(price_df["close_time"], utc=True)
    funding_df["fundingTime"] = pd.to_datetime(funding_df["fundingTime"], utc=True)
    metrics_df["open_time"] = pd.to_datetime(metrics_df["open_time"], utc=True)
    
    price_df = price_df.sort_values("open_time").reset_index(drop=True)
    funding_df = funding_df.sort_values("fundingTime").reset_index(drop=True)
    metrics_df = metrics_df.sort_values("open_time").reset_index(drop=True)
    
    # Merge funding rates without lookahead
    f_df = funding_df[["fundingTime", "fundingRate"]].rename(
        columns={"fundingTime": "funding_publish_time", "fundingRate": "funding_rate"}
    )
    df = pd.merge_asof(price_df, f_df, left_on="open_time", right_on="funding_publish_time", direction="backward")
    
    # Merge metrics without lookahead
    m_df = metrics_df.rename(columns={
        "open_time": "metrics_open_time",
        "sum_open_interest": "open_interest",
        "sum_open_interest_value": "open_interest_value",
        "count_long_short_ratio": "global_ls_ratio",
        "count_toptrader_long_short_ratio": "toptrader_ls_ratio",
        "sum_taker_long_short_vol_ratio": "taker_vol_ratio"
    })
    df = pd.merge_asof(df, m_df, left_on="open_time", right_on="metrics_open_time", direction="backward")
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
    df["cost_R"] = TOTAL_ROUND_TRIP_FRICTION / df["atr_pct"]
    
    # Technicals
    df["ret_1"] = c.pct_change(1)
    df["ret_4"] = c.pct_change(4)
    df["ret_12"] = c.pct_change(12)
    df["ret_24"] = c.pct_change(24)
    
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / (down + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))
    
    df["donchian_high_20"] = h.rolling(20).max()
    df["donchian_low_20"] = l.rolling(20).min()
    df["donchian_pos"] = (c - df["donchian_low_20"]) / (df["donchian_high_20"] - df["donchian_low_20"] + 1e-9)
    
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = (ema12 - ema26) / c
    df["parkinson_vol"] = np.sqrt((1.0 / (4.0 * np.log(2.0))) * ((np.log(h / l)) ** 2).rolling(14).mean())
    
    # Group G0
    g0_cols = ["ret_1", "ret_4", "ret_12", "ret_24", "rsi14", "donchian_pos", "macd", "parkinson_vol", "atr_pct"]
    
    # Group G1
    df["funding_rate_cum3"] = df["funding_rate"].rolling(3).sum()
    df["funding_rate_dev30d"] = df["funding_rate"] - df["funding_rate"].rolling(90).mean()
    df["funding_rate_rank"] = df["funding_rate"].expanding(min_periods=90).rank(pct=True)
    g1_cols = g0_cols + ["funding_rate", "funding_rate_cum3", "funding_rate_dev30d", "funding_rate_rank"]
    
    # Group G2
    df["oi_pct_change_1"] = df["open_interest"].pct_change(1)
    df["oi_pct_change_24"] = df["open_interest"].pct_change(24)
    df["oi_vol_ratio"] = df["open_interest_value"] / (v * c + 1e-6)
    g2_cols = g0_cols + [
        "open_interest", "oi_pct_change_1", "oi_pct_change_24", "oi_vol_ratio",
        "global_ls_ratio", "toptrader_ls_ratio", "taker_vol_ratio"
    ]
    
    # Group G3
    g3_cols = list(set(g1_cols + g2_cols))
    
    return df, {"G0": g0_cols, "G1": g1_cols, "G2": g2_cols, "G3": g3_cols}

def compute_labels(df, max_holding=12):
    n = len(df)
    y_long = []
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atrs = df["atr14"].values
    cost_Rs = df["cost_R"].values
    
    for i in range(n - max_holding):
        entry_p = closes[i]
        atr = atrs[i]
        cost_R = cost_Rs[i]
        
        if np.isnan(atr) or atr <= 0 or cost_R > MAX_COST_R:
            y_long.append(np.nan)
            continue
            
        sl_dist = atr
        tp_dist = atr
        be_trigger = entry_p + 0.30 * atr
        be_locked_sl = entry_p + (cost_R * atr)
        
        long_sl = entry_p - sl_dist
        long_tp = entry_p + tp_dist
        be_active = False
        outcome_r = 0.0
        exit_found = False
        
        for k in range(1, max_holding + 1):
            h_k = highs[i + k]
            l_k = lows[i + k]
            
            if not be_active and h_k >= be_trigger:
                be_active = True
                long_sl = be_locked_sl
                
            if h_k >= long_tp:
                outcome_r = 1.0 - cost_R
                exit_found = True
                break
            elif l_k <= long_sl:
                outcome_r = (cost_R if be_active else -1.0) - cost_R
                exit_found = True
                break
                
        if not exit_found:
            exit_p = closes[i + max_holding]
            outcome_r = ((exit_p - entry_p) / sl_dist) - cost_R
            
        y_long.append(1 if outcome_r > 0 else 0)
        
    for _ in range(max_holding):
        y_long.append(np.nan)
        
    df["y_long"] = y_long
    return df

def simulate_one_trade_at_a_time(sub_df, signals, max_holding=12):
    n = len(sub_df)
    closes = sub_df["close"].values
    highs = sub_df["high"].values
    lows = sub_df["low"].values
    atrs = sub_df["atr14"].values
    cost_Rs = sub_df["cost_R"].values
    open_times = sub_df["open_time"].values
    
    trades = []
    total_signals = 0
    overlapping_signals = 0
    current_exit_bar = -1
    
    for i in range(n - max_holding):
        if signals[i]:
            total_signals += 1
            if i <= current_exit_bar:
                overlapping_signals += 1
                continue
                
            entry_p = closes[i]
            atr = atrs[i]
            cost_R = cost_Rs[i]
            
            if np.isnan(atr) or atr <= 0 or cost_R > MAX_COST_R:
                continue
                
            sl_dist = atr
            tp_dist = atr
            be_trigger = entry_p + 0.30 * atr
            be_locked_sl = entry_p + (cost_R * atr)
            
            long_sl = entry_p - sl_dist
            long_tp = entry_p + tp_dist
            be_active = False
            outcome_r = 0.0
            exit_type = "HOLD"
            exit_bar = i + max_holding
            exit_time = open_times[i + max_holding]
            exit_p = closes[i + max_holding]
            
            for k in range(1, max_holding + 1):
                h_k = highs[i + k]
                l_k = lows[i + k]
                
                if not be_active and h_k >= be_trigger:
                    be_active = True
                    long_sl = be_locked_sl
                    
                if h_k >= long_tp:
                    outcome_r = 1.0 - cost_R
                    exit_type = "TP"
                    exit_bar = i + k
                    exit_p = long_tp
                    exit_time = open_times[i + k]
                    break
                elif l_k <= long_sl:
                    outcome_r = (cost_R if be_active else -1.0) - cost_R
                    exit_type = "BE" if be_active else "SL"
                    exit_bar = i + k
                    exit_p = long_sl
                    exit_time = open_times[i + k]
                    break
                    
            if exit_type == "HOLD":
                exit_p = closes[i + max_holding]
                outcome_r = ((exit_p - entry_p) / sl_dist) - cost_R
                exit_type = "TIME"
                
            current_exit_bar = exit_bar
            
            trades.append({
                "entry_bar": i,
                "exit_bar": exit_bar,
                "entry_time": open_times[i],
                "exit_time": exit_time,
                "entry_p": entry_p,
                "exit_p": exit_p,
                "exit_type": exit_type,
                "net_R": outcome_r,
                "cost_R": cost_R,
                "gross_R": outcome_r + cost_R
            })
            
    return trades, total_signals, overlapping_signals

def monthly_block_bootstrap(trades, n_resamples=10000):
    if len(trades) == 0:
        return 0.0, 0.0
    df_t = pd.DataFrame(trades)
    df_t["ym"] = pd.to_datetime(df_t["entry_time"]).dt.to_period("M").astype(str)
    months = df_t["ym"].unique()
    n_m = len(months)
    
    if n_m <= 1:
        arr = df_t["net_R"].values
        means = np.array([np.random.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_resamples)])
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
        
    month_dict = {m: df_t[df_t["ym"] == m]["net_R"].values for m in months}
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        s_months = np.random.choice(months, size=n_m, replace=True)
        pooled = np.concatenate([month_dict[m] for m in s_months])
        means[i] = pooled.mean()
        
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def compute_metrics(trades):
    if len(trades) == 0:
        return {"trades": 0, "true_wr": 0.0, "tp": 0, "be": 0, "sl": 0, "time": 0, "avg_net_r": 0.0, "avg_gross_r": 0.0, "pf": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "max_dd": 0.0}
    df_t = pd.DataFrame(trades)
    exits = df_t["exit_type"].value_counts().to_dict()
    tp = exits.get("TP", 0)
    be = exits.get("BE", 0)
    sl = exits.get("SL", 0)
    time_exits = exits.get("TIME", 0)
    
    true_wr = (tp / (tp + sl) * 100) if (tp + sl) > 0 else 0.0
    net_wr = ((df_t["net_R"] > 0).sum() / len(df_t) * 100)
    avg_net_r = float(df_t["net_R"].mean())
    avg_gross_r = float(df_t["gross_R"].mean())
    
    wins = df_t[df_t["net_R"] > 0]["net_R"].sum()
    losses = np.abs(df_t[df_t["net_R"] < 0]["net_R"].sum())
    pf = (wins / losses) if losses > 0 else 9.99
    
    ci_l, ci_u = monthly_block_bootstrap(trades)
    
    # Max Drawdown
    cum = np.cumsum(df_t["net_R"].values)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0
    
    return {
        "trades": len(trades),
        "true_wr": true_wr,
        "net_wr": net_wr,
        "tp": tp, "be": be, "sl": sl, "time": time_exits,
        "avg_net_r": avg_net_r,
        "avg_gross_r": avg_gross_r,
        "pf": pf,
        "ci_lower": ci_l,
        "ci_upper": ci_u,
        "max_dd": max_dd
    }

def run_pre_checks_and_evaluation():
    np.random.seed(42)
    df = load_data()
    df, feature_groups = compute_features(df)
    df = compute_labels(df, max_holding=12)
    
    # 1. Splits
    dev_mask = (df["open_time"] <= DEV_END) & df["y_long"].notna()
    val_mask = (df["open_time"] >= VAL_START) & (df["open_time"] <= VAL_END) & df["y_long"].notna()
    holdout_mask = (df["open_time"] >= HOLDOUT_START) & (df["open_time"] <= HOLDOUT_END) & df["y_long"].notna()
    sealed_mask = (df["open_time"] >= SEALED_START) & (df["open_time"] <= SEALED_END) & df["y_long"].notna()
    
    dev_df = df[dev_mask].copy().reset_index(drop=True)
    val_df = df[val_mask].copy().reset_index(drop=True)
    holdout_df = df[holdout_mask].copy().reset_index(drop=True)
    sealed_df = df[sealed_mask].copy().reset_index(drop=True)
    
    print(f"Dataset Row Counts:")
    print(f"  DEV:        {len(dev_df)} rows ({dev_df['open_time'].min()} to {dev_df['open_time'].max()})")
    print(f"  VALIDATION: {len(val_df)} rows ({val_df['open_time'].min()} to {val_df['open_time'].max()})")
    print(f"  HOLD-OUT:   {len(holdout_df)} rows ({holdout_df['open_time'].min()} to {holdout_df['open_time'].max()})")
    print(f"  SEALED:     {len(sealed_df)} rows ({sealed_df['open_time'].min()} to {sealed_df['open_time'].max()})")
    
    # Load G-orig out-of-fold predictions
    oos_4h = pd.read_parquet(r"D:\Projects_Main\Trade-Pro\Trader-Pro\notebooks\model\4h\oos_predictions.parquet")
    btc4h = pd.read_parquet(r"D:\Projects_Main\Trade-Pro\Trader-Pro\notebooks\data\4h\BTCUSDT_4h.parquet")
    btc4h["open_time"] = pd.to_datetime(btc4h["open_time"], utc=True)
    oos_merged = btc4h.iloc[oos_4h["pos"]].copy().reset_index(drop=True)
    oos_merged["pL"] = oos_4h["pL"].values
    oos_merged["thrL"] = oos_4h["thrL"].values
    
    # Pre-Checks Models Training
    lgb_params = {
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
    
    results = {}
    
    for g_name in ["G0", "G1", "G2", "G3"]:
        cols = feature_groups[g_name]
        
        # 1. Walk-forward Purged CV on DEV (5 folds)
        dev_oof_probs = np.zeros(len(dev_df))
        n_dev = len(dev_df)
        fold_size = n_dev // 5
        purge = 12
        
        for f in range(1, 5):
            train_end_idx = f * fold_size
            test_start_idx = train_end_idx + purge
            test_end_idx = min(n_dev, (f + 1) * fold_size)
            if test_start_idx >= test_end_idx:
                continue
            
            X_tr = dev_df.iloc[:train_end_idx][cols].fillna(0)
            y_tr = dev_df.iloc[:train_end_idx]["y_long"]
            m_f = lgb.LGBMClassifier(**lgb_params, n_estimators=100)
            m_f.fit(X_tr, y_tr)
            
            X_te = dev_df.iloc[test_start_idx:test_end_idx][cols].fillna(0)
            dev_oof_probs[test_start_idx:test_end_idx] = m_f.predict_proba(X_te)[:, 1]
            
        # 2. In-sample fit on all DEV for final model
        X_dev_all = dev_df[cols].fillna(0)
        y_dev_all = dev_df["y_long"]
        final_model = lgb.LGBMClassifier(**lgb_params, n_estimators=100)
        final_model.fit(X_dev_all, y_dev_all)
        
        dev_in_sample_probs = final_model.predict_proba(X_dev_all)[:, 1]
        val_probs = final_model.predict_proba(val_df[cols].fillna(0))[:, 1]
        holdout_probs = final_model.predict_proba(holdout_df[cols].fillna(0))[:, 1]
        sealed_probs = final_model.predict_proba(sealed_df[cols].fillna(0))[:, 1]
        
        # Conviction Cutoff: Top 20% on DEV (80th percentile)
        cutoff_in = float(np.percentile(dev_in_sample_probs, 80))
        valid_oof_mask = dev_oof_probs > 0
        cutoff_oof = float(np.percentile(dev_oof_probs[valid_oof_mask], 80)) if valid_oof_mask.sum() > 0 else cutoff_in
        
        # Run one-trade-at-a-time on DEV
        dev_trades_in, dev_tot_sig_in, dev_ovl_sig_in = simulate_one_trade_at_a_time(dev_df, dev_in_sample_probs >= cutoff_in)
        dev_trades_oof, dev_tot_sig_oof, dev_ovl_sig_oof = simulate_one_trade_at_a_time(dev_df, (dev_oof_probs >= cutoff_oof) & valid_oof_mask)
        
        # Run one-trade-at-a-time on VAL
        val_trades, val_tot_sig, val_ovl_sig = simulate_one_trade_at_a_time(val_df, val_probs >= cutoff_in)
        
        # Run one-trade-at-a-time on HOLDOUT
        holdout_trades, holdout_tot_sig, holdout_ovl_sig = simulate_one_trade_at_a_time(holdout_df, holdout_probs >= cutoff_in)
        
        # Run one-trade-at-a-time on SEALED
        sealed_trades, sealed_tot_sig, sealed_ovl_sig = simulate_one_trade_at_a_time(sealed_df, sealed_probs >= cutoff_in)
        
        results[g_name] = {
            "model": final_model,
            "cutoff": cutoff_in,
            "dev_in": compute_metrics(dev_trades_in),
            "dev_oof": compute_metrics(dev_trades_oof),
            "val": compute_metrics(val_trades),
            "val_overlap": {"total": val_tot_sig, "overlapping": val_ovl_sig, "trades": len(val_trades)},
            "holdout": compute_metrics(holdout_trades),
            "sealed": compute_metrics(sealed_trades),
            "holdout_trades_raw": holdout_trades,
            "sealed_trades_raw": sealed_trades
        }

    # Evaluate G-orig baseline
    val_merged_gorig = pd.merge(val_df[["open_time", "close", "high", "low", "atr14", "cost_R"]], oos_merged[["open_time", "pL", "thrL"]], on="open_time", how="inner")
    val_gorig_signals = (val_merged_gorig["pL"] >= val_merged_gorig["thrL"]).values
    gorig_val_trades, gorig_val_tot, gorig_val_ovl = simulate_one_trade_at_a_time(val_merged_gorig, val_gorig_signals)
    gorig_val_metrics = compute_metrics(gorig_val_trades)
    
    holdout_merged_gorig = pd.merge(holdout_df[["open_time", "close", "high", "low", "atr14", "cost_R"]], oos_merged[["open_time", "pL", "thrL"]], on="open_time", how="inner")
    holdout_gorig_signals = (holdout_merged_gorig["pL"] >= holdout_merged_gorig["thrL"]).values
    gorig_holdout_trades, gorig_ho_tot, gorig_ho_ovl = simulate_one_trade_at_a_time(holdout_merged_gorig, holdout_gorig_signals)
    gorig_holdout_metrics = compute_metrics(gorig_holdout_trades)
    
    sealed_merged_gorig = pd.merge(sealed_df[["open_time", "close", "high", "low", "atr14", "cost_R"]], oos_merged[["open_time", "pL", "thrL"]], on="open_time", how="inner")
    sealed_gorig_signals = (sealed_merged_gorig["pL"] >= sealed_merged_gorig["thrL"]).values
    gorig_sealed_trades, gorig_se_tot, gorig_se_ovl = simulate_one_trade_at_a_time(sealed_merged_gorig, sealed_gorig_signals)
    gorig_sealed_metrics = compute_metrics(gorig_sealed_trades)
    
    results["G-orig"] = {
        "val": gorig_val_metrics,
        "val_overlap": {"total": gorig_val_tot, "overlapping": gorig_val_ovl, "trades": len(gorig_val_trades)},
        "holdout": gorig_holdout_metrics,
        "sealed": gorig_sealed_metrics,
        "holdout_trades_raw": gorig_holdout_trades,
        "sealed_trades_raw": gorig_sealed_trades
    }
    
    return results

if __name__ == "__main__":
    results = run_pre_checks_and_evaluation()
    
    print("\n" + "=" * 110)
    print("PRE-CHECK 1 & 2: 4H VALIDATION (2024) ONE-TRADE-AT-A-TIME & OVERLAP REPORT")
    print("=" * 110)
    print(f"{'Group':<8} | {'Total Sigs':<10} | {'Ovl Sigs':<10} | {'% Overlap':<10} | {'Unique Trades':<14} | {'VAL True WR':<12} | {'VAL Avg Net R':<14} | {'VAL Monthly 95% CI':<20} | {'VAL PF':<6}")
    print("-" * 115)
    for g in ["G0", "G1", "G2", "G3", "G-orig"]:
        v = results[g]["val"]
        ovl = results[g]["val_overlap"]
        ovl_pct = (ovl["overlapping"] / ovl["total"] * 100) if ovl["total"] > 0 else 0
        ci_str = f"[{v['ci_lower']:+.3f}, {v['ci_upper']:+.3f}]"
        print(f"{g:<8} | {ovl['total']:<10} | {ovl['overlapping']:<10} | {ovl_pct:>8.1f}% | {v['trades']:<14} | {v['true_wr']:>9.1f}%  | {v['avg_net_r']:+.4f} R      | {ci_str:<20} | {v['pf']:<6.2f}")
        
    print("\n" + "=" * 110)
    print("PRE-CHECK 3: DEV IN-SAMPLE VS. WALK-FORWARD OUT-OF-SAMPLE")
    print("=" * 110)
    print(f"{'Group':<8} | {'DEV In-Sample Trades':<20} | {'DEV In-Sample Avg Net R':<24} | {'DEV OOS Trades':<16} | {'DEV OOS Avg Net R':<20} | {'DEV OOS 95% CI':<20}")
    print("-" * 110)
    for g in ["G0", "G1", "G2", "G3"]:
        d_in = results[g]["dev_in"]
        d_oof = results[g]["dev_oof"]
        ci_str = f"[{d_oof['ci_lower']:+.3f}, {d_oof['ci_upper']:+.3f}]"
        print(f"{g:<8} | {d_in['trades']:<20} | {d_in['avg_net_r']:+.4f} R                 | {d_oof['trades']:<16} | {d_oof['avg_net_r']:+.4f} R             | {ci_str:<20}")

    print("\n" + "=" * 110)
    print("STEP 5: EVALUATION ON HOLD-OUT (2025-01-01 to 2026-08-31)")
    print("=" * 110)
    print(f"{'Group':<8} | {'Trades':<8} | {'True WR':<10} | {'TP/BE/SL Split':<20} | {'Avg Gross R':<12} | {'Avg Net R':<12} | {'Net PF':<8} | {'Max DD':<10} | {'Monthly 95% CI':<20}")
    print("-" * 115)
    for g in ["G2", "G-orig"]:
        ho = results[g]["holdout"]
        split = f"{ho['tp']} TP / {ho['be']} BE / {ho['sl']} SL"
        ci_str = f"[{ho['ci_lower']:+.3f}, {ho['ci_upper']:+.3f}]"
        print(f"{g:<8} | {ho['trades']:<8} | {ho['true_wr']:>7.1f}%  | {split:<20} | {ho['avg_gross_r']:+.4f} R   | {ho['avg_net_r']:+.4f} R  | {ho['pf']:<8.2f} | {ho['max_dd']:+.2f} R    | {ci_str:<20}")

    # Holdout Net R by Year
    print("\nHOLD-OUT Net R Breakdown by Year:")
    for g in ["G2", "G-orig"]:
        df_ho = pd.DataFrame(results[g]["holdout_trades_raw"])
        if len(df_ho) > 0:
            df_ho["year"] = pd.to_datetime(df_ho["entry_time"]).dt.year
            for y, grp in df_ho.groupby("year"):
                tp = (grp["exit_type"] == "TP").sum()
                sl = (grp["exit_type"] == "SL").sum()
                wr = (tp / (tp + sl) * 100) if (tp + sl) > 0 else 0.0
                print(f"  {g} - {y}: {len(grp)} trades | True WR {wr:.1f}% | Avg Net R {grp['net_R'].mean():+.4f} R | Total Net R {grp['net_R'].sum():+.2f} R")

    print("\n" + "=" * 110)
    print("STEP 6: EVALUATION ON SEALED FINAL WINDOW (2026-09-01 to 2026-10-31)")
    print("=" * 110)
    print(f"{'Group':<8} | {'Trades':<8} | {'True WR':<10} | {'TP/BE/SL Split':<20} | {'Avg Gross R':<12} | {'Avg Net R':<12} | {'Net PF':<8} | {'Monthly 95% CI':<20} | {'Status':<15}")
    print("-" * 125)
    for g in ["G2", "G-orig"]:
        se = results[g]["sealed"]
        split = f"{se['tp']} TP / {se['be']} BE / {se['sl']} SL"
        ci_str = f"[{se['ci_lower']:+.3f}, {se['ci_upper']:+.3f}]"
        ci_width = se["ci_upper"] - se["ci_lower"]
        status = "INCONCLUSIVE" if ci_width > 0.20 or se["trades"] < 25 else ("POSITIVE" if se["avg_net_r"] > 0 else "NEGATIVE")
        print(f"{g:<8} | {se['trades']:<8} | {se['true_wr']:>7.1f}%  | {split:<20} | {se['avg_gross_r']:+.4f} R   | {se['avg_net_r']:+.4f} R  | {se['pf']:<8.2f} | {ci_str:<20} | {status:<15} (CI width {ci_width:.3f}R)")

    # Combined Holdout + Sealed Monthly Block Bootstrap CI
    print("\n" + "=" * 110)
    print("STEP 7: POOLED (HOLD-OUT + SEALED) BOOTSTRAP CI & DECISION RULE EVALUATION")
    print("=" * 110)
    for g in ["G2", "G-orig"]:
        pooled_trades = results[g]["holdout_trades_raw"] + results[g]["sealed_trades_raw"]
        pooled_metrics = compute_metrics(pooled_trades)
        ci_str = f"[{pooled_metrics['ci_lower']:+.3f}, {pooled_metrics['ci_upper']:+.3f}]"
        ci_width = pooled_metrics["ci_upper"] - pooled_metrics["ci_lower"]
        print(f"{g}: {pooled_metrics['trades']} total pooled trades | Avg Net R: {pooled_metrics['avg_net_r']:+.4f} R | Monthly 95% CI: {ci_str} (Width: {ci_width:.3f}R)")
