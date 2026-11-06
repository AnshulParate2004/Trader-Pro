import pandas as pd
import numpy as np
import os

# ==============================================================================
# COST CONSTANTS
# ==============================================================================
# Strategy A: Carry
SPOT_TAKER_FEE = 0.00100       # 0.10%
SPOT_SLIPPAGE = 0.00010        # 0.01%
PERP_TAKER_FEE = 0.00040 * 1.18 # 0.04% + 18% GST = 0.0472%
PERP_SLIPPAGE = 0.00010        # 0.01%
BASIS_SLIPPAGE_HALF = 0.00025  # 0.025% per side (0.05% round trip)

CARRY_ENTRY_FEE = (SPOT_TAKER_FEE + SPOT_SLIPPAGE) + (PERP_TAKER_FEE + PERP_SLIPPAGE) + BASIS_SLIPPAGE_HALF # 0.001922 (0.1922%)
CARRY_EXIT_FEE = CARRY_ENTRY_FEE # 0.001922

# Strategy B: Trend
TREND_TAKER_FEE_SIDE = PERP_TAKER_FEE + PERP_SLIPPAGE # 0.000572 (0.0572% per side)

# Dates
DEV_END = pd.to_datetime("2022-12-31 23:59:59", utc=True)
VAL_START = pd.to_datetime("2023-01-01 00:00:00", utc=True)
VAL_END = pd.to_datetime("2024-12-31 23:59:59", utc=True)

# ==============================================================================
# DATA LOADERS
# ==============================================================================
def load_data():
    base_dir = r"D:\Projects_Main\Trade-Pro\Trader-Pro\notebooks\data"
    fr_path = os.path.join(base_dir, "derivatives", "funding_rates_BTCUSDT.parquet")
    d1_path = os.path.join(base_dir, "1d", "BTCUSDT_1d.parquet")
    
    df_fr = pd.read_parquet(fr_path)
    df_fr["fundingTime"] = pd.to_datetime(df_fr["fundingTime"], utc=True)
    df_fr = df_fr.sort_values("fundingTime").reset_index(drop=True)
    
    df_1d = pd.read_parquet(d1_path)
    df_1d["open_time"] = pd.to_datetime(df_1d["open_time"], utc=True)
    df_1d["close_time"] = pd.to_datetime(df_1d["close_time"], utc=True)
    df_1d = df_1d.sort_values("open_time").reset_index(drop=True)
    
    # Merge daily funding rate into 1D
    df_fr["date"] = df_fr["fundingTime"].dt.floor("D")
    daily_fr = df_fr.groupby("date")["fundingRate"].sum().reset_index()
    daily_fr.rename(columns={"fundingRate": "daily_funding_rate"}, inplace=True)
    
    df_1d = pd.merge(df_1d, daily_fr, left_on="open_time", right_on="date", how="left")
    df_1d["daily_funding_rate"] = df_1d["daily_funding_rate"].fillna(0.0)
    
    return df_fr, df_1d

# ==============================================================================
# STRATEGY A: FUNDING-RATE CARRY
# ==============================================================================
def simulate_carry(df_fr, config):
    df = df_fr.copy()
    
    # Point-in-time trailing mean calculation
    # 7 days = 21 periods; 30 days = 90 periods
    df["fr_7d"] = df["fundingRate"].shift(1).rolling(21).mean()
    df["fr_30d"] = df["fundingRate"].shift(1).rolling(90).mean()
    
    if config == "A1":
        # Always on
        df["target_pos"] = 1
    elif config == "A2":
        # Trailing 7-day mean > 0.01% per 8h (0.00010)
        df["target_pos"] = (df["fr_7d"] > 0.00010).astype(int)
    elif config == "A3":
        # Trailing 30-day mean > 0
        df["target_pos"] = (df["fr_30d"] > 0.0).astype(int)
    else:
        raise ValueError(f"Unknown carry config: {config}")
        
    return df

def evaluate_carry_split(df_sim, start_time, end_time):
    mask = (df_sim["fundingTime"] >= start_time) & (df_sim["fundingTime"] <= end_time)
    sub = df_sim[mask].copy().reset_index(drop=True)
    
    if len(sub) == 0:
        return {}
        
    n = len(sub)
    pos = sub["target_pos"].values
    fr = sub["fundingRate"].values
    times = sub["fundingTime"].values
    
    net_cf = np.zeros(n)
    gross_cf = np.zeros(n)
    friction = np.zeros(n)
    
    entries = 0
    exits = 0
    current_pos = 0
    
    for i in range(n):
        target = pos[i]
        f = 0.0
        
        # Check transition
        if target == 1 and current_pos == 0:
            f += CARRY_ENTRY_FEE
            entries += 1
            current_pos = 1
        elif target == 0 and current_pos == 1:
            f += CARRY_EXIT_FEE
            exits += 1
            current_pos = 0
            
        friction[i] = f
        # Positive funding = short gets paid; negative funding = short pays
        gross = current_pos * fr[i]
        gross_cf[i] = gross
        net_cf[i] = gross - f
        
    # If still in position at the end, pay exit fee to cleanly liquidate
    if current_pos == 1:
        net_cf[-1] -= CARRY_EXIT_FEE
        friction[-1] += CARRY_EXIT_FEE
        exits += 1
        
    sub["net_cf"] = net_cf
    sub["gross_cf"] = gross_cf
    sub["friction"] = friction
    
    # Cumulative notional equity
    cum_net = np.cumsum(net_cf)
    peak = np.maximum.accumulate(cum_net)
    dd = cum_net - peak
    max_dd_notional = float(np.min(dd)) if len(dd) > 0 else 0.0
    
    # Time elapsed in years
    days = (pd.to_datetime(times[-1]) - pd.to_datetime(times[0])).total_seconds() / 86400.0
    years = max(days / 365.25, 0.001)
    
    tot_net_notional = float(np.sum(net_cf))
    ann_return_notional = tot_net_notional / years * 100.0 # Annualized % on notional
    ann_return_2x_capital = (tot_net_notional / 2.0) / years * 100.0 # Annualized % on 2x collateral capital
    
    # Max drawdown on 2x capital
    max_dd_2x_capital = (max_dd_notional / 2.0) * 100.0
    
    # Funding periods stats
    pct_fr_pos_total = float((fr > 0).mean() * 100.0)
    invested_mask = pos == 1
    pct_fr_pos_invested = float((fr[invested_mask] > 0).mean() * 100.0) if invested_mask.sum() > 0 else 0.0
    
    # Negative funding days count
    # Group by calendar day to find days where daily net funding was negative
    sub["date"] = pd.to_datetime(sub["fundingTime"]).dt.floor("D")
    daily_fr = sub.groupby("date")["fundingRate"].sum()
    neg_funding_days = int((daily_fr < 0).sum())
    total_days = len(daily_fr)
    
    # Worst 30-day rolling net return (rolling 90 8h periods)
    sub["rolling_30d_net"] = sub["net_cf"].rolling(90).sum()
    worst_30d_notional = float(sub["rolling_30d_net"].min() * 100.0) if len(sub) >= 90 else float(tot_net_notional * 100.0)
    worst_30d_2x_capital = worst_30d_notional / 2.0
    
    # Bootstrap 95% CI on daily net return
    daily_net = sub.groupby("date")["net_cf"].sum().reset_index()
    daily_net["ym"] = daily_net["date"].dt.to_period("M").astype(str)
    months = daily_net["ym"].unique()
    n_m = len(months)
    month_dict = {m: daily_net[daily_net["ym"] == m]["net_cf"].values for m in months}
    
    np.random.seed(42)
    n_boot = 10000
    boot_ann_notional = np.empty(n_boot)
    for b in range(n_boot):
        sampled_m = np.random.choice(months, size=n_m, replace=True)
        pooled = np.concatenate([month_dict[m] for m in sampled_m])
        boot_ann_notional[b] = (np.sum(pooled) / (len(pooled) / 365.25)) * 100.0
        
    ci_l_notional = float(np.percentile(boot_ann_notional, 2.5))
    ci_u_notional = float(np.percentile(boot_ann_notional, 97.5))
    
    return {
        "days": days,
        "years": years,
        "entries": entries,
        "exits": exits,
        "tot_net_notional_pct": tot_net_notional * 100.0,
        "ann_return_notional": ann_return_notional,
        "ann_return_2x_capital": ann_return_2x_capital,
        "ci_notional": (ci_l_notional, ci_u_notional),
        "ci_2x": (ci_l_notional / 2.0, ci_u_notional / 2.0),
        "max_dd_notional_pct": max_dd_notional * 100.0,
        "max_dd_2x_pct": max_dd_2x_capital,
        "pct_fr_pos_total": pct_fr_pos_total,
        "pct_fr_pos_invested": pct_fr_pos_invested,
        "worst_30d_notional_pct": worst_30d_notional,
        "worst_30d_2x_pct": worst_30d_2x_capital,
        "neg_funding_days": neg_funding_days,
        "total_days": total_days
    }

# ==============================================================================
# STRATEGY B: DAILY TREND FOLLOWING
# ==============================================================================
def simulate_trend(df_1d, config):
    df = df_1d.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    
    n = len(df)
    target_pos = np.zeros(n)
    
    if config == "B1":
        # 100-day SMA
        sma100 = c.rolling(100).mean()
        # Signal at close t: 1 if close > sma100 else 0
        sig = (c > sma100).astype(float)
        # Shift 1: execute at open t+1
        target_pos = sig.shift(1).fillna(0.0).values
        
    elif config == "B2":
        # 50-day Donchian High / 20-day Low
        # Breakout of previous 50 bars (excluding today)
        donch_high = h.shift(1).rolling(50).max()
        donch_low = l.shift(1).rolling(20).min()
        
        pos_state = 0.0
        states = []
        for i in range(n):
            close_i = c.iloc[i]
            dh = donch_high.iloc[i]
            dl = donch_low.iloc[i]
            
            if not np.isnan(dh) and close_i > dh:
                pos_state = 1.0
            elif not np.isnan(dl) and close_i < dl:
                pos_state = 0.0
            states.append(pos_state)
            
        states = pd.Series(states, index=df.index)
        target_pos = states.shift(1).fillna(0.0).values
        
    elif config == "B3":
        # 50 SMA > 200 SMA
        sma50 = c.rolling(50).mean()
        sma200 = c.rolling(200).mean()
        sig = (sma50 > sma200).astype(float)
        target_pos = sig.shift(1).fillna(0.0).values
        
    elif config == "B4":
        # B2 with Volatility Scaling (target 20% annualized vol)
        donch_high = h.shift(1).rolling(50).max()
        donch_low = l.shift(1).rolling(20).min()
        
        # 30-day realized annualized volatility
        ret_daily = c.pct_change(1)
        vol30 = ret_daily.shift(1).rolling(30).std() * np.sqrt(365)
        
        pos_state = 0.0
        weights = []
        for i in range(n):
            close_i = c.iloc[i]
            dh = donch_high.iloc[i]
            dl = donch_low.iloc[i]
            v = vol30.iloc[i]
            
            if not np.isnan(dh) and close_i > dh:
                pos_state = 1.0
            elif not np.isnan(dl) and close_i < dl:
                pos_state = 0.0
                
            if pos_state == 1.0 and not np.isnan(v) and v > 0:
                w = min(1.0, 0.20 / max(v, 0.05))
            else:
                w = 0.0
            weights.append(w)
            
        weights = pd.Series(weights, index=df.index)
        target_pos = weights.shift(1).fillna(0.0).values
        
    else:
        raise ValueError(f"Unknown trend config: {config}")
        
    df["target_weight"] = target_pos
    return df

def evaluate_trend_split(df_sim, start_time, end_time):
    mask = (df_sim["open_time"] >= start_time) & (df_sim["open_time"] <= end_time)
    sub = df_sim[mask].copy().reset_index(drop=True)
    
    if len(sub) == 0:
        return {}
        
    n = len(sub)
    weights = sub["target_weight"].values
    opens = sub["open"].values
    closes = sub["close"].values
    daily_frs = sub["daily_funding_rate"].values
    times = sub["open_time"].values
    
    # Track trade metrics
    trades = []
    in_trade = False
    trade_start_idx = 0
    trade_ret_factors = []
    
    daily_net_ret = np.zeros(n)
    daily_gross_ret = np.zeros(n)
    daily_fee = np.zeros(n)
    daily_funding = np.zeros(n)
    
    prev_w = 0.0
    
    for i in range(n):
        w = weights[i]
        
        # Transaction cost on weight turnover
        dw = abs(w - prev_w)
        fee = dw * TREND_TAKER_FEE_SIDE
        daily_fee[i] = fee
        
        # Price return from Open to Close on day i
        # Since crypto trades 24/7, Open_i to Close_i captures the day's move
        price_ret = (closes[i] / opens[i]) - 1.0
        gross = w * price_ret
        daily_gross_ret[i] = gross
        
        # Funding paid while holding long perpetual futures
        funding_cost = w * daily_frs[i]
        daily_funding[i] = funding_cost
        
        # Net daily return
        net = gross - funding_cost - fee
        daily_net_ret[i] = net
        
        # Trade log tracking
        if not in_trade and w > 0:
            in_trade = True
            trade_start_idx = i
            trade_ret_factors = [1.0 + net]
        elif in_trade:
            trade_ret_factors.append(1.0 + net)
            if w == 0:
                in_trade = False
                trade_cum_ret = np.prod(trade_ret_factors) - 1.0
                trades.append({
                    "start": times[trade_start_idx],
                    "end": times[i],
                    "duration": i - trade_start_idx + 1,
                    "net_ret": trade_cum_ret
                })
                
        prev_w = w
        
    # If still in trade at the end of the window, close it out
    if in_trade:
        trade_cum_ret = np.prod(trade_ret_factors) - 1.0
        trades.append({
            "start": times[trade_start_idx],
            "end": times[-1],
            "duration": n - trade_start_idx,
            "net_ret": trade_cum_ret
        })
        
    sub["daily_net_ret"] = daily_net_ret
    
    # Portfolio equity curve
    equity = np.cumprod(1.0 + daily_net_ret)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd) * 100.0) if len(dd) > 0 else 0.0
    
    days = (pd.to_datetime(times[-1]) - pd.to_datetime(times[0])).total_seconds() / 86400.0
    years = max(days / 365.25, 0.001)
    
    tot_ret = float((equity[-1] - 1.0) * 100.0)
    cagr = float(((equity[-1]) ** (1.0 / years) - 1.0) * 100.0) if equity[-1] > 0 else -100.0
    ann_vol = float(np.std(daily_net_ret, ddof=1) * np.sqrt(365) * 100.0)
    sharpe = float((cagr / ann_vol)) if ann_vol > 0 else 0.0
    
    # Trade statistics
    n_trades = len(trades)
    trades_per_year = float(n_trades / years)
    if n_trades > 0:
        t_rets = np.array([t["net_ret"] for t in trades])
        wins = t_rets[t_rets > 0]
        losses = t_rets[t_rets < 0]
        true_wr = float(len(wins) / n_trades * 100.0)
        avg_win = float(np.mean(wins) * 100.0) if len(wins) > 0 else 0.0
        avg_loss = float(np.mean(losses) * 100.0) if len(losses) > 0 else 0.0
        win_loss_ratio = float(abs(avg_win / avg_loss)) if avg_loss != 0 else 9.99
        sum_gains = float(np.sum(wins)) if len(wins) > 0 else 0.0
        sum_losses = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
        pf = float(sum_gains / sum_losses) if sum_losses > 0 else 9.99
    else:
        true_wr, avg_win, avg_loss, win_loss_ratio, pf = 0.0, 0.0, 0.0, 0.0, 0.0
        
    time_in_market = float((weights > 0).mean() * 100.0)
    
    # Buy and Hold Benchmark
    bh_ret = closes / closes[0]
    bh_peak = np.maximum.accumulate(bh_ret)
    bh_dd = (bh_ret - bh_peak) / bh_peak
    bh_max_dd = float(np.min(bh_dd) * 100.0)
    bh_cagr = float(((bh_ret[-1]) ** (1.0 / years) - 1.0) * 100.0) if bh_ret[-1] > 0 else -100.0
    bh_daily = np.diff(np.log(closes))
    bh_vol = float(np.std(bh_daily, ddof=1) * np.sqrt(365) * 100.0)
    bh_sharpe = float(bh_cagr / bh_vol) if bh_vol > 0 else 0.0
    
    # Monthly Block Bootstrap on daily net return
    sub["date"] = pd.to_datetime(sub["open_time"])
    sub["ym"] = sub["date"].dt.to_period("M").astype(str)
    months = sub["ym"].unique()
    n_m = len(months)
    month_dict = {m: sub[sub["ym"] == m]["daily_net_ret"].values for m in months}
    
    np.random.seed(42)
    n_boot = 10000
    boot_cagr = np.empty(n_boot)
    for b in range(n_boot):
        sampled_m = np.random.choice(months, size=n_m, replace=True)
        pooled = np.concatenate([month_dict[m] for m in sampled_m])
        # Compound annual growth
        cum_b = np.prod(1.0 + pooled)
        y_b = len(pooled) / 365.25
        boot_cagr[b] = ((cum_b) ** (1.0 / y_b) - 1.0) * 100.0 if cum_b > 0 else -100.0
        
    ci_l = float(np.percentile(boot_cagr, 2.5))
    ci_u = float(np.percentile(boot_cagr, 97.5))
    
    return {
        "days": days,
        "years": years,
        "cagr": cagr,
        "ci_cagr": (ci_l, ci_u),
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "trades": n_trades,
        "trades_per_year": trades_per_year,
        "true_wr": true_wr,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": win_loss_ratio,
        "pf": pf,
        "time_in_market": time_in_market,
        "bh_cagr": bh_cagr,
        "bh_vol": bh_vol,
        "bh_sharpe": bh_sharpe,
        "bh_max_dd": bh_max_dd
    }

# ==============================================================================
# MAIN RUNNER (DEV & VALIDATION ONLY)
# ==============================================================================
def main():
    print("=" * 110)
    print("STRUCTURAL EDGE STUDY: DEV & VALIDATION EVALUATION (STRICT NO-FINAL RULE)")
    print("Pre-Registration Git Commit Hash: 287dd23a0a21ef6c5e1f528e88f42ce30c4c1837")
    print("=" * 110)
    
    df_fr, df_1d = load_data()
    
    # ---------------------------------------------------------
    # STRATEGY A: FUNDING-RATE CARRY
    # ---------------------------------------------------------
    print("\n" + "=" * 110)
    print("STRATEGY A: FUNDING-RATE CARRY (3 CONFIGS)")
    print("=" * 110)
    
    splits_carry = {
        "DEV (2019-09 to 2022-12)": (df_fr["fundingTime"].min(), DEV_END),
        "VALIDATION (2023-01 to 2024-12)": (VAL_START, VAL_END)
    }
    
    carry_results = {}
    for cfg in ["A1", "A2", "A3"]:
        df_sim = simulate_carry(df_fr, cfg)
        carry_results[cfg] = {}
        for s_name, (st, en) in splits_carry.items():
            res = evaluate_carry_split(df_sim, st, en)
            carry_results[cfg][s_name] = res
            
    # Print Strategy A DEV Table
    for s_name in splits_carry.keys():
        print(f"\n--- {s_name} ---")
        print(f"{'Config':<6} | {'Ann Net Yield (Notional)':<24} | {'Ann Return (2x Capital)':<24} | {'Max DD (2x Cap)':<16} | {'Trades':<8} | {'% FR Pos (Invested)':<20} | {'Worst 30d (2x Cap)':<20} | {'Neg FR Days':<12}")
        print("-" * 140)
        for cfg in ["A1", "A2", "A3"]:
            r = carry_results[cfg][s_name]
            ci_str_2x = f"[{r['ci_2x'][0]:+.1f}%, {r['ci_2x'][1]:+.1f}%]"
            ann_2x_str = f"{r['ann_return_2x_capital']:+.2f}% {ci_str_2x}"
            print(f"{cfg:<6} | {r['ann_return_notional']:>+8.2f}% [{'%.1f' % r['ci_notional'][0]}%, {'%.1f' % r['ci_notional'][1]}%] | {ann_2x_str:<24} | {r['max_dd_2x_pct']:>8.2f}%       | {r['entries'] + r['exits']:<8} | {r['pct_fr_pos_invested']:>6.1f}%              | {r['worst_30d_2x_pct']:>8.2f}%           | {r['neg_funding_days']}/{r['total_days']}")
            
    # ---------------------------------------------------------
    # STRATEGY B: DAILY TREND FOLLOWING
    # ---------------------------------------------------------
    print("\n" + "=" * 110)
    print("STRATEGY B: DAILY TREND FOLLOWING (4 CONFIGS)")
    print("=" * 110)
    
    splits_trend = {
        "DEV (2020-01 to 2022-12)": (df_1d["open_time"].min(), DEV_END),
        "VALIDATION (2023-01 to 2024-12)": (VAL_START, VAL_END)
    }
    
    trend_results = {}
    for cfg in ["B1", "B2", "B3", "B4"]:
        df_sim = simulate_trend(df_1d, cfg)
        trend_results[cfg] = {}
        for s_name, (st, en) in splits_trend.items():
            res = evaluate_trend_split(df_sim, st, en)
            trend_results[cfg][s_name] = res
            
    # Print Strategy B Tables
    for s_name in splits_trend.keys():
        print(f"\n--- {s_name} ---")
        print(f"{'Config':<6} | {'CAGR':<10} | {'95% Monthly CI':<18} | {'Ann Vol':<9} | {'Sharpe':<7} | {'Max DD':<9} | {'Trades/Yr':<10} | {'True WR':<8} | {'Avg W/L':<8} | {'PF':<6} | {'TimeInMkt':<10}")
        print("-" * 125)
        for cfg in ["B1", "B2", "B3", "B4"]:
            r = trend_results[cfg][s_name]
            ci_str = f"[{r['ci_cagr'][0]:+.1f}%, {r['ci_cagr'][1]:+.1f}%]"
            print(f"{cfg:<6} | {r['cagr']:>+6.1f}%   | {ci_str:<18} | {r['ann_vol']:>6.1f}%   | {r['sharpe']:>5.2f}  | {r['max_dd']:>6.1f}%   | {r['trades_per_year']:>6.1f}     | {r['true_wr']:>5.1f}%  | {r['win_loss_ratio']:>5.2f}    | {r['pf']:<6.2f} | {r['time_in_market']:>6.1f}%")
        # Buy & hold benchmark
        r_bh = trend_results["B1"][s_name]
        print(f"{'BTC B&H':<6} | {r_bh['bh_cagr']:>+6.1f}%   | {'N/A':<18} | {r_bh['bh_vol']:>6.1f}%   | {r_bh['bh_sharpe']:>5.2f}  | {r_bh['bh_max_dd']:>6.1f}%   | {'0.0':<10} | {'N/A':<8} | {'N/A':<8} | {'N/A':<6} | {'100.0%':<10}")

if __name__ == "__main__":
    main()
