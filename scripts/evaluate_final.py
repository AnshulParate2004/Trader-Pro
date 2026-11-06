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

CARRY_ENTRY_FEE = (SPOT_TAKER_FEE + SPOT_SLIPPAGE) + (PERP_TAKER_FEE + PERP_SLIPPAGE) + BASIS_SLIPPAGE_HALF # 0.001922
CARRY_EXIT_FEE = CARRY_ENTRY_FEE # 0.001922

# Strategy B: Trend
TREND_TAKER_FEE_SIDE = PERP_TAKER_FEE + PERP_SLIPPAGE # 0.000572 (0.0572% per side)

# FINAL WINDOW
FINAL_START = pd.to_datetime("2025-01-01 00:00:00", utc=True)
FINAL_END = pd.to_datetime("2026-10-31 23:59:59", utc=True)

def load_data():
    base_dir = r"D:\Projects_Main\Trade-Pro\Trader-Pro\notebooks\data"
    fr_path = os.path.join(base_dir, "derivatives", "funding_rates_BTCUSDT.parquet")
    d1_path = os.path.join(base_dir, "1d", "BTCUSDT_1d.parquet")
    d4_path = os.path.join(base_dir, "4h", "BTCUSDT_4h.parquet")
    
    df_fr = pd.read_parquet(fr_path)
    df_fr["fundingTime"] = pd.to_datetime(df_fr["fundingTime"], utc=True)
    df_fr = df_fr.sort_values("fundingTime").reset_index(drop=True)
    
    df_1d = pd.read_parquet(d1_path)
    df_1d["open_time"] = pd.to_datetime(df_1d["open_time"], utc=True)
    df_1d["close_time"] = pd.to_datetime(df_1d["close_time"], utc=True)
    df_1d = df_1d.sort_values("open_time").reset_index(drop=True)
    
    # Merge daily funding into 1D
    df_fr["date"] = df_fr["fundingTime"].dt.floor("D")
    daily_fr = df_fr.groupby("date")["fundingRate"].sum().reset_index()
    daily_fr.rename(columns={"fundingRate": "daily_funding_rate"}, inplace=True)
    df_1d = pd.merge(df_1d, daily_fr, left_on="open_time", right_on="date", how="left")
    df_1d["daily_funding_rate"] = df_1d["daily_funding_rate"].fillna(0.0)
    
    # Load 4h for spot mark price basis calculation
    df_4h = pd.read_parquet(d4_path)
    df_4h["open_time"] = pd.to_datetime(df_4h["open_time"], utc=True)
    
    return df_fr, df_1d, df_4h

# ==============================================================================
# STRATEGY A3 EVALUATION ON FINAL
# ==============================================================================
def evaluate_strategy_a3(df_fr, df_4h):
    df = df_fr.copy()
    # 30-day trailing mean (90 periods)
    df["fr_30d"] = df["fundingRate"].shift(1).rolling(90).mean()
    df["target_pos"] = (df["fr_30d"] > 0.0).astype(int)
    
    # Filter to FINAL window
    mask = (df["fundingTime"] >= FINAL_START) & (df["fundingTime"] <= FINAL_END)
    sub = df[mask].copy().reset_index(drop=True)
    
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
        if target == 1 and current_pos == 0:
            f += CARRY_ENTRY_FEE
            entries += 1
            current_pos = 1
        elif target == 0 and current_pos == 1:
            f += CARRY_EXIT_FEE
            exits += 1
            current_pos = 0
            
        friction[i] = f
        gross = current_pos * fr[i]
        gross_cf[i] = gross
        net_cf[i] = gross - f
        
    if current_pos == 1:
        net_cf[-1] -= CARRY_EXIT_FEE
        friction[-1] += CARRY_EXIT_FEE
        exits += 1
        
    sub["net_cf"] = net_cf
    sub["gross_cf"] = gross_cf
    sub["friction"] = friction
    
    days = (pd.to_datetime(times[-1]) - pd.to_datetime(times[0])).total_seconds() / 86400.0
    years = max(days / 365.25, 0.001)
    
    tot_net_notional = float(np.sum(net_cf))
    ann_return_notional = (tot_net_notional / years) * 100.0
    ann_return_2x_capital = (tot_net_notional / 2.0 / years) * 100.0
    
    # Cumulative notional equity
    cum_net = np.cumsum(net_cf)
    peak = np.maximum.accumulate(cum_net)
    dd_notional = cum_net - peak
    max_dd_notional = float(np.min(dd_notional) * 100.0) if len(dd_notional) > 0 else 0.0
    max_dd_2x = max_dd_notional / 2.0
    
    # Positive funding stats
    pct_fr_pos_total = float((fr > 0).mean() * 100.0)
    invested_mask = pos == 1
    pct_fr_pos_invested = float((fr[invested_mask] > 0).mean() * 100.0) if invested_mask.sum() > 0 else 0.0
    
    # Negative funding days
    sub["date"] = pd.to_datetime(sub["fundingTime"]).dt.floor("D")
    daily_fr = sub.groupby("date")["fundingRate"].sum()
    neg_funding_days = int((daily_fr < 0).sum())
    total_days = len(daily_fr)
    
    # Worst 30-day period (rolling 90 periods)
    sub["rolling_30d"] = sub["net_cf"].rolling(90).sum()
    worst_30d_notional = float(sub["rolling_30d"].min() * 100.0) if len(sub) >= 90 else float(tot_net_notional * 100.0)
    worst_30d_2x = worst_30d_notional / 2.0
    
    # Monthly Block Bootstrap 95% CI
    daily_net = sub.groupby("date")["net_cf"].sum().reset_index()
    daily_net["ym"] = daily_net["date"].dt.strftime("%Y-%m")
    months = daily_net["ym"].unique()
    n_m = len(months)
    month_dict = {m: daily_net[daily_net["ym"] == m]["net_cf"].values for m in months}
    
    np.random.seed(42)
    n_boot = 10000
    boot_ann_notional = np.empty(n_boot)
    for b in range(n_boot):
        s_m = np.random.choice(months, size=n_m, replace=True)
        pooled = np.concatenate([month_dict[m] for m in s_m])
        boot_ann_notional[b] = (np.sum(pooled) / (len(pooled) / 365.25)) * 100.0
        
    ci_l_notional = float(np.percentile(boot_ann_notional, 2.5))
    ci_u_notional = float(np.percentile(boot_ann_notional, 97.5))
    ci_l_2x = ci_l_notional / 2.0
    ci_u_2x = ci_u_notional / 2.0
    
    # Calendar Quarter Breakdown
    sub["quarter"] = pd.to_datetime(sub["fundingTime"]).dt.to_period("Q").astype(str)
    q_stats = []
    for q, grp in sub.groupby("quarter"):
        q_days = len(grp) / 3.0
        q_net_notional = float(grp["net_cf"].sum() * 100.0)
        q_ann_notional = q_net_notional / (q_days / 365.25) if q_days > 0 else 0.0
        q_net_2x = q_net_notional / 2.0
        q_ann_2x = q_ann_notional / 2.0
        pct_pos = float((grp["fundingRate"] > 0).mean() * 100.0)
        q_stats.append({
            "quarter": q,
            "days": q_days,
            "net_notional_pct": q_net_notional,
            "ann_notional_pct": q_ann_notional,
            "net_2x_pct": q_net_2x,
            "ann_2x_pct": q_ann_2x,
            "pct_fr_pos": pct_pos
        })
    df_quarterly = pd.DataFrame(q_stats)
    
    # Basis Move Analysis
    m_basis = pd.merge(sub[["fundingTime", "markPrice"]], df_4h[["open_time", "open"]], left_on="fundingTime", right_on="open_time", how="inner")
    if len(m_basis) > 0 and m_basis["markPrice"].notna().sum() > 0:
        m_basis["basis_pct"] = (m_basis["markPrice"] - m_basis["open"]) / m_basis["open"] * 100.0
        max_basis_contango = float(m_basis["basis_pct"].max())
        max_basis_backwardation = float(m_basis["basis_pct"].min())
        max_abs_basis_move = float(m_basis["basis_pct"].abs().max())
        mean_basis = float(m_basis["basis_pct"].mean())
    else:
        max_basis_contango, max_basis_backwardation, max_abs_basis_move, mean_basis = np.nan, np.nan, np.nan, np.nan
        
    return {
        "days": days,
        "years": years,
        "entries": entries,
        "exits": exits,
        "tot_net_notional_pct": tot_net_notional * 100.0,
        "ann_return_notional": ann_return_notional,
        "ci_notional": (ci_l_notional, ci_u_notional),
        "ann_return_2x": ann_return_2x_capital,
        "ci_2x": (ci_l_2x, ci_u_2x),
        "max_dd_notional": max_dd_notional,
        "max_dd_2x": max_dd_2x,
        "pct_fr_pos_total": pct_fr_pos_total,
        "pct_fr_pos_invested": pct_fr_pos_invested,
        "neg_funding_days": neg_funding_days,
        "total_days": total_days,
        "worst_30d_notional": worst_30d_notional,
        "worst_30d_2x": worst_30d_2x,
        "quarterly": df_quarterly,
        "basis_stats": {
            "mean": mean_basis,
            "max_contango": max_basis_contango,
            "max_backwardation": max_basis_backwardation,
            "max_abs": max_abs_basis_move
        }
    }

# ==============================================================================
# STRATEGY B EVALUATION ON FINAL (B1, B1-SMA80, B1-SMA120)
# ==============================================================================
def simulate_trend_sma(df_1d, sma_len):
    df = df_1d.copy()
    c = df["close"]
    sma = c.rolling(sma_len).mean()
    sig = (c > sma).astype(float)
    # Execute at next day's open
    df["target_weight"] = sig.shift(1).fillna(0.0).values
    return df

def evaluate_trend_final(df_sim):
    mask = (df_sim["open_time"] >= FINAL_START) & (df_sim["open_time"] <= FINAL_END)
    sub = df_sim[mask].copy().reset_index(drop=True)
    
    n = len(sub)
    weights = sub["target_weight"].values
    opens = sub["open"].values
    closes = sub["close"].values
    daily_frs = sub["daily_funding_rate"].values
    times = sub["open_time"].values
    
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
        dw = abs(w - prev_w)
        fee = dw * TREND_TAKER_FEE_SIDE
        daily_fee[i] = fee
        
        price_ret = (closes[i] / opens[i]) - 1.0
        gross = w * price_ret
        daily_gross_ret[i] = gross
        
        funding_cost = w * daily_frs[i]
        daily_funding[i] = funding_cost
        
        net = gross - funding_cost - fee
        daily_net_ret[i] = net
        
        if not in_trade and w > 0:
            in_trade = True
            trade_start_idx = i
            trade_ret_factors = [1.0 + net]
        elif in_trade:
            trade_ret_factors.append(1.0 + net)
            if w == 0:
                in_trade = False
                trades.append({
                    "start": times[trade_start_idx],
                    "end": times[i],
                    "duration": i - trade_start_idx + 1,
                    "net_ret": np.prod(trade_ret_factors) - 1.0
                })
        prev_w = w
        
    if in_trade:
        trades.append({
            "start": times[trade_start_idx],
            "end": times[-1],
            "duration": n - trade_start_idx,
            "net_ret": np.prod(trade_ret_factors) - 1.0
        })
        
    sub["daily_net_ret"] = daily_net_ret
    equity = np.cumprod(1.0 + daily_net_ret)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd) * 100.0) if len(dd) > 0 else 0.0
    
    days = (pd.to_datetime(times[-1]) - pd.to_datetime(times[0])).total_seconds() / 86400.0
    years = max(days / 365.25, 0.001)
    
    cagr = float(((equity[-1]) ** (1.0 / years) - 1.0) * 100.0) if equity[-1] > 0 else -100.0
    ann_vol = float(np.std(daily_net_ret, ddof=1) * np.sqrt(365) * 100.0)
    sharpe = float(cagr / ann_vol) if ann_vol > 0 else 0.0
    
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
    
    # Monthly Block Bootstrap on CAGR
    sub["date"] = pd.to_datetime(sub["open_time"])
    sub["ym"] = sub["date"].dt.strftime("%Y-%m")
    months = sub["ym"].unique()
    n_m = len(months)
    month_dict = {m: sub[sub["ym"] == m]["daily_net_ret"].values for m in months}
    
    np.random.seed(42)
    n_boot = 10000
    boot_cagr = np.empty(n_boot)
    for b in range(n_boot):
        s_m = np.random.choice(months, size=n_m, replace=True)
        pooled = np.concatenate([month_dict[m] for m in s_m])
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
        "trades_list": trades,
        "bh_cagr": bh_cagr,
        "bh_vol": bh_vol,
        "bh_sharpe": bh_sharpe,
        "bh_max_dd": bh_max_dd
    }

def main():
    print("=" * 110)
    print("FINAL EVALUATION: 2025-01-01 to 2026-10-31 (STRICTLY ONCE)")
    print("Protocol Commit Hash: 9c882a846fac7b5cc778ff6dcef5c202d708c2c2")
    print("=" * 110)
    
    df_fr, df_1d, df_4h = load_data()
    
    # ---------------------------------------------------------
    # STRATEGY A3 EVALUATION
    # ---------------------------------------------------------
    res_a3 = evaluate_strategy_a3(df_fr, df_4h)
    print("\n" + "=" * 110)
    print("1. STRATEGY A3 (FUNDING-RATE CARRY, 30D FILTER) ON FINAL")
    print("=" * 110)
    print(f"Window: 2025-01-01 to 2026-10-31 ({res_a3['days']:.1f} days / {res_a3['years']:.2f} years)")
    print(f"Annualized Yield on 1x Notional:    {res_a3['ann_return_notional']:>+7.2f}%  (95% Monthly CI: [{res_a3['ci_notional'][0]:+.1f}%, {res_a3['ci_notional'][1]:+.1f}%])")
    print(f"Annualized Return on 2x Capital:    {res_a3['ann_return_2x']:>+7.2f}%  (95% Monthly CI: [{res_a3['ci_2x'][0]:+.1f}%, {res_a3['ci_2x'][1]:+.1f}%])")
    print(f"5% Risk-Free Reference Hurdle:       +5.00%")
    print(f"Max Drawdown (2x Capital):           {res_a3['max_dd_2x']:>7.2f}%  (Notional Max DD: {res_a3['max_dd_notional']:.2f}%)")
    print(f"Worst 30-Day Rolling Net Return:     {res_a3['worst_30d_2x']:>7.2f}%  (Notional: {res_a3['worst_30d_notional']:.2f}%)")
    print(f"Total Completed Turns (Entries/Exits): {res_a3['entries'] + res_a3['exits']} ({res_a3['entries']} entries, {res_a3['exits']} exits)")
    print(f"% Funding Periods Positive:          {res_a3['pct_fr_pos_total']:.1f}% (Total) | {res_a3['pct_fr_pos_invested']:.1f}% (While Invested)")
    print(f"Negative Funding Days:               {res_a3['neg_funding_days']} / {res_a3['total_days']} days ({(res_a3['neg_funding_days']/res_a3['total_days']*100):.1f}%)")
    
    print("\nStrategy A3 Calendar Quarter Compression Breakdown:")
    print(f"{'Quarter':<10} | {'Days':<6} | {'Quarter Net Notional':<22} | {'Quarter Net 2x Cap':<20} | {'Ann Yield Notional':<20} | {'Ann Return 2x Cap':<20} | {'% FR Pos':<10}")
    print("-" * 125)
    for idx, row in res_a3["quarterly"].iterrows():
        print(f"{row['quarter']:<10} | {row['days']:>4.0f}   | {row['net_notional_pct']:>+19.2f}%  | {row['net_2x_pct']:>+17.2f}%  | {row['ann_notional_pct']:>+17.2f}%  | {row['ann_2x_pct']:>+17.2f}%  | {row['pct_fr_pos']:>7.1f}%")
        
    print("\nPerp-Spot Basis Metrics in FINAL Window:")
    bs = res_a3["basis_stats"]
    print(f"  Mean Basis:                {bs['mean']:+.4f}%")
    print(f"  Max Contango Basis:        {bs['max_contango']:+.4f}%")
    print(f"  Max Backwardation Basis:   {bs['max_backwardation']:+.4f}%")
    print(f"  Max Absolute Basis Move:   {bs['max_abs']:+.4f}%")
    
    # ---------------------------------------------------------
    # STRATEGY B1 EVALUATION ON FINAL
    # ---------------------------------------------------------
    res_b1 = evaluate_trend_final(simulate_trend_sma(df_1d, 100))
    res_b80 = evaluate_trend_final(simulate_trend_sma(df_1d, 80))
    res_b120 = evaluate_trend_final(simulate_trend_sma(df_1d, 120))
    
    print("\n" + "=" * 110)
    print("2. STRATEGY B1 (DAILY TREND FOLLOWING, 100-DAY SMA) ON FINAL")
    print("=" * 110)
    print(f"{'Metric':<30} | {'Strategy B1 (100 SMA)':<24} | {'BTC Buy & Hold':<20}")
    print("-" * 80)
    print(f"{'CAGR':<30} | {res_b1['cagr']:>+21.2f}%  | {res_b1['bh_cagr']:>+17.2f}%")
    print(f"{'95% Monthly Block CI':<30} | [{res_b1['ci_cagr'][0]:+.1f}%, {res_b1['ci_cagr'][1]:+.1f}%]              | {'N/A':<20}")
    print(f"{'Annualized Volatility':<30} | {res_b1['ann_vol']:>21.2f}%  | {res_b1['bh_vol']:>17.2f}%")
    print(f"{'Sharpe Ratio':<30} | {res_b1['sharpe']:>21.2f}   | {res_b1['bh_sharpe']:>17.2f}")
    print(f"{'Max Drawdown':<30} | {res_b1['max_dd']:>21.2f}%  | {res_b1['bh_max_dd']:>17.2f}%")
    print(f"{'Completed Trades in Window':<30} | {res_b1['trades']:>21d}   | {'0':>17}")
    print(f"{'Trades Per Year':<30} | {res_b1['trades_per_year']:>21.1f}   | {'0.0':>17}")
    print(f"{'True Win Rate':<30} | {res_b1['true_wr']:>21.1f}%  | {'N/A':>17}")
    print(f"{'Avg Win / Avg Loss Ratio':<30} | {res_b1['win_loss_ratio']:>21.2f}   | {'N/A':>17}")
    print(f"{'Profit Factor':<30} | {res_b1['pf']:>21.2f}   | {'N/A':>17}")
    print(f"{'Time in Market':<30} | {res_b1['time_in_market']:>21.1f}%  | {'100.0%':>17}")
    
    print("\nStrategy B1 Trade Log on FINAL:")
    for idx, t in enumerate(res_b1["trades_list"]):
        st_str = pd.to_datetime(t['start']).strftime('%Y-%m-%d')
        en_str = pd.to_datetime(t['end']).strftime('%Y-%m-%d')
        print(f"  Trade {idx+1}: {st_str} -> {en_str} ({t['duration']} days) | Net Return: {t['net_ret']*100:>+7.2f}%")

    # ---------------------------------------------------------
    # ROBUSTNESS ANALYSIS (SMA 80 and SMA 120)
    # ---------------------------------------------------------
    print("\n" + "=" * 110)
    print("3. PARAMETER STABILITY / ROBUSTNESS CHECK (NOT FOR SELECTION)")
    print("=" * 110)
    print(f"{'Variant':<16} | {'CAGR':<10} | {'95% Monthly CI':<20} | {'Ann Vol':<9} | {'Sharpe':<7} | {'Max DD':<9} | {'Trades':<8} | {'True WR':<8} | {'PF':<6}")
    print("-" * 105)
    for name, r in [("B1-SMA80", res_b80), ("B1 (100 SMA)", res_b1), ("B1-SMA120", res_b120)]:
        ci_str = f"[{r['ci_cagr'][0]:+.1f}%, {r['ci_cagr'][1]:+.1f}%]"
        print(f"{name:<16} | {r['cagr']:>+7.2f}% | {ci_str:<20} | {r['ann_vol']:>6.1f}%   | {r['sharpe']:>5.2f}  | {r['max_dd']:>6.1f}%   | {r['trades']:<8} | {r['true_wr']:>5.1f}%  | {r['pf']:<6.2f}")

if __name__ == "__main__":
    main()
