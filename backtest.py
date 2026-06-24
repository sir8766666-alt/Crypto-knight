"""
Crypto Knight — Backtester
Runs our exact strategy on 60 days of historical M5 data
Shows real win rate, best times, best assets — BEFORE risking real money
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

IST = timezone(timedelta(hours=5, minutes=30))

ASSETS = {
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/JPY": "AUDJPY=X",
    "EUR/JPY": "EURJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
}

SEP = "=" * 58

# ── Indicators ────────────────────────────────────────────────────────────────
def calc_adx(df, p=14):
    h, l, c = df["high"], df["low"], df["close"]
    pdm = h.diff().clip(lower=0)
    mdm = (-l.diff()).clip(lower=0)
    pdm[pdm <= mdm] = 0
    mdm[mdm <= pdm] = 0
    tr  = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr = tr.ewm(span=p,adjust=False).mean()
    pdi = 100*pdm.ewm(span=p,adjust=False).mean()/atr
    mdi = 100*mdm.ewm(span=p,adjust=False).mean()/atr
    dx  = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(span=p,adjust=False).mean(), pdi, mdi

def calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(span=p,adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=p,adjust=False).mean()
    return 100-(100/(1+g/l.replace(0,np.nan)))

def calc_macd(s):
    m = s.ewm(span=12,adjust=False).mean() - s.ewm(span=26,adjust=False).mean()
    sig = m.ewm(span=9,adjust=False).mean()
    return m, sig, m - sig

def calc_bb(s, p=20):
    mid = s.rolling(p).mean()
    std = s.rolling(p).std()
    return mid+2*std, mid, mid-2*std

def calc_stoch(df, k=14, d=3):
    lo = df["low"].rolling(k).min()
    hi = df["high"].rolling(k).max()
    kl = 100*(df["close"]-lo)/(hi-lo).replace(0,np.nan)
    return kl, kl.rolling(d).mean()

# ── Strategy signal — same as live scanner ────────────────────────────────────
def get_signal(df, i):
    """
    Given dataframe and index i, return UP/DOWN/None
    using same 6/7 indicator logic as live scanner
    """
    if i < 60: return None

    sl = df.iloc[i-60:i+1].copy()
    sl = sl.reset_index(drop=True)

    sl["sma50"]              = sl["close"].rolling(50).mean()
    sl["ema9"]               = sl["close"].ewm(span=9,  adjust=False).mean()
    sl["ema21"]              = sl["close"].ewm(span=21, adjust=False).mean()
    sl["adx"],sl["pdi"],sl["mdi"] = calc_adx(sl)
    sl["rsi"]                = calc_rsi(sl["close"])
    sl["macd"],sl["ms"],sl["mh"]  = calc_macd(sl["close"])
    sl["bb_up"],_,sl["bb_lo"]     = calc_bb(sl["close"])
    sl["sk"],sl["sd"]        = calc_stoch(sl)

    r  = sl.iloc[-1]
    p2 = sl.iloc[-2]

    price    = float(r["close"])
    sma50    = float(r["sma50"])
    adx_val  = float(r["adx"])
    pdi      = float(r["pdi"])
    mdi      = float(r["mdi"])
    rsi_val  = float(r["rsi"])
    ema9     = float(r["ema9"])
    ema21    = float(r["ema21"])
    macd_now = float(r["macd"])
    sig_now  = float(r["ms"])
    hist_now = float(r["mh"])
    hist_prv = float(p2["mh"])
    bb_up    = float(r["bb_up"])
    bb_lo    = float(r["bb_lo"])
    sk       = float(r["sk"])
    sd       = float(r["sd"])
    slope    = sma50 - float(p2["sma50"])

    recent    = sl.tail(10)
    range_pct = (float(recent["high"].max()-recent["low"].min())/price)*100
    if adx_val < 20 or range_pct < 0.04: return None

    bull = bear = 0

    if price > sma50 and slope > 0:           bull += 1
    elif price < sma50 and slope < 0:         bear += 1

    if ema9 > ema21:                           bull += 1
    elif ema9 < ema21:                         bear += 1

    if pdi > mdi and adx_val > 22:            bull += 1
    elif mdi > pdi and adx_val > 22:          bear += 1

    if 55 < rsi_val < 75:                     bull += 1
    elif 25 < rsi_val < 45:                   bear += 1

    if macd_now > sig_now and hist_now > hist_prv: bull += 1
    elif macd_now < sig_now and hist_now < hist_prv: bear += 1

    bb_pos = (price-bb_lo)/(bb_up-bb_lo) if (bb_up-bb_lo)>0 else 0.5
    if 0.5 < bb_pos < 0.85:                   bull += 1
    elif 0.15 < bb_pos < 0.5:                 bear += 1

    if sk > sd and sk < 80:                   bull += 1
    elif sk < sd and sk > 20:                 bear += 1

    if bull >= 6 and bull > bear:   return "UP"
    if bear >= 6 and bear > bull:   return "DOWN"
    return None

# ── Backtest one asset ────────────────────────────────────────────────────────
def backtest_asset(name, ticker):
    print(f"\n  Downloading {name} — 15 days M5...")
    try:
        df = yf.download(ticker, period="15d", interval="5m",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 100:
            print(f"  No data for {name}")
            return None
        df.columns = [c[0].lower() if isinstance(c,tuple) else c.lower()
                      for c in df.columns]
        df = df.reset_index()
    except Exception as e:
        print(f"  Error: {e}")
        return None

    trades  = []
    EXPIRY  = 1   # 1 candle forward = 5 mins

    for i in range(60, len(df) - EXPIRY):
        signal = get_signal(df, i)
        if signal is None:
            continue

        entry_price = float(df.iloc[i]["close"])
        exit_price  = float(df.iloc[i + EXPIRY]["close"])
        ts          = df.iloc[i]["datetime"] if "datetime" in df.columns else df.index[i]

        # Convert to IST hour
        try:
            ts_ist  = pd.Timestamp(ts).tz_localize("UTC").tz_convert(IST) if pd.Timestamp(ts).tzinfo is None else pd.Timestamp(ts).tz_convert(IST)
            hour_ist = ts_ist.hour
            weekday  = ts_ist.weekday()
        except:
            hour_ist = 12
            weekday  = 0

        # Skip weekends
        if weekday >= 5:
            continue

        if signal == "UP":
            result = "WIN" if exit_price > entry_price else ("TIE" if exit_price == entry_price else "LOSS")
        else:
            result = "WIN" if exit_price < entry_price else ("TIE" if exit_price == entry_price else "LOSS")

        trades.append({
            "signal":    signal,
            "result":    result,
            "hour_ist":  hour_ist,
            "weekday":   weekday,
            "entry":     entry_price,
            "exit":      exit_price,
        })

    return trades

# ── Analysis ──────────────────────────────────────────────────────────────────
def analyze_results(name, trades):
    if not trades:
        print(f"  {name}: no trades found")
        return

    df = pd.DataFrame(trades)
    total  = len(df)
    wins   = len(df[df["result"]=="WIN"])
    losses = len(df[df["result"]=="LOSS"])
    ties   = len(df[df["result"]=="TIE"])
    wr     = wins/(wins+losses)*100 if (wins+losses)>0 else 0

    print(f"\n  {name}")
    print(f"  {'─'*40}")
    print(f"  Total signals : {total}")
    print(f"  Wins          : {wins}")
    print(f"  Losses        : {losses}")
    print(f"  Ties          : {ties}")
    print(f"  Win Rate      : {wr:.1f}%")

    verdict = "✅ PROFITABLE" if wr >= 58 else ("⚠️  BREAKEVEN" if wr >= 50 else "❌ LOSING")
    print(f"  Verdict       : {verdict}")

    # Best hours
    print(f"\n  Best hours to trade (IST):")
    hour_stats = []
    for hour in sorted(df["hour_ist"].unique()):
        h_df = df[df["hour_ist"]==hour]
        h_wins = len(h_df[h_df["result"]=="WIN"])
        h_loss = len(h_df[h_df["result"]=="LOSS"])
        h_total = h_wins + h_loss
        if h_total < 3: continue
        h_wr = h_wins/h_total*100
        hour_stats.append((hour, h_wr, h_total))

    # Sort by win rate
    hour_stats.sort(key=lambda x: x[1], reverse=True)
    for hour, h_wr, h_total in hour_stats[:5]:
        bar   = "█" * int(h_wr/10)
        ist_h = f"{hour:02d}:00–{hour+1:02d}:00"
        flag  = "⭐" if h_wr >= 60 else ""
        print(f"    {ist_h} IST  {bar:<10} {h_wr:.0f}%  ({h_total} trades) {flag}")

    return {"name": name, "wr": wr, "total": total,
            "wins": wins, "losses": losses, "hour_stats": hour_stats}

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(SEP)
    print("  CRYPTO KNIGHT — BACKTESTER")
    print("  15 days M5 historical data")
    print("  Same strategy as live scanner")
    print("  6/7 indicators required")
    print(SEP)

    all_results = []

    # Download all 6 assets in parallel — much faster
    def run_asset(args):
        name, ticker = args
        trades = backtest_asset(name, ticker)
        if trades:
            return analyze_results(name, trades)
        return None

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(run_asset, (n, t)): n for n, t in ASSETS.items()}
        for future in as_completed(futures):
            res = future.result()
            if res:
                all_results.append(res)

    # ── Overall summary ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  OVERALL SUMMARY")
    print(f"  {'─'*50}")

    if not all_results:
        print("  No results — check internet connection")
        return

    # Sort by win rate
    all_results.sort(key=lambda x: x["wr"], reverse=True)

    print(f"\n  ASSET RANKING (by win rate):")
    for r in all_results:
        bar  = "█" * int(r["wr"]/10)
        flag = "✅ TRADE" if r["wr"] >= 58 else ("⚠️  MAYBE" if r["wr"] >= 50 else "❌ AVOID")
        print(f"  {r['name']:<10} {bar:<10} {r['wr']:.1f}%  {flag}")

    # Best overall hours across all assets
    all_hours = {}
    for r in all_results:
        for hour, wr, total in r.get("hour_stats", []):
            if hour not in all_hours:
                all_hours[hour] = {"wins": 0, "total": 0}
            wins_n = int(wr/100 * total)
            all_hours[hour]["wins"]  += wins_n
            all_hours[hour]["total"] += total

    print(f"\n  BEST TIMES TO TRADE (across all assets):")
    sorted_hours = sorted(all_hours.items(),
                          key=lambda x: x[1]["wins"]/x[1]["total"] if x[1]["total"]>5 else 0,
                          reverse=True)
    for hour, stat in sorted_hours[:6]:
        if stat["total"] < 5: continue
        wr   = stat["wins"]/stat["total"]*100
        ist  = f"{hour:02d}:00–{hour+1:02d}:00 IST"
        flag = "⭐ BEST" if wr >= 60 else ""
        print(f"  {ist}   {wr:.0f}% win rate   {flag}")

    # Top asset to trade
    best = all_results[0]
    print(f"\n  TOP ASSET    : {best['name']} ({best['wr']:.1f}% win rate)")
    if best.get("hour_stats"):
        bh = best["hour_stats"][0]
        print(f"  BEST TIME    : {bh[0]:02d}:00–{bh[0]+1:02d}:00 IST ({bh[1]:.0f}% on this asset)")
    print(f"\n  Run this backtest weekly to stay updated.")
    print(SEP)

if __name__ == "__main__":
    main()
    
