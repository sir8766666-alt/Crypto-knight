"""
Crypto Knight — Ultimate Scanner
Prints full trade details in GitHub Actions output AND sends to Telegram
SMA50 + MACD + ADX + RSI + Bollinger | M5 candles | 5-min expiry
"""

import os, httpx, yfinance as yf, pandas as pd, numpy as np
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

def ist_now(dt=None):
    d = dt or datetime.now(IST)
    return d.strftime("%d %b %Y  %I:%M %p IST")

def trade_times():
    now     = datetime.now(IST)
    open_t  = (now.replace(second=0, microsecond=0) + timedelta(minutes=1)).strftime("%I:%M %p")
    close_t = (now.replace(second=0, microsecond=0) + timedelta(minutes=6)).strftime("%I:%M %p")
    return open_t, close_t

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    raise EnvironmentError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in GitHub Secrets")

ASSETS = {
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/JPY": "AUDJPY=X",
    "EUR/JPY": "EURJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
}

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
    ema12  = s.ewm(span=12, adjust=False).mean()
    ema26  = s.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal
    return macd, signal, hist

def calc_bollinger(s, p=20):
    mid = s.rolling(p).mean()
    std = s.rolling(p).std()
    return mid + 2*std, mid, mid - 2*std

# ── Separator helpers for console output ─────────────────────────────────────
SEP  = "=" * 52
SEP2 = "-" * 52

def print_header():
    print(SEP)
    print("   CRYPTO KNIGHT — SIGNAL SCANNER")
    print(f"   {ist_now()}")
    print(f"   Strategy : SMA50 + MACD + ADX + RSI + BB")
    print(f"   Timeframe: M5 candles | Expiry: 5 mins")
    print(SEP)

def print_asset_result(name, result, skip_reason):
    if result:
        arrow = "▲ UP   (CALL)" if result["signal"] == "UP" else "▼ DOWN (PUT) "
        cross = " ⚡ FRESH CROSS" if result.get("fresh_cross") else ""
        print(f"\n  ✅ {name}")
        print(f"     Signal     : {arrow}{cross}")
        print(f"     Price      : {result['price']}")
        print(f"     SMA-50     : {result['sma50']}")
        print(f"     ADX        : {result['adx']}")
        print(f"     RSI        : {result['rsi']}")
        print(f"     MACD       : {result['macd']}  |  Signal: {result['macd_sig']}")
        print(f"     Confidence : {result['confidence']}%")
    else:
        print(f"  ✗  {name:<10} → {skip_reason}")

def print_summary(signals, skips):
    print(f"\n{SEP2}")
    print(f"  SCAN COMPLETE")
    print(f"  Signals found : {len(signals)}/3")
    print(f"  Skipped       : {len(skips)} assets")
    if signals:
        open_at, close_at = trade_times()
        print(f"\n  ⏰ Open trades  : {open_at} IST")
        print(f"  ⏰ Close trades : {close_at} IST")
        print(f"\n  TRADES TO PLACE:")
        for i, s in enumerate(signals, 1):
            arrow = "▲ CALL (UP)" if s["signal"] == "UP" else "▼ PUT (DOWN)"
            print(f"  {i}. {s['asset']:<10} {arrow}  conf={s['confidence']}%")
    else:
        print(f"\n  ❌ No tradeable signals this scan")
        print(f"  Best times: 14:00–16:00 IST or 19:00–21:00 IST")
    print(SEP)

# ── Core analysis ─────────────────────────────────────────────────────────────
def analyze(name, ticker):
    try:
        df = yf.download(ticker, period="5d", interval="5m",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 60:
            return None, "No data fetched"
        df.columns = [c[0].lower() if isinstance(c,tuple) else c.lower()
                      for c in df.columns]
    except Exception as e:
        return None, f"Fetch error: {e}"

    df["sma50"]                         = df["close"].rolling(50).mean()
    df["ema9"]                          = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema21"]                         = df["close"].ewm(span=21, adjust=False).mean()
    df["adx"], df["pdi"], df["mdi"]     = calc_adx(df)
    df["rsi"]                           = calc_rsi(df["close"])
    df["macd"], df["macd_sig"], df["macd_hist"] = calc_macd(df["close"])
    df["bb_up"], df["bb_mid"], df["bb_low"]     = calc_bollinger(df["close"])

    r, p2 = df.iloc[-1], df.iloc[-2]

    price     = float(r["close"])
    sma50     = float(r["sma50"])
    adx_val   = float(r["adx"])
    pdi       = float(r["pdi"])
    mdi       = float(r["mdi"])
    rsi_val   = float(r["rsi"])
    ema9      = float(r["ema9"])
    ema21     = float(r["ema21"])
    macd_now  = float(r["macd"])
    sig_now   = float(r["macd_sig"])
    hist_now  = float(r["macd_hist"])
    hist_prv  = float(p2["macd_hist"])
    bb_up     = float(r["bb_up"])
    bb_low    = float(r["bb_low"])
    sma_slope = sma50 - float(p2["sma50"])

    # ── Sideways filters ──────────────────────────────────────────────────────
    recent    = df.tail(10)
    range_pct = (float(recent["high"].max()-recent["low"].min())/price)*100

    if adx_val < 18:
        return None, f"ADX {adx_val:.1f} < 18 — no trend"
    if range_pct < 0.04:
        return None, f"Range {range_pct:.3f}% — dead sideways"
    if 47 < rsi_val < 53:
        return None, f"RSI {rsi_val:.1f} — dead zone"

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd_bull       = hist_now > hist_prv and macd_now > sig_now
    macd_bear       = hist_now < hist_prv and macd_now < sig_now
    macd_cross_up   = macd_now > sig_now and float(p2["macd"]) <= float(p2["macd_sig"])
    macd_cross_down = macd_now < sig_now and float(p2["macd"]) >= float(p2["macd_sig"])

    # ── Trend ─────────────────────────────────────────────────────────────────
    trend_up   = price > sma50 and sma_slope > 0 and ema9 > ema21 and pdi > mdi
    trend_down = price < sma50 and sma_slope < 0 and ema9 < ema21 and mdi > pdi

    # ── RSI ───────────────────────────────────────────────────────────────────
    rsi_bull = rsi_val > 54 and rsi_val < 75
    rsi_bear = rsi_val < 46 and rsi_val > 25

    # ── Bollinger ─────────────────────────────────────────────────────────────
    at_bb_top = price >= bb_up  * 0.9999
    at_bb_bot = price <= bb_low * 1.0001

    # ── Final signal ──────────────────────────────────────────────────────────
    bull = trend_up   and rsi_bull and (macd_bull or macd_cross_up)   and not at_bb_top
    bear = trend_down and rsi_bear and (macd_bear or macd_cross_down) and not at_bb_bot

    if not bull and not bear:
        reasons = []
        if not trend_up   and not trend_down: reasons.append("trend not aligned")
        if not rsi_bull   and not rsi_bear:   reasons.append(f"RSI {rsi_val:.0f} neutral")
        if not macd_bull  and not macd_bear:  reasons.append("MACD flat")
        if at_bb_top: reasons.append("at BB upper — overbought")
        if at_bb_bot: reasons.append("at BB lower — oversold")
        return None, " | ".join(reasons) or "mixed signals"

    signal = "UP" if bull else "DOWN"

    conf  = 50
    conf += min(15, int((adx_val - 18) * 0.8))
    conf += min(12, int(abs(rsi_val - 50) * 0.5))
    conf += 15 if (macd_cross_up or macd_cross_down) else 8
    conf += min(8,  int(abs(ema9 - ema21) / price * 10000))
    conf += min(8,  int(abs(pdi - mdi) * 0.3))
    conf  = min(conf, 98)

    if conf < 75:
        return None, f"Confidence {conf}% < 75%"

    return {
        "asset":       name,
        "signal":      signal,
        "price":       round(price, 5),
        "sma50":       round(sma50, 5),
        "adx":         round(adx_val, 1),
        "rsi":         round(rsi_val, 1),
        "macd":        round(macd_now, 6),
        "macd_sig":    round(sig_now, 6),
        "confidence":  conf,
        "fresh_cross": macd_cross_up or macd_cross_down,
    }, None

# ── Telegram ──────────────────────────────────────────────────────────────────
def send_signals(signals):
    open_at, close_at = trade_times()
    lines = [
        "🎯 <b>Crypto Knight — Signal</b>",
        f"⏰ <code>{ist_now()}</code>",
        "",
    ]
    for i, s in enumerate(signals, 1):
        em  = "🟢" if s["signal"] == "UP" else "🔴"
        act = "CALL  ▲  (UP)" if s["signal"] == "UP" else "PUT  ▼  (DOWN)"
        cross = "\n   ⚡ <b>Fresh MACD crossover!</b>" if s.get("fresh_cross") else ""
        lines += [
            "──────────────────────" if i > 1 else "",
            f"{em} <b>TRADE {i}/3 — {s['asset']}</b>",
            f"   Action     : <b>{act}</b>",
            f"   Price      : <code>{s['price']}</code>",
            f"   SMA-50     : <code>{s['sma50']}</code>",
            f"   ADX        : <code>{s['adx']}</code>",
            f"   RSI        : <code>{s['rsi']}</code>",
            f"   MACD       : <code>{s['macd']}</code>",
            f"   Confidence : <code>{s['confidence']}%</code>{cross}",
            f"   Open at    : <code>{open_at} IST</code>",
            f"   Close at   : <code>{close_at} IST</code>",
            "",
        ]
    lines += [
        "──────────────────────",
        "📌 <b>Pocket Option → expiry 5 mins</b>",
        "⚠️ <i>Max 3 trades. Stop after 1 loss.</i>",
    ]
    _tg("\n".join(lines))

def send_no_signal(skips):
    lines = [
        "🔍 <b>Crypto Knight</b>",
        f"⏰ <code>{ist_now()}</code>",
        "",
        "❌ <b>No trades this scan</b>",
        "",
        "📋 Skip reasons:",
    ]
    for name, reason in skips.items():
        lines.append(f"   • <code>{name}</code>: {reason}")
    lines += [
        "",
        "<i>Best times:\n• 14:00–16:00 IST (London peak)\n• 19:00–21:00 IST (NY session)</i>",
    ]
    _tg("\n".join(lines))

def _tg(text):
    try:
        r = httpx.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=15,
        )
        if r.status_code == 403:
            print("  ❌ Telegram 403 — send /start to your bot")
            return
        r.raise_for_status()
        print("  ✅ Telegram message sent")
    except Exception as e:
        print(f"  ❌ Telegram error: {e}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print_header()

    signals, skips = [], {}

    for name, ticker in ASSETS.items():
        result, skip_reason = analyze(name, ticker)
        print_asset_result(name, result, skip_reason)
        if result:
            signals.append(result)
            if len(signals) >= 3:
                break
        else:
            skips[name] = skip_reason

    print_summary(signals, skips)

    # Send to Telegram
    print("\n  Sending to Telegram...")
    if signals:
        send_signals(signals)
    else:
        send_no_signal(skips)

if __name__ == "__main__":
    main()
    
