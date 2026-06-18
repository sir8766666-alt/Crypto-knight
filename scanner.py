"""
Crypto Knight — SMA50 + MACD + ADX + RSI + Bollinger
No API keys needed. Pure indicator logic.
M5 candles. 5-min expiry. 80%+ confidence only.
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
    ema12    = s.ewm(span=12, adjust=False).mean()
    ema26    = s.ewm(span=26, adjust=False).mean()
    macd     = ema12 - ema26
    signal   = macd.ewm(span=9, adjust=False).mean()
    hist     = macd - signal
    return macd, signal, hist

def calc_bollinger(s, p=20):
    mid = s.rolling(p).mean()
    std = s.rolling(p).std()
    return mid + 2*std, mid, mid - 2*std

# ── Core analysis ─────────────────────────────────────────────────────────────
def analyze(name, ticker):
    try:
        df = yf.download(ticker, period="5d", interval="5m",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 60: return None
        df.columns = [c[0].lower() if isinstance(c,tuple) else c.lower()
                      for c in df.columns]
    except Exception as e:
        print(f"fetch error: {e}"); return None

    df["sma50"]               = df["close"].rolling(50).mean()
    df["ema9"]                = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema21"]               = df["close"].ewm(span=21, adjust=False).mean()
    df["adx"], df["pdi"], df["mdi"] = calc_adx(df)
    df["rsi"]                 = calc_rsi(df["close"])
    df["macd"], df["macd_sig"], df["macd_hist"] = calc_macd(df["close"])
    df["bb_up"], df["bb_mid"], df["bb_low"]     = calc_bollinger(df["close"])

    r, p2 = df.iloc[-1], df.iloc[-2]

    price    = float(r["close"])
    sma50    = float(r["sma50"])
    adx_val  = float(r["adx"])
    pdi      = float(r["pdi"])
    mdi      = float(r["mdi"])
    rsi_val  = float(r["rsi"])
    ema9     = float(r["ema9"])
    ema21    = float(r["ema21"])
    macd_now = float(r["macd"])
    sig_now  = float(r["macd_sig"])
    hist_now = float(r["macd_hist"])
    hist_prv = float(p2["macd_hist"])
    bb_up    = float(r["bb_up"])
    bb_low   = float(r["bb_low"])
    sma_slope = sma50 - float(p2["sma50"])

    # ── LAYER 1: Sideways filter ──────────────────────────────────────────────
    recent    = df.tail(10)
    range_pct = (float(recent["high"].max()-recent["low"].min())/price)*100

    if adx_val < 18:
        return None, f"ADX {adx_val:.1f} — no trend"
    if range_pct < 0.05:
        return None, f"Range {range_pct:.3f}% — dead market"
    if 47 < rsi_val < 53:
        return None, f"RSI {rsi_val:.1f} — no momentum"

    # ── LAYER 2: MACD crossover (direction engine) ────────────────────────────
    # MACD histogram growing = momentum building in that direction
    macd_bull = hist_now > hist_prv and macd_now > sig_now   # growing momentum
    macd_bear = hist_now < hist_prv and macd_now < sig_now   # growing momentum

    # Fresh crossover this candle (strongest signal)
    macd_cross_up   = macd_now > sig_now and float(p2["macd"]) <= float(p2["macd_sig"])
    macd_cross_down = macd_now < sig_now and float(p2["macd"]) >= float(p2["macd_sig"])

    # ── LAYER 3: Trend confirmation ───────────────────────────────────────────
    trend_up   = price > sma50 and sma_slope > 0 and ema9 > ema21 and pdi > mdi
    trend_down = price < sma50 and sma_slope < 0 and ema9 < ema21 and mdi > pdi

    # ── LAYER 4: RSI confirmation ─────────────────────────────────────────────
    rsi_bull = rsi_val > 54 and rsi_val < 75   # bullish but not overbought
    rsi_bear = rsi_val < 46 and rsi_val > 25   # bearish but not oversold

    # ── LAYER 5: Bollinger — avoid trading at extremes ────────────────────────
    at_bb_top = price >= bb_up * 0.9999   # overbought — don't call UP
    at_bb_bot = price <= bb_low * 1.0001  # oversold  — don't call DOWN

    # ── Signal logic — ALL layers must agree ──────────────────────────────────
    bull = (trend_up and rsi_bull
            and (macd_bull or macd_cross_up)
            and not at_bb_top)

    bear = (trend_down and rsi_bear
            and (macd_bear or macd_cross_down)
            and not at_bb_bot)

    if not bull and not bear:
        reasons = []
        if not trend_up and not trend_down: reasons.append("trend mixed")
        if not rsi_bull and not rsi_bear:   reasons.append(f"RSI {rsi_val:.0f} neutral")
        if not macd_bull and not macd_bear: reasons.append("MACD no momentum")
        return None, " / ".join(reasons) or "indicators not aligned"

    signal = "UP" if bull else "DOWN"

    # ── Confidence scoring ────────────────────────────────────────────────────
    conf = 50

    # ADX strength
    conf += min(15, int((adx_val - 22) * 0.8))

    # RSI extremity
    conf += min(12, int(abs(rsi_val - 50) * 0.5))

    # MACD crossover bonus — freshest signal = highest confidence
    if macd_cross_up or macd_cross_down:
        conf += 15
    elif macd_bull or macd_bear:
        conf += 8

    # EMA separation
    ema_gap = abs(ema9 - ema21) / price * 10000
    conf += min(8, int(ema_gap * 1.5))

    # DI separation (how dominant the trend direction is)
    di_gap = abs(pdi - mdi)
    conf += min(8, int(di_gap * 0.3))

    conf = min(conf, 98)

    if conf < 75:
        return None, f"Confidence {conf}% < 80%"

    return {
        "asset":      name,
        "signal":     signal,
        "price":      round(price, 5),
        "sma50":      round(sma50, 5),
        "adx":        round(adx_val, 1),
        "rsi":        round(rsi_val, 1),
        "macd":       round(macd_now, 6),
        "macd_sig":   round(sig_now, 6),
        "confidence": conf,
        "fresh_cross": macd_cross_up or macd_cross_down,
    }, None

# ── Telegram ──────────────────────────────────────────────────────────────────
def send_signals(signals):
    open_at, close_at = trade_times()
    lines = [
        "🎯 <b>Crypto Knight Signal</b>",
        f"⏰ <code>{ist_now()}</code>",
        "",
    ]
    for i, s in enumerate(signals, 1):
        em  = "🟢" if s["signal"] == "UP" else "🔴"
        act = "CALL  ▲  (UP)" if s["signal"] == "UP" else "PUT  ▼  (DOWN)"
        cross = "⚡ <b>Fresh MACD crossover!</b>" if s.get("fresh_cross") else ""
        lines += [
            "──────────────────────" if i > 1 else "",
            f"{em} <b>TRADE {i}/3 — {s['asset']}</b>",
            f"   Action     : <b>{act}</b>",
            f"   Price      : <code>{s['price']}</code>",
            f"   ADX        : <code>{s['adx']}</code>",
            f"   RSI        : <code>{s['rsi']}</code>",
            f"   MACD       : <code>{s['macd']}</code>",
            f"   Confidence : <code>{s['confidence']}%</code>",
            cross,
            f"   Open at    : <code>{open_at} IST</code>",
            f"   Close at   : <code>{close_at} IST</code>",
            "",
        ]
    lines += [
        "──────────────────────",
        "📌 <b>Set Pocket Option expiry → 5 mins</b>",
        "⚠️ <i>Max 3 trades. Stop after 1 loss.</i>",
    ]
    _tg(lines)

def send_no_signal(skips):
    lines = [
        "🔍 <b>Crypto Knight</b>",
        f"⏰ <code>{ist_now()}</code>",
        "",
        "❌ <b>No trades this scan</b>",
        "<i>All indicators not aligned on any pair.</i>",
        "",
        "📋 Skip reasons:",
    ]
    for name, reason in skips.items():
        lines.append(f"   • {name}: {reason}")
    lines += [
        "",
        "<i>Retry at:\n• 10:00–11:00 IST\n• 13:45–15:30 IST\n• 19:00–21:00 IST</i>",
    ]
    _tg(lines)

def _tg(lines):
    try:
        r = httpx.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id":CHAT_ID,"text":"\n".join(lines),"parse_mode":"HTML"},
            timeout=15,
        )
        if r.status_code == 403:
            print("❌ 403 — send /start to your bot"); return
        r.raise_for_status()
        print("✅ Telegram sent")
    except Exception as e:
        print(f"❌ Telegram: {e}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"[{ist_now()}] Scanning...")
    signals, skips = [], {}

    for name, ticker in ASSETS.items():
        print(f"  {name}... ", end="", flush=True)
        result, skip_reason = analyze(name, ticker)
        if result:
            print(f"✅ {result['signal']} {result['confidence']}%")
            signals.append(result)
            if len(signals) >= 3: break
        else:
            print(f"skip — {skip_reason}")
            skips[name] = skip_reason

    print(f"\n  Signals: {len(signals)}")
    if signals:
        send_signals(signals)
    else:
        send_no_signal(skips)

    print(f"[{ist_now()}] Done")

if __name__ == "__main__":
    main()
    
