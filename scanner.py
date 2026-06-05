"""
Crypto Knight — Manual Session Scanner
- workflow_dispatch only (you press Run)
- Scans ONCE, sends signal if 90%+ confidence, exits
- No oil
- OTC-friendly pairs for 09:00–11:00 IST
- If no signal → sends "No trade this time" message
"""

import os, httpx, yfinance as yf, pandas as pd, numpy as np
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

def ist_now():
    return datetime.now(IST).strftime("%d %b %Y  %I:%M %p IST")

def expiry_time(minutes=5):
    now = datetime.now(IST)
    exp = now.replace(second=0, microsecond=0) + timedelta(minutes=minutes+1)
    return exp.strftime("%I:%M %p IST")

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    raise EnvironmentError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in GitHub Secrets")

# ── Assets — OTC pairs available on Pocket Option, no oil ────────────────────
# These are the real-market equivalents — direction matches OTC feed
ASSETS = {
    "AUD/CAD OTC": "AUDCAD=X",
    "AUD/CHF OTC": "AUDCHF=X",
    "EUR/USD OTC": "EURUSD=X",
    "GBP/USD OTC": "GBPUSD=X",
    "USD/JPY OTC": "JPY=X",
    "EUR/JPY OTC": "EURJPY=X",
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
    return dx.ewm(span=p,adjust=False).mean()

def calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(span=p,adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=p,adjust=False).mean()
    return 100-(100/(1+g/l.replace(0,np.nan)))

def calc_ema(s, p):
    return s.ewm(span=p,adjust=False).mean()

# ── Analysis ──────────────────────────────────────────────────────────────────
def analyze(name, ticker):
    try:
        df = yf.download(ticker, period="3d", interval="1m",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 60: return None
        df.columns = [c[0].lower() if isinstance(c,tuple) else c.lower()
                      for c in df.columns]
    except Exception as e:
        print(f"    {name} fetch error: {e}")
        return None

    df["sma50"] = df["close"].rolling(50).mean()
    df["ema9"]  = calc_ema(df["close"], 9)
    df["ema21"] = calc_ema(df["close"], 21)
    df["adx"]   = calc_adx(df)
    df["rsi"]   = calc_rsi(df["close"])

    r, p2, p3 = df.iloc[-1], df.iloc[-2], df.iloc[-3]

    price     = float(r["close"])
    sma50     = float(r["sma50"])
    adx_val   = float(r["adx"])
    rsi_val   = float(r["rsi"])
    ema9_val  = float(r["ema9"])
    ema21_val = float(r["ema21"])
    sma_slope = float(r["sma50"]) - float(p2["sma50"])

    # 3 consecutive candles same direction = strong momentum
    candles_up   = (float(r["close"]) > float(p2["close"]) > float(p3["close"]))
    candles_down = (float(r["close"]) < float(p2["close"]) < float(p3["close"]))

    # Strict filters
    if adx_val < 25: return None
    if 42 < rsi_val < 58: return None

    bull = (price > sma50 and sma_slope > 0
            and ema9_val > ema21_val
            and rsi_val > 58 and candles_up)

    bear = (price < sma50 and sma_slope < 0
            and ema9_val < ema21_val
            and rsi_val < 42 and candles_down)

    if not bull and not bear: return None

    signal = "UP" if bull else "DOWN"

    conf  = 50
    conf += min(22, int((adx_val - 25) * 1.1))
    conf += min(18, int(abs(rsi_val - 50) * 0.6))
    conf += min(12, int(abs(ema9_val - ema21_val) / price * 25000))
    conf += 8 if (candles_up or candles_down) else 0
    conf  = min(conf, 98)

    if conf < 90: return None

    return {
        "asset": name, "signal": signal,
        "price": round(price, 5),
        "sma50": round(sma50, 5),
        "adx":   round(adx_val, 1),
        "rsi":   round(rsi_val, 1),
        "confidence": conf,
    }

# ── Telegram ──────────────────────────────────────────────────────────────────
def tg(text):
    try:
        r = httpx.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=15,
        )
        if r.status_code == 403:
            print("  ❌ 403 — send /start to your bot first")
            return
        r.raise_for_status()
    except Exception as e:
        print(f"  Telegram error: {e}")

def send_signal(s, num, total):
    em, act, btn = (
        ("🟢", "CALL  ▲", "BUY") if s["signal"] == "UP"
        else ("🔴", "PUT   ▼", "SELL")
    )
    exp = expiry_time(5)
    tg(
        f"🎯 <b>TRADE {num}/{total} — Crypto Knight</b>\n"
        f"\n"
        f"{em} <b>{s['asset']}</b>\n"
        f"   Direction  : <b>{act}</b>\n"
        f"   Press      : <b>{btn} on Pocket Option</b>\n"
        f"\n"
        f"   Price      : <code>{s['price']}</code>\n"
        f"   SMA-50     : <code>{s['sma50']}</code>\n"
        f"   ADX        : <code>{s['adx']}</code>\n"
        f"   RSI        : <code>{s['rsi']}</code>\n"
        f"   Confidence : <code>{s['confidence']}%</code>\n"
        f"\n"
        f"⏰ Scanned    : <code>{ist_now()}</code>\n"
        f"⏱️ Set expiry  : <code>5 minutes</code>\n"
        f"🔒 Closes at  : <code>{exp}</code>\n"
        f"\n"
        f"──────────────────────\n"
        f"⚠️ <i>Paper trading only. Max 3 trades today.</i>"
    )
    print(f"  ✅ {s['asset']} {s['signal']} {s['confidence']}%")

def send_no_trade():
    tg(
        f"🔍 <b>Crypto Knight — No Trade This Time</b>\n"
        f"\n"
        f"⏰ Scanned : <code>{ist_now()}</code>\n"
        f"📊 Result  : No asset crossed 90% confidence\n"
        f"\n"
        f"<i>Market conditions not ideal right now.\n"
        f"Best windows: 09:00–11:00 or 13:30–15:30 IST\n"
        f"Try again in 15–30 minutes.</i>"
    )
    print("  No signals — sent no-trade message")

# ── Main — single scan, instant result ───────────────────────────────────────
def main():
    MAX_TRADES = 3
    print(f"[{ist_now()}] Scanning {len(ASSETS)} assets for 90%+ signals...")

    signals = []
    for name, ticker in ASSETS.items():
        print(f"  {name}...", end=" ")
        result = analyze(name, ticker)
        if result:
            print(f"{result['signal']} {result['confidence']}% ← SIGNAL")
            signals.append(result)
        else:
            print("no signal")

    # Sort best confidence first
    signals.sort(key=lambda x: -x["confidence"])

    # Take top 3 max
    top = signals[:MAX_TRADES]

    if top:
        print(f"\n  {len(top)} signal(s) found — sending to Telegram")
        for i, s in enumerate(top):
            send_signal(s, i+1, len(top))
    else:
        send_no_trade()

    print(f"\n[{ist_now()}] Done")

if __name__ == "__main__":
    main()
    
