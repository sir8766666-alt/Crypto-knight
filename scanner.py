"""
Autonomous SMA-50 Signal Scanner
Runs via GitHub Actions every 5 minutes — no server needed
Sends Telegram alerts ONLY for high-confidence UP/DOWN signals
"""

import os
import httpx
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ── Indian Standard Time (UTC+5:30) ──────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))

def ist_now() -> str:
    return datetime.now(IST).strftime("%d %b %Y  %I:%M %p IST")

# ── Config from GitHub Secrets ────────────────────────────────────────────────
BOT_TOKEN = "8754354364:AAEVl7JUG9TT-Qg7E32dacUhGtJyjQ7khoI"  # set in GitHub Secrets
CHAT_ID   = "8754354364"    # your chat/group/channel id

# ── Assets ────────────────────────────────────────────────────────────────────
ASSETS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "XAU/USD": "GC=F",
    "OIL/USD": "CL=F",
    "BTC/USD": "BTC-USD",
    "USD/JPY": "JPY=X",
}

# ── Market session check — avoid dead hours ───────────────────────────────────
def is_market_active() -> tuple[bool, str]:
    """
    Forex/Commodities: 24/5 (Mon-Fri).
    Crypto: always.
    Skip scanning 00:00–06:00 IST on weekdays (thin liquidity).
    Skip weekends entirely for forex.
    Returns (active, reason).
    """
    now_ist = datetime.now(IST)
    weekday = now_ist.weekday()   # 0=Mon … 6=Sun
    hour    = now_ist.hour

    if weekday == 5:   # Saturday
        return False, "Weekend — forex markets closed"
    if weekday == 6 and hour < 19:  # Sunday before ~19:30 IST (Sydney open)
        return False, "Weekend — markets not yet open"
    if 0 <= hour < 6:
        return False, "Dead hours 00:00–06:00 IST — thin liquidity, skipping"
    return True, "Market active"

# ── Indicators ────────────────────────────────────────────────────────────────
def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm   <= minus_dm] = 0
    minus_dm[minus_dm <= plus_dm]  = 0
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr      = tr.ewm(span=period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(span=period,  adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean()

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

# ── Core analysis ─────────────────────────────────────────────────────────────
def analyze(name: str, ticker: str) -> dict | None:
    """
    Returns signal dict if tradeable, None if sideways/neutral.

    Strategy (WIN-FOCUSED):
      - Sideways filter : ADX < 22  → skip
      - Trend filter    : price side of SMA-50
      - Momentum filter : EMA-9 vs EMA-21 (fast trend confirm)
      - Strength filter : RSI not in 45–55 dead zone
      - Candle confirm  : last close stronger than previous close

    Confidence built from:
      ADX strength + RSI extremity + EMA separation + candle momentum
    Only fires signal if confidence >= 65 (reduces noise trades)
    """
    try:
        df = yf.download(ticker, period="3d", interval="1m",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 60:
            return None
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                      for c in df.columns]
    except Exception:
        return None

    df["sma50"] = df["close"].rolling(50).mean()
    df["ema9"]  = compute_ema(df["close"], 9)
    df["ema21"] = compute_ema(df["close"], 21)
    df["adx"]   = compute_adx(df)
    df["rsi"]   = compute_rsi(df["close"])

    r = df.iloc[-1]   # latest bar
    p = df.iloc[-2]   # previous bar

    price    = float(r["close"])
    sma50    = float(r["sma50"])
    ema9     = float(r["ema9"])
    ema21    = float(r["ema21"])
    adx_val  = float(r["adx"])
    rsi_val  = float(r["rsi"])
    sma_slope = float(r["sma50"]) - float(p["sma50"])
    candle_up = float(r["close"]) > float(p["close"])

    # ── Hard filters ──────────────────────────────────────────────────────────
    if adx_val < 22:                        # no trend
        return None
    if 44 < rsi_val < 56:                   # RSI in dead zone
        return None

    # ── Signal detection ──────────────────────────────────────────────────────
    bull = (price > sma50 and sma_slope > 0
            and ema9 > ema21 and rsi_val > 56 and candle_up)

    bear = (price < sma50 and sma_slope < 0
            and ema9 < ema21 and rsi_val < 44 and not candle_up)

    if not bull and not bear:
        return None

    signal = "UP" if bull else "DOWN"

    # ── Confidence score ──────────────────────────────────────────────────────
    conf = 50
    conf += min(20, int((adx_val - 22) * 0.9))     # trend strength
    rsi_ext = abs(rsi_val - 50)
    conf += min(15, int(rsi_ext * 0.5))             # RSI extremity
    ema_sep = abs(ema9 - ema21) / price * 10000
    conf += min(10, int(ema_sep * 2))               # EMA separation
    conf  = min(conf, 95)

    if conf < 65:                                    # not worth trading
        return None

    decimals = 2 if "BTC" in name else 5

    return {
        "asset":     name,
        "signal":    signal,
        "price":     round(price, decimals),
        "sma50":     round(sma50, decimals),
        "adx":       round(adx_val, 1),
        "rsi":       round(rsi_val, 1),
        "confidence":conf,
        "time_ist":  ist_now(),
    }

# ── Telegram sender ───────────────────────────────────────────────────────────
def send_telegram(signals: list[dict]):
    if not signals:
        return

    lines = ["<b>🎯 SMA-50 SIGNAL ALERT</b>", ""]

    for s in signals:
        if s["signal"] == "UP":
            emoji  = "🟢"
            action = "CALL ▲"
            advice = "Place a <b>CALL (UP)</b> trade"
        else:
            emoji  = "🔴"
            action = "PUT ▼"
            advice = "Place a <b>PUT (DOWN)</b> trade"

        lines += [
            f"{emoji} <b>{s['asset']}</b>  —  <b>{action}</b>",
            f"💰 Price       : <code>{s['price']}</code>",
            f"📈 SMA-50      : <code>{s['sma50']}</code>",
            f"⚡ ADX         : <code>{s['adx']}</code>",
            f"📊 RSI         : <code>{s['rsi']}</code>",
            f"🎯 Confidence  : <code>{s['confidence']}%</code>",
            f"⏰ Time (IST)  : <code>{s['time_ist']}</code>",
            f"⏱️ Expiry       : <b>1 MINUTE</b>",
            f"📌 Action      : {advice}",
            "──────────────────────",
        ]

    lines += [
        "",
        "⚠️ <i>Paper trading only. Always verify before live money.</i>",
        f"🤖 <i>Auto-scan by SMA-50 Engine</i>",
    ]

    text = "\n".join(lines)

    url  = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    resp = httpx.post(url, json={
        "chat_id":    CHAT_ID,
        "text":       text,
        "parse_mode": "HTML",
    }, timeout=15)
    print("Status Code:", resp.status_code)
    print("Response Body:", resp.text)
    resp.raise_for_status()
    print(f"✅ Sent {len(signals)} signal(s) to Telegram")


def send_no_signal_ping():
    """Sends a quiet status ping every hour so you know the bot is alive."""
    now = datetime.now(IST)
    if now.minute > 5:   # only send near the top of the hour
        return
    url  = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    httpx.post(url, json={
        "chat_id":    CHAT_ID,
        "text":       f"🔍 <b>SMA-50 Scanner</b> — No signal this scan\n⏰ <code>{ist_now()}</code>\n<i>Markets active, watching all 6 pairs...</i>",
        "parse_mode": "HTML",
    }, timeout=15)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"[{ist_now()}] Scanner starting...")

    active, reason = is_market_active()
    if not active:
        print(f"⏸  Skipping: {reason}")
        return

    signals = []
    for name, ticker in ASSETS.items():
        print(f"  Analyzing {name}...")
        result = analyze(name, ticker)
        if result:
            print(f"  → {result['signal']} | conf={result['confidence']}%")
            signals.append(result)
        else:
            print(f"  → No signal")

    if signals:
        send_telegram(signals)
    else:
        print("No tradeable signals this scan.")
        send_no_signal_ping()

if __name__ == "__main__":
    main()

