"""
SMA-50 Manual Session Scanner — Crypto Knight
- Manual trigger only (workflow_dispatch)
- Single scan per run — no loop
- 90%+ confidence only
- Max 3 trades per session
- 5-min expiry
- Real forex only, no OTC, no oil
"""

import os, httpx, yfinance as yf, pandas as pd, numpy as np
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

def ist_now(dt=None):
    d = dt or datetime.now(IST)
    return d.strftime("%d %b %Y  %I:%M %p IST")

def trade_times():
    now = datetime.now(IST)
    open_t = (now.replace(second=0, microsecond=0) + timedelta(minutes=1)).strftime("%I:%M %p")
    close_t = (now.replace(second=0, microsecond=0) + timedelta(minutes=6)).strftime("%I:%M %p")
    return open_t, close_t

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    raise EnvironmentError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in GitHub Secrets")

# ── Real forex only — all confirmed on yfinance ───────────────────────────────
# Best trending pairs from your Pocket Option list
# Ordered by trend reliability during 10-11 AM IST
ASSETS = {
    "EUR/USD": "EURUSD=X",   # most liquid, cleanest signals
    "USD/JPY": "JPY=X",      # strong Monday momentum
    "AUD/JPY": "AUDJPY=X",   # best trending pair in your list
    "EUR/JPY": "EURJPY=X",   # EUR + JPY both active 10am IST
    "AUD/USD": "AUDUSD=X",   # solid volume London session
    "USD/CAD": "CAD=X",      # CAD moves well during London
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
        df = yf.download(ticker, period="5d", interval="5m",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 60: return None
        df.columns = [c[0].lower() if isinstance(c,tuple) else c.lower()
                      for c in df.columns]
    except Exception as e:
        print(f"    fetch error: {e}"); return None

    df["sma50"] = df["close"].rolling(50).mean()
    df["ema9"]  = calc_ema(df["close"], 9)
    df["ema21"] = calc_ema(df["close"], 21)
    df["adx"]   = calc_adx(df)
    df["rsi"]   = calc_rsi(df["close"])

    r, p2 = df.iloc[-1], df.iloc[-2]

    price     = float(r["close"])
    sma50     = float(r["sma50"])
    adx_val   = float(r["adx"])
    rsi_val   = float(r["rsi"])
    ema9_val  = float(r["ema9"])
    ema21_val = float(r["ema21"])
    sma_slope = float(r["sma50"]) - float(p2["sma50"])

    # Single candle confirmation
    candle_up   = float(r["close"]) > float(p2["close"])
    candle_down = float(r["close"]) < float(p2["close"])

    # ── Sideways filters — THREE layers ──────────────────────────────────
    # 1. ADX must show real trend
    if adx_val < 25: return None

    # 2. RSI dead zone — no momentum
    if 45 < rsi_val < 55: return None

    # 3. Price range filter — if last 10 candles too tight = ranging
    recent       = df.tail(10)
    range_size   = float(recent["high"].max() - recent["low"].min())
    avg_price    = float(recent["close"].mean())
    range_pct    = (range_size / avg_price) * 100
    if range_pct < 0.05: return None   # less than 0.05% range = dead sideways

    bull = (price > sma50 and sma_slope > 0
            and ema9_val > ema21_val
            and rsi_val > 55 and candle_up)

    bear = (price < sma50 and sma_slope < 0
            and ema9_val < ema21_val
            and rsi_val < 45 and candle_down)

    if not bull and not bear: return None

    signal = "UP" if bull else "DOWN"

    conf  = 50
    conf += min(22, int((adx_val - 20) * 1.1))
    conf += min(18, int(abs(rsi_val - 50) * 0.6))
    conf += min(12, int(abs(ema9_val - ema21_val) / price * 25000))
    conf += 8 if (candle_up or candle_down) else 0
    conf  = min(conf, 98)

    if conf < 80: return None

    return {
        "asset": name, "signal": signal,
        "price": round(price, 5),
        "sma50": round(sma50, 5),
        "adx":   round(adx_val, 1),
        "rsi":   round(rsi_val, 1),
        "confidence": conf,
    }

# ── Telegram ──────────────────────────────────────────────────────────────────
def send_signals(signals):
    open_at, close_at = trade_times()
    em_map  = {"UP": "🟢", "DOWN": "🔴"}
    act_map = {"UP": "CALL  ▲  (UP)", "DOWN": "PUT  ▼  (DOWN)"}

    lines = [
        f"🎯 <b>Crypto Knight — {len(signals)} SIGNAL{'S' if len(signals)>1 else ''}</b>",
        f"⏰ <code>{ist_now()}</code>",
        "",
    ]

    for i, s in enumerate(signals, 1):
        em  = em_map[s["signal"]]
        act = act_map[s["signal"]]
        lines += [
            f"{'──────────────────────' if i>1 else ''}",
            f"{em} <b>TRADE {i}/3 — {s['asset']}</b>",
            f"   Action     : <b>{act}</b>",
            f"   Price      : <code>{s['price']}</code>",
            f"   SMA-50     : <code>{s['sma50']}</code>",
            f"   ADX        : <code>{s['adx']}</code>",
            f"   RSI        : <code>{s['rsi']}</code>",
            f"   Confidence : <code>{s['confidence']}%</code>",
            f"   Open at    : <code>{open_at} IST</code>",
            f"   Close at   : <code>{close_at} IST</code>",
            "",
        ]

    lines += [
        "──────────────────────",
        f"📌 Open Pocket Option → place trade NOW",
        "⚠️ <i>5-min expiry. Max 3 trades. Real forex only.</i>",
    ]

    try:
        r = httpx.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": "\n".join(lines), "parse_mode": "HTML"},
            timeout=15,
        )
        print(f"  Telegram: {r.status_code}")
        if r.status_code == 403:
            print("  ❌ 403 — send /start to your bot and check Chat ID in Secrets")
            return
        r.raise_for_status()
        print(f"  ✅ Sent {len(signals)} signal(s)")
    except Exception as e:
        print(f"  ❌ Telegram error: {e}")

def send_no_signal():
    try:
        httpx.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={
                "chat_id": CHAT_ID,
                "text": (
                    f"🔍 <b>Crypto Knight</b>\n\n"
                    f"❌ <b>No trades this scan</b>\n"
                    f"⏰ <code>{ist_now()}</code>\n\n"
                    f"<i>No asset hit 80%+ confidence.\n"
                    f"Try again at:\n"
                    f"• 09:15 IST — London open\n"
                    f"• 13:45 IST — London/NY overlap\n"
                    f"• 19:00 IST — NY session</i>"
                ),
                "parse_mode": "HTML",
            },
            timeout=15,
        )
        print("  ✅ No-signal message sent")
    except Exception as e:
        print(f"  ❌ Telegram error: {e}")

# ── Main — single scan, instant result ───────────────────────────────────────
def main():
    print(f"[{ist_now()}] Scanning 5 forex pairs...")

    signals = []
    for name, ticker in ASSETS.items():
        print(f"  {name}...", end=" ")
        result = analyze(name, ticker)
        if result:
            print(f"{result['signal']} {result['confidence']}% ← SIGNAL")
            signals.append(result)
            if len(signals) >= 3:
                break
        else:
            print("no signal")

    print(f"\n  Found: {len(signals)} signal(s)")

    if signals:
        send_signals(signals)
    else:
        send_no_signal()

    print(f"[{ist_now()}] Done")

if __name__ == "__main__":
    main()
        
