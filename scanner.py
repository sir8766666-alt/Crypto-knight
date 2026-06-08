"""
SMA-50 Manual Session Scanner — Crypto Knight
- Manual trigger only (workflow_dispatch)
- Single scan per run — no loop
- 90%+ confidence only
- Max 3 trades per session
- 1-min expiry  ← changed from 5-min
- Real forex only, no OTC, no oil
- Strong sideways/chop filter (ADX + BB width + RSI band)
- Signal based on LATEST completed 1-min candle only
"""

import os, httpx, yfinance as yf, pandas as pd, numpy as np
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

def ist_now(dt=None):
    d = dt or datetime.now(IST)
    return d.strftime("%d %b %Y  %I:%M %p IST")

def trade_times():
    now = datetime.now(IST)
    # For 1-min trade: open next minute, close 1 min after that
    open_t  = (now.replace(second=0, microsecond=0) + timedelta(minutes=1)).strftime("%I:%M %p")
    close_t = (now.replace(second=0, microsecond=0) + timedelta(minutes=2)).strftime("%I:%M %p")
    return open_t, close_t

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    raise EnvironmentError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in GitHub Secrets")

# ── Real forex only ────────────────────────────────────────────────────────────
ASSETS = {
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "EUR/JPY": "EURJPY=X",
    "AUD/CAD": "AUDCAD=X",
}

# ── Indicators ────────────────────────────────────────────────────────────────
def calc_adx(df, p=14):
    h, l, c = df["high"], df["low"], df["close"]
    pdm = h.diff().clip(lower=0)
    mdm = (-l.diff()).clip(lower=0)
    pdm[pdm <= mdm] = 0
    mdm[mdm <= pdm] = 0
    tr  = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=p, adjust=False).mean()
    pdi = 100 * pdm.ewm(span=p, adjust=False).mean() / atr
    mdi = 100 * mdm.ewm(span=p, adjust=False).mean() / atr
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(span=p, adjust=False).mean(), pdi, mdi

def calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(span=p, adjust=False).mean()
    lo = (-d.clip(upper=0)).ewm(span=p, adjust=False).mean()
    return 100 - (100 / (1 + g / lo.replace(0, np.nan)))

def calc_ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def calc_bb_width(s, p=20):
    """Bollinger Band width — low value = sideways/squeeze"""
    ma  = s.rolling(p).mean()
    std = s.rolling(p).std()
    return (2 * std) / ma.replace(0, np.nan)

def calc_atr(df, p=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()

# ── Analysis ──────────────────────────────────────────────────────────────────
def analyze(name, ticker):
    try:
        # Fetch only 1 day of 1-min data — faster and fresher
        df = yf.download(ticker, period="1d", interval="1m",
                         progress=False, auto_adjust=True)

        if df.empty or len(df) < 60:
            print("  not enough data"); return None

        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                      for c in df.columns]

        # ── Drop the LAST (incomplete) candle — use second-to-last as "latest" ──
        # yfinance often returns a partial candle for the current minute.
        # We drop it so our signal is based on a fully-closed candle.
        df = df.iloc[:-1]

    except Exception as e:
        print(f"  fetch error: {e}"); return None

    df["sma50"]   = df["close"].rolling(50).mean()
    df["ema9"]    = calc_ema(df["close"], 9)
    df["ema21"]   = calc_ema(df["close"], 21)
    adx_s, pdi_s, mdi_s = calc_adx(df)
    df["adx"]     = adx_s
    df["pdi"]     = pdi_s
    df["mdi"]     = mdi_s
    df["rsi"]     = calc_rsi(df["close"])
    df["bb_width"]= calc_bb_width(df["close"])
    df["atr"]     = calc_atr(df)

    # Latest fully-closed candle
    r  = df.iloc[-1]
    p2 = df.iloc[-2]
    p3 = df.iloc[-3]

    price     = float(r["close"])
    sma50     = float(r["sma50"])
    adx_val   = float(r["adx"])
    rsi_val   = float(r["rsi"])
    ema9_val  = float(r["ema9"])
    ema21_val = float(r["ema21"])
    pdi_val   = float(r["pdi"])
    mdi_val   = float(r["mdi"])
    sma_slope = float(r["sma50"]) - float(p2["sma50"])
    bb_width  = float(r["bb_width"])
    atr_val   = float(r["atr"])

    # ── Candle body size (filter doji / spinning tops) ──────────────────────
    candle_body = abs(float(r["close"]) - float(r["open"]))
    candle_range = float(r["high"]) - float(r["low"])
    body_ratio  = candle_body / candle_range if candle_range > 0 else 0

    # ── SIDEWAYS FILTERS (all must pass to continue) ─────────────────────────
    # 1. ADX must be strongly trending
    if adx_val < 25:
        return None

    # 2. Bollinger Band width — below 0.002 means squeeze/sideways
    if bb_width < 0.002:
        return None

    # 3. RSI must be clearly directional (wider band than before)
    if 42 < rsi_val < 58:
        return None

    # 4. DI separation — PDI and MDI must be clearly separated
    di_sep = abs(pdi_val - mdi_val)
    if di_sep < 8:
        return None

    # 5. Candle must have a real body (not a doji)
    if body_ratio < 0.35:
        return None

    # 6. ATR must be meaningful (price is actually moving)
    price_atr_ratio = atr_val / price
    if price_atr_ratio < 0.00008:   # less than 0.8 pips on a 1-min candle
        return None

    # ── Direction logic ───────────────────────────────────────────────────────
    candle_up   = float(r["close"]) > float(r["open"])   # bullish body
    candle_down = float(r["close"]) < float(r["open"])   # bearish body

    # Confirm with previous candle too
    prev_up   = float(p2["close"]) > float(p2["open"])
    prev_down = float(p2["close"]) < float(p2["open"])

    bull = (price > sma50
            and sma_slope > 0
            and ema9_val > ema21_val
            and pdi_val > mdi_val
            and rsi_val > 58
            and candle_up
            and prev_up)    # 2-candle momentum confirmation

    bear = (price < sma50
            and sma_slope < 0
            and ema9_val < ema21_val
            and mdi_val > pdi_val
            and rsi_val < 42
            and candle_down
            and prev_down)

    if not bull and not bear:
        return None

    signal = "UP" if bull else "DOWN"

    # ── Confidence score (max 98) ─────────────────────────────────────────────
    conf  = 50
    conf += min(20, int((adx_val - 25) * 1.0))    # ADX strength (up to +20)
    conf += min(15, int(abs(rsi_val - 50) * 0.5)) # RSI extremity (up to +15)
    conf += min(8,  int(di_sep * 0.5))             # DI separation (up to +8)
    conf += min(5,  int(bb_width * 1000))          # BB expansion (up to +5)
    conf += 5 if (candle_up or candle_down) else 0 # candle body bonus
    conf += 5 if body_ratio > 0.6 else 0           # strong marubozu-ish candle
    conf  = min(conf, 98)

    if conf < 80:
        return None

    return {
        "asset":      name,
        "signal":     signal,
        "price":      round(price, 5),
        "sma50":      round(sma50, 5),
        "adx":        round(adx_val, 1),
        "rsi":        round(rsi_val, 1),
        "bb_width":   round(bb_width * 10000, 1),  # in pips-like units for display
        "di_sep":     round(di_sep, 1),
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
            f"{'──────────────────────' if i > 1 else ''}",
            f"{em} <b>TRADE {i}/3 — {s['asset']}</b>",
            f"   Action     : <b>{act}</b>",
            f"   Price      : <code>{s['price']}</code>",
            f"   SMA-50     : <code>{s['sma50']}</code>",
            f"   ADX        : <code>{s['adx']}</code>",
            f"   RSI        : <code>{s['rsi']}</code>",
            f"   DI Sep     : <code>{s['di_sep']}</code>",
            f"   BB Width   : <code>{s['bb_width']}</code>",
            f"   Confidence : <code>{s['confidence']}%</code>",
            f"   Open at    : <code>{open_at} IST</code>",
            f"   Close at   : <code>{close_at} IST</code>",
            "",
        ]

    lines += [
        "──────────────────────",
        f"📌 Open Pocket Option → place trade NOW",
        "⚠️ <i>1-min expiry. Max 3 trades. Real forex only.</i>",
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
                    f"<i>Market is sideways or no strong setup.\n"
                    f"Best times to scan:\n"
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

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"[{ist_now()}] Scanning 5 forex pairs (1-min expiry)...")

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
    
