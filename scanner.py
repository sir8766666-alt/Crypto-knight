"""
SMA-50 Signal Scanner v3 — GitHub Actions, every 5 min
Internal 60s loop covers each minute. Signals point to NEXT candle.
Secrets come from GitHub Actions env — NEVER hardcode tokens.
"""

import os, time, httpx, yfinance as yf, pandas as pd, numpy as np
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

def ist_now(dt=None):
    d = dt or datetime.now(IST)
    return d.strftime("%d %b %Y  %I:%M %p IST")

def next_candle_time():
    now = datetime.now(IST)
    nxt = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    return nxt.strftime("%I:%M %p IST")

# ── Read from GitHub Secrets — will raise clear error if missing ──────────────
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    raise EnvironmentError(
        "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in GitHub Secrets.\n"
        "Go to: Repo → Settings → Secrets and variables → Actions → New repository secret"
    )

ASSETS = {
    "EUR/USD": "EURUSD=X",   # confirmed on Pocket Option
    "USD/JPY": "JPY=X",      # confirmed on Pocket Option
    "AUD/USD": "AUDUSD=X",   # confirmed on Pocket Option
    "EUR/JPY": "EURJPY=X",   # confirmed on Pocket Option
    "AUD/CAD": "AUDCAD=X",   # confirmed on Pocket Option
    "WTI/OIL": "CL=F",       # WTI Crude Oil — same price as OTC on Pocket Option
}

sent_this_run: set = set()

# ── Market hours filter ───────────────────────────────────────────────────────
OIL_ASSETS = {"WTI/OIL"}

def is_market_active():
    now  = datetime.now(IST)
    wday = now.weekday()
    hour = now.hour
    if wday == 5: return False, "Saturday — forex closed"
    if wday == 6 and hour < 19: return False, "Sunday — not open yet"
    if 0 <= hour < 6: return False, "00:00–06:00 IST dead hours"
    return True, "ok"

def is_asset_active(name):
    """WTI futures close 01:00-02:30 IST daily — skip that window."""
    now  = datetime.now(IST)
    hour, minu = now.hour, now.minute
    if name in OIL_ASSETS and hour == 1: return False
    if name in OIL_ASSETS and hour == 2 and minu < 30: return False
    return True

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
    dx  = 100 * (pdi-mdi).abs() / (pdi+mdi).replace(0, np.nan)
    return dx.ewm(span=p, adjust=False).mean()

def calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(span=p, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=p, adjust=False).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def calc_ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

# ── Analysis ──────────────────────────────────────────────────────────────────
def analyze(name, ticker):
    try:
        df = yf.download(ticker, period="3d", interval="1m",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 60:
            return None
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                      for c in df.columns]
    except Exception as e:
        print(f"    fetch error: {e}")
        return None

    df["sma50"] = df["close"].rolling(50).mean()
    df["ema9"]  = calc_ema(df["close"], 9)
    df["ema21"] = calc_ema(df["close"], 21)
    df["adx"]   = calc_adx(df)
    df["rsi"]   = calc_rsi(df["close"])

    r, prev = df.iloc[-1], df.iloc[-2]
    price     = float(r["close"])
    sma50     = float(r["sma50"])
    adx_val   = float(r["adx"])
    rsi_val   = float(r["rsi"])
    sma_slope = float(r["sma50"]) - float(prev["sma50"])
    candle_up = float(r["close"]) > float(prev["close"])
    ema9_val  = float(r["ema9"])
    ema21_val = float(r["ema21"])

    if adx_val < 22: return None
    if 44 < rsi_val < 56: return None

    bull = (price > sma50 and sma_slope > 0
            and ema9_val > ema21_val and rsi_val > 56 and candle_up)
    bear = (price < sma50 and sma_slope < 0
            and ema9_val < ema21_val and rsi_val < 44 and not candle_up)

    if not bull and not bear:
        return None

    signal = "UP" if bull else "DOWN"
    conf   = 50
    conf  += min(20, int((adx_val - 22) * 0.9))
    conf  += min(15, int(abs(rsi_val - 50) * 0.5))
    conf  += min(10, int(abs(ema9_val - ema21_val) / price * 20000))
    conf   = min(conf, 95)
    if conf < 65: return None

    dec = 2 if "BTC" in name else 5
    return {
        "asset": name, "signal": signal,
        "price": round(price, dec), "sma50": round(sma50, dec),
        "adx": round(adx_val, 1), "rsi": round(rsi_val, 1),
        "confidence": conf,
    }

# ── Telegram ──────────────────────────────────────────────────────────────────
def send_signals(signals):
    if not signals:
        return
    trade_at = next_candle_time()
    scan_at  = ist_now()

    lines = [
        "🎯 <b>SMA-50 SIGNAL — Crypto Knight</b>", "",
        f"⏰ <b>Scanned at   :</b> <code>{scan_at}</code>",
        f"🕐 <b>Trade candle :</b> <code>{trade_at}</code>  ← open trade here",
        f"⏱️ <b>Expiry       :</b> <code>1 MINUTE</code>",
        "──────────────────────", "",
    ]
    for s in signals:
        em, act = ("🟢", "CALL  ▲  (UP)") if s["signal"] == "UP" else ("🔴", "PUT   ▼  (DOWN)")
        lines += [
            f"{em} <b>{s['asset']}</b>",
            f"   Signal     : <b>{act}</b>",
            f"   Price      : <code>{s['price']}</code>",
            f"   SMA-50     : <code>{s['sma50']}</code>",
            f"   ADX        : <code>{s['adx']}</code>",
            f"   RSI        : <code>{s['rsi']}</code>",
            f"   Confidence : <code>{s['confidence']}%</code>",
            "",
        ]
    lines += ["──────────────────────", "⚠️ <i>Paper trading only.</i>"]

    try:
        r = httpx.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": "\n".join(lines), "parse_mode": "HTML"},
            timeout=15,
        )
        print(f"  Telegram: {r.status_code}")
        if r.status_code == 403:
            print("  ❌ 403 Forbidden — bot not started or wrong Chat ID")
            print("  👉 Open this in browser to find your real Chat ID:")
            print(f"     https://api.telegram.org/bot{BOT_TOKEN}/getUpdates")
            print("  👉 Then send /start to your bot and try again")
            return
        r.raise_for_status()
        print(f"  ✅ Sent {len(signals)} signal(s)")
    except httpx.HTTPStatusError as e:
        print(f"  ❌ Telegram error: {e} — scanner continues")
    except Exception as e:
        print(f"  ❌ Network error: {e} — scanner continues")

def heartbeat_if_needed():
    if datetime.now(IST).minute > 4:
        return
    httpx.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        json={
            "chat_id": CHAT_ID,
            "text": f"🤖 <b>Crypto Knight Scanner</b> — alive\n⏰ <code>{ist_now()}</code>\n<i>Watching 6 pairs every 60s</i>",
            "parse_mode": "HTML",
        },
        timeout=15,
    )

# ── Main loop — 4 scans × 60s inside one GitHub Actions job ──────────────────
def main():
    print(f"[{ist_now()}] Job started")

    active, reason = is_market_active()
    if not active:
        print(f"⏸  {reason} — exiting")
        return

    JOB_DURATION  = 4 * 60   # 4 minutes
    SCAN_INTERVAL = 60
    job_start     = time.time()
    iteration     = 0

    while time.time() - job_start < JOB_DURATION:
        iteration += 1
        scan_start = time.time()
        print(f"\n── Scan #{iteration}  [{ist_now()}] ──")

        signals = []
        for name, ticker in ASSETS.items():
            key = f"{name}-{datetime.now(IST).strftime('%H:%M')}"
            if key in sent_this_run:
                print(f"  {name}: already sent this minute")
                continue
            if not is_asset_active(name):
                print(f"  {name}: market closed right now")
                continue
            result = analyze(name, ticker)
            if result:
                print(f"  {name}: {result['signal']} conf={result['confidence']}%")
                signals.append(result)
                sent_this_run.add(key)
            else:
                print(f"  {name}: no signal")

        send_signals(signals)
        heartbeat_if_needed()

        elapsed   = time.time() - scan_start
        sleep_for = max(0, SCAN_INTERVAL - elapsed)
        remaining = JOB_DURATION - (time.time() - job_start)
        if remaining <= sleep_for + 5:
            break
        print(f"  Sleeping {sleep_for:.0f}s...")
        time.sleep(sleep_for)

    print(f"\n[{ist_now()}] Done — {iteration} scans")

if __name__ == "__main__":
    main()
    
