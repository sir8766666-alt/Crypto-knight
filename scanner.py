"""
SMA-50 Manual Session Scanner — Crypto Knight
Triggered manually via GitHub Actions workflow_dispatch.
- Scans every 5 min for up to 30 min
- Only fires signals with confidence >= 90%
- Stops after 3 trades sent
- 5-min expiry — enough time to open app and place trade
"""

import os, time, httpx, yfinance as yf, pandas as pd, numpy as np
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

def ist_now(dt=None):
    d = dt or datetime.now(IST)
    return d.strftime("%d %b %Y  %I:%M %p IST")

def trade_candle_time():
    """5-min expiry — tells trader exactly when candle closes."""
    now = datetime.now(IST)
    nxt = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    exp = nxt + timedelta(minutes=5)
    return nxt.strftime("%I:%M %p"), exp.strftime("%I:%M %p")

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    raise EnvironmentError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in GitHub Secrets")

ASSETS = {
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "EUR/JPY": "EURJPY=X",
    "AUD/CAD": "AUDCAD=X",
    "WTI/OIL": "CL=F",
}

OIL_ASSETS = {"WTI/OIL"}

def is_asset_active(name):
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

# ── Analysis — 90%+ confidence only ──────────────────────────────────────────
def analyze(name, ticker):
    try:
        df = yf.download(ticker, period="3d", interval="1m",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 60: return None
        df.columns = [c[0].lower() if isinstance(c,tuple) else c.lower()
                      for c in df.columns]
    except Exception as e:
        print(f"    fetch error: {e}")
        return None

    df["sma50"] = df["close"].rolling(50).mean()
    df["ema9"]  = calc_ema(df["close"], 9)
    df["ema21"] = calc_ema(df["close"], 21)
    df["adx"]   = calc_adx(df)
    df["rsi"]   = calc_rsi(df["close"])

    # Use last 3 candles for stronger confirmation
    r, p2, p3 = df.iloc[-1], df.iloc[-2], df.iloc[-3]

    price     = float(r["close"])
    sma50     = float(r["sma50"])
    adx_val   = float(r["adx"])
    rsi_val   = float(r["rsi"])
    ema9_val  = float(r["ema9"])
    ema21_val = float(r["ema21"])
    sma_slope = float(r["sma50"]) - float(p2["sma50"])

    # Consecutive candles in same direction (stronger signal)
    candles_up   = float(r["close"]) > float(p2["close"]) > float(p3["close"])
    candles_down = float(r["close"]) < float(p2["close"]) < float(p3["close"])

    # ── Very strict filters for 90%+ confidence ───────────────────────────────
    if adx_val < 25: return None          # stronger trend required
    if 42 < rsi_val < 58: return None     # tighter dead zone

    bull = (price > sma50 and sma_slope > 0
            and ema9_val > ema21_val
            and rsi_val > 58 and candles_up)

    bear = (price < sma50 and sma_slope < 0
            and ema9_val < ema21_val
            and rsi_val < 42 and candles_down)

    if not bull and not bear: return None

    signal = "UP" if bull else "DOWN"

    # ── Confidence — needs all components strong to hit 90 ───────────────────
    conf = 50
    conf += min(22, int((adx_val - 25) * 1.1))       # ADX above 25
    conf += min(18, int(abs(rsi_val - 50) * 0.6))     # RSI extremity
    conf += min(12, int(abs(ema9_val-ema21_val)/price*25000))  # EMA gap
    conf += 8 if candles_up or candles_down else 0     # 3-candle streak bonus
    conf  = min(conf, 98)

    if conf < 90: return None             # hard cutoff — 90% minimum

    dec = 2 if "BTC" in name else 5
    return {
        "asset": name, "signal": signal,
        "price": round(price, dec),
        "sma50": round(sma50, dec),
        "adx":   round(adx_val, 1),
        "rsi":   round(rsi_val, 1),
        "confidence": conf,
    }

# ── Telegram ──────────────────────────────────────────────────────────────────
def send_signal(s, trade_num, total_allowed):
    open_at, close_at = trade_candle_time()
    em, act = ("🟢","CALL  ▲  (UP)") if s["signal"]=="UP" else ("🔴","PUT   ▼  (DOWN)")

    lines = [
        f"🎯 <b>TRADE {trade_num}/{total_allowed} — Crypto Knight</b>",
        "",
        f"{em} <b>{s['asset']}  —  {act}</b>",
        "",
        f"💰 Price      : <code>{s['price']}</code>",
        f"📈 SMA-50     : <code>{s['sma50']}</code>",
        f"⚡ ADX        : <code>{s['adx']}</code>",
        f"📊 RSI        : <code>{s['rsi']}</code>",
        f"🎯 Confidence : <code>{s['confidence']}%</code>",
        "",
        f"⏰ Open trade : <code>{open_at} IST</code>",
        f"⏱️ Close at   : <code>{close_at} IST</code>  (5-min expiry)",
        f"🕐 Scanned    : <code>{ist_now()}</code>",
        "",
        "──────────────────────",
        f"📌 <b>Open Pocket Option NOW → select {s['asset']} → place {act.split()[0]}</b>",
        "──────────────────────",
        "⚠️ <i>Paper trading. Max 3 trades this session.</i>",
    ]

    try:
        r = httpx.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id":CHAT_ID,"text":"\n".join(lines),"parse_mode":"HTML"},
            timeout=15,
        )
        print(f"  Telegram: {r.status_code}")
        if r.status_code == 403:
            print("  ❌ 403 — check Chat ID and that you started the bot")
            return False
        r.raise_for_status()
        print(f"  ✅ Signal sent: {s['asset']} {s['signal']} {s['confidence']}%")
        return True
    except Exception as e:
        print(f"  ❌ Telegram error: {e}")
        return False

def send_session_start(scan_end_ist):
    try:
        httpx.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={
                "chat_id": CHAT_ID,
                "text": (
                    f"🚀 <b>Crypto Knight — Session Started</b>\n\n"
                    f"⏰ Started  : <code>{ist_now()}</code>\n"
                    f"⏳ Scanning : <code>30 minutes</code>\n"
                    f"🔍 Strategy : <code>SMA-50 + ADX + RSI</code>\n"
                    f"🎯 Min conf : <code>{MIN_CONFIDENCE}</code>\n"
                    f"⏱️ Expiry   : <code>5 minutes</code>\n"
                    f"📊 Max trades: <code>3</code>\n\n"
                    f"<i>Will alert you instantly when signal found.\nSession ends {scan_end_ist} IST or after 3 trades.</i>"
                ),
                "parse_mode": "HTML",
            },
            timeout=15,
        )
    except Exception as e:
        print(f"  Session start ping failed: {e}")

def send_session_end(trades_sent):
    msg = (
        f"🏁 <b>Session Complete — Crypto Knight</b>\n\n"
        f"📊 Trades sent : <code>{trades_sent}/3</code>\n"
        f"⏰ Ended at    : <code>{ist_now()}</code>\n\n"
    )
    if trades_sent == 0:
        msg += "<i>No 90%+ signals found this session.\nTry during 09:00–11:00 or 13:30–15:30 IST for best results.</i>"
    elif trades_sent < 3:
        msg += f"<i>Only {trades_sent} high-confidence signal(s) found.\nQuality over quantity — good discipline.</i>"
    else:
        msg += "<i>3 trades placed. Step away and let them close.\nDo NOT place more trades.</i>"

    try:
        httpx.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id":CHAT_ID,"text":msg,"parse_mode":"HTML"},
            timeout=15,
        )
    except Exception as e:
        print(f"  Session end ping failed: {e}")

# ── Main — 30 min session, scan every 5 min, max 3 trades ────────────────────
def main():
    SESSION_DURATION = 15 * 60    # 15 minutes
    SCAN_INTERVAL    = 5  * 60    # scan every 5 minutes
    MAX_TRADES       = 3
    MIN_CONFIDENCE   = 70

    job_start   = time.time()
    trades_sent = 0
    scan_count  = 0
    sent_assets = set()           # don't double-signal same asset

    session_end_ist = (datetime.now(IST) + timedelta(minutes=30)).strftime("%I:%M %p")
    print(f"[{ist_now()}] Session started — max {MAX_TRADES} trades, {MIN_CONFIDENCE}%+ confidence")

    send_session_start(session_end_ist)

    while True:
        elapsed = time.time() - job_start

        if trades_sent >= MAX_TRADES:
            print(f"\n✅ {MAX_TRADES} trades sent — session complete")
            break
        if elapsed >= SESSION_DURATION:
            print(f"\n⏰ 30 min session ended")
            break

        scan_count += 1
        print(f"\n── Scan #{scan_count}  [{ist_now()}]  trades={trades_sent}/{MAX_TRADES} ──")

        for name, ticker in ASSETS.items():
            if trades_sent >= MAX_TRADES:
                break
            if name in sent_assets:
                continue
            if not is_asset_active(name):
                print(f"  {name}: market closed")
                continue

            result = analyze(name, ticker)
            if result:
                print(f"  {name}: {result['signal']} conf={result['confidence']}% ← SIGNAL!")
                ok = send_signal(result, trades_sent+1, MAX_TRADES)
                if ok:
                    trades_sent += 1
                    sent_assets.add(name)
            else:
                print(f"  {name}: no signal")

        if trades_sent >= MAX_TRADES:
            break

        # Sleep until next 5-min scan
        scan_elapsed = time.time() - job_start
        remaining    = SESSION_DURATION - scan_elapsed
        sleep_for    = min(SCAN_INTERVAL, remaining)
        if sleep_for <= 10:
            break
        print(f"\n  ⏳ Next scan in {int(sleep_for/60)}m {int(sleep_for%60)}s  |  {trades_sent}/{MAX_TRADES} trades  |  {int(remaining/60)}m left in session")
        time.sleep(sleep_for)

    send_session_end(trades_sent)
    print(f"\n[{ist_now()}] Done — {trades_sent} trades, {scan_count} scans")

if __name__ == "__main__":
    main()
            
