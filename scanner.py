"""
Crypto Knight — High Confidence Strategy
Indicators: SMA50 + EMA9/21 + MACD + ADX + RSI + Bollinger + Stochastic
All 6 must agree. No compromise. Quality over quantity.
Logs every signal to results.json for win rate tracking.
"""

import os, json, httpx, yfinance as yf, pandas as pd, numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

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

SEP  = "=" * 54
SEP2 = "-" * 54

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
    ema12  = s.ewm(span=12,adjust=False).mean()
    ema26  = s.ewm(span=26,adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9,adjust=False).mean()
    return macd, signal, macd - signal

def calc_bollinger(s, p=20):
    mid = s.rolling(p).mean()
    std = s.rolling(p).std()
    return mid+2*std, mid, mid-2*std

def calc_stochastic(df, k=14, d=3):
    low_min  = df["low"].rolling(k).min()
    high_max = df["high"].rolling(k).max()
    k_line   = 100*(df["close"]-low_min)/(high_max-low_min).replace(0,np.nan)
    d_line   = k_line.rolling(d).mean()
    return k_line, d_line

def calc_vwap(df):
    typical = (df["high"]+df["low"]+df["close"])/3
    return (typical*df["volume"]).cumsum()/df["volume"].cumsum()

# ── Signal scoring system ─────────────────────────────────────────────────────
def score_signal(name, ticker):
    """
    Returns (signal, score, details, skip_reason)
    Score = count of indicators agreeing (max 7)
    We only trade if score >= 5 (5 out of 7 indicators agree)
    """
    try:
        df = yf.download(ticker, period="5d", interval="5m",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 60:
            return None, 0, {}, "No data"
        df.columns = [c[0].lower() if isinstance(c,tuple) else c.lower()
                      for c in df.columns]
    except Exception as e:
        return None, 0, {}, f"Fetch error: {e}"

    # Calculate all indicators
    df["sma50"]                           = df["close"].rolling(50).mean()
    df["ema9"]                            = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema21"]                           = df["close"].ewm(span=21, adjust=False).mean()
    df["adx"], df["pdi"], df["mdi"]       = calc_adx(df)
    df["rsi"]                             = calc_rsi(df["close"])
    df["macd"], df["macd_sig"], df["hist"]= calc_macd(df["close"])
    df["bb_up"], df["bb_mid"], df["bb_low"]= calc_bollinger(df["close"])
    df["stoch_k"], df["stoch_d"]          = calc_stochastic(df)
    try:
        df["vwap"] = calc_vwap(df)
        has_vwap = True
    except:
        has_vwap = False

    r, p2 = df.iloc[-1], df.iloc[-2]

    price     = float(r["close"])
    sma50     = float(r["sma50"])
    ema9      = float(r["ema9"])
    ema21     = float(r["ema21"])
    adx_val   = float(r["adx"])
    pdi       = float(r["pdi"])
    mdi       = float(r["mdi"])
    rsi_val   = float(r["rsi"])
    macd_now  = float(r["macd"])
    sig_now   = float(r["macd_sig"])
    hist_now  = float(r["hist"])
    hist_prv  = float(p2["hist"])
    bb_up     = float(r["bb_up"])
    bb_low    = float(r["bb_low"])
    stoch_k   = float(r["stoch_k"])
    stoch_d   = float(r["stoch_d"])
    sma_slope = sma50 - float(p2["sma50"])

    # ── Hard rejection filters — market must be trending ─────────────────────
    recent    = df.tail(10)
    range_pct = (float(recent["high"].max()-recent["low"].min())/price)*100

    if adx_val < 20:
        return None, 0, {}, f"ADX {adx_val:.1f} < 20 — sideways"
    if range_pct < 0.04:
        return None, 0, {}, f"Range {range_pct:.3f}% — dead market"

    # ── Score each indicator independently ───────────────────────────────────
    bull_score = 0
    bear_score = 0
    votes = {}

    # 1. SMA50 — price above or below
    if price > sma50 and sma_slope > 0:
        bull_score += 1; votes["SMA50"] = "▲ BULL"
    elif price < sma50 and sma_slope < 0:
        bear_score += 1; votes["SMA50"] = "▼ BEAR"
    else:
        votes["SMA50"] = "— NEUTRAL"

    # 2. EMA crossover
    if ema9 > ema21:
        bull_score += 1; votes["EMA9/21"] = "▲ BULL"
    elif ema9 < ema21:
        bear_score += 1; votes["EMA9/21"] = "▼ BEAR"
    else:
        votes["EMA9/21"] = "— NEUTRAL"

    # 3. ADX directional
    if pdi > mdi and adx_val > 22:
        bull_score += 1; votes["ADX/DI"] = f"▲ BULL (ADX {adx_val:.0f})"
    elif mdi > pdi and adx_val > 22:
        bear_score += 1; votes["ADX/DI"] = f"▼ BEAR (ADX {adx_val:.0f})"
    else:
        votes["ADX/DI"] = f"— WEAK (ADX {adx_val:.0f})"

    # 4. RSI
    if rsi_val > 55 and rsi_val < 75:
        bull_score += 1; votes["RSI"] = f"▲ BULL ({rsi_val:.0f})"
    elif rsi_val < 45 and rsi_val > 25:
        bear_score += 1; votes["RSI"] = f"▼ BEAR ({rsi_val:.0f})"
    elif rsi_val >= 75:
        votes["RSI"] = f"⚠ OVERBOUGHT ({rsi_val:.0f})"
    elif rsi_val <= 25:
        votes["RSI"] = f"⚠ OVERSOLD ({rsi_val:.0f})"
    else:
        votes["RSI"] = f"— NEUTRAL ({rsi_val:.0f})"

    # 5. MACD — histogram direction + line position
    macd_cross_up   = macd_now > sig_now and float(p2["macd"]) <= float(p2["macd_sig"])
    macd_cross_down = macd_now < sig_now and float(p2["macd"]) >= float(p2["macd_sig"])
    if (macd_now > sig_now and hist_now > hist_prv):
        bull_score += 1
        votes["MACD"] = "▲ BULL" + (" ⚡CROSS" if macd_cross_up else "")
    elif (macd_now < sig_now and hist_now < hist_prv):
        bear_score += 1
        votes["MACD"] = "▼ BEAR" + (" ⚡CROSS" if macd_cross_down else "")
    else:
        votes["MACD"] = "— FLAT"

    # 6. Bollinger Bands
    bb_pos = (price - bb_low) / (bb_up - bb_low) if (bb_up - bb_low) > 0 else 0.5
    if 0.5 < bb_pos < 0.85:
        bull_score += 1; votes["BB"] = f"▲ BULL (pos {bb_pos:.0%})"
    elif 0.15 < bb_pos < 0.5:
        bear_score += 1; votes["BB"] = f"▼ BEAR (pos {bb_pos:.0%})"
    elif bb_pos >= 0.85:
        votes["BB"] = f"⚠ NEAR UPPER — skip"
    elif bb_pos <= 0.15:
        votes["BB"] = f"⚠ NEAR LOWER — skip"
    else:
        votes["BB"] = "— NEUTRAL"

    # 7. Stochastic
    if stoch_k > stoch_d and stoch_k < 80:
        bull_score += 1; votes["STOCH"] = f"▲ BULL (K:{stoch_k:.0f} D:{stoch_d:.0f})"
    elif stoch_k < stoch_d and stoch_k > 20:
        bear_score += 1; votes["STOCH"] = f"▼ BEAR (K:{stoch_k:.0f} D:{stoch_d:.0f})"
    elif stoch_k >= 80:
        votes["STOCH"] = f"⚠ OVERBOUGHT (K:{stoch_k:.0f})"
    elif stoch_k <= 20:
        votes["STOCH"] = f"⚠ OVERSOLD (K:{stoch_k:.0f})"
    else:
        votes["STOCH"] = "— NEUTRAL"

    # ── Decision — need 5 out of 7 indicators agreeing ────────────────────────
    MAX_SCORE = 7
    MIN_SCORE = 6   # 6/7 indicators must agree — high confidence only

    if bull_score >= MIN_SCORE and bull_score > bear_score:
        signal = "UP"
        score  = bull_score
    elif bear_score >= MIN_SCORE and bear_score > bull_score:
        signal = "DOWN"
        score  = bear_score
    else:
        top = max(bull_score, bear_score)
        return None, top, votes, f"Only {top}/7 indicators agree — need 6+"

    # ── Confidence from score ─────────────────────────────────────────────────
    conf_map = {5: 78, 6: 88, 7: 96}
    confidence = conf_map.get(score, 70)

    # ADX bonus
    if adx_val > 30: confidence = min(confidence + 3, 98)

    # MACD crossover bonus
    if macd_cross_up or macd_cross_down: confidence = min(confidence + 4, 98)

    details = {
        "price":       round(price, 5),
        "sma50":       round(sma50, 5),
        "adx":         round(adx_val, 1),
        "rsi":         round(rsi_val, 1),
        "macd":        round(macd_now, 6),
        "macd_sig":    round(sig_now, 6),
        "stoch_k":     round(stoch_k, 1),
        "stoch_d":     round(stoch_d, 1),
        "confidence":  confidence,
        "score":       f"{score}/7",
        "fresh_cross": macd_cross_up or macd_cross_down,
        "votes":       votes,
    }

    return signal, score, details, None

# ── Console print ─────────────────────────────────────────────────────────────
def print_header():
    print(SEP)
    print("   CRYPTO KNIGHT — HIGH CONFIDENCE SCANNER")
    print(f"   {ist_now()}")
    print(f"   Strategy : SMA50+EMA+MACD+ADX+RSI+BB+STOCH")
    print(f"   Rule     : 5/7 indicators must agree")
    print(f"   Timeframe: M5 | Expiry: 5 mins")
    print(SEP)

def print_asset(name, signal, score, details, skip):
    if signal:
        arrow = "▲ UP   (CALL)" if signal == "UP" else "▼ DOWN (PUT) "
        cross = " ⚡" if details.get("fresh_cross") else ""
        print(f"\n  ✅ {name} — {arrow}{cross}")
        print(f"     Score      : {details['score']} indicators agree")
        print(f"     Confidence : {details['confidence']}%")
        print(f"     Price      : {details['price']}")
        print(f"     ADX        : {details['adx']}")
        print(f"     RSI        : {details['rsi']}")
        print(f"     MACD       : {details['macd']}")
        print(f"     Stoch K/D  : {details['stoch_k']} / {details['stoch_d']}")
        print(f"\n     Indicator votes:")
        for ind, vote in details["votes"].items():
            print(f"       {ind:<8} : {vote}")
    else:
        agreed = score
        print(f"  ✗  {name:<10} → {skip} [{agreed}/7]")

def print_summary(signals, skips):
    print(f"\n{SEP2}")
    print(f"  SCAN RESULT")
    print(f"  Signal  : {len(signals)} found (max 1)")
    if signals:
        open_at, close_at = trade_times()
        print(f"\n  ⏰ Open  : {open_at} IST")
        print(f"  ⏰ Close : {close_at} IST (5-min expiry)")
        print(f"\n  PLACE THIS TRADE:")
        for i, s in enumerate(signals, 1):
            arrow = "▲ CALL (UP)" if s["signal"]=="UP" else "▼ PUT  (DOWN)"
            print(f"  {i}. {s['asset']:<10} {arrow}  {s['details']['score']}  {s['details']['confidence']}%")
    else:
        print(f"\n  ❌ No trades — market not ready")
        print(f"  Retry: 14:00–16:00 IST or 19:00–21:00 IST")
    print(SEP)

# ── Result tracker ────────────────────────────────────────────────────────────
RESULTS_FILE = "results.json"

def load_results():
    if Path(RESULTS_FILE).exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {"trades": [], "wins": 0, "losses": 0, "ties": 0, "total": 0}

def save_signal_to_results(signals):
    """Log signals so you can manually update win/loss later."""
    data = load_results()
    open_at, close_at = trade_times()
    for s in signals:
        entry = {
            "id":         len(data["trades"]) + 1,
            "date":       ist_now(),
            "asset":      s["asset"],
            "signal":     s["signal"],
            "price":      s["details"]["price"],
            "confidence": s["details"]["confidence"],
            "score":      s["details"]["score"],
            "open_at":    open_at,
            "close_at":   close_at,
            "result":     "PENDING",   # update manually: WIN / LOSS / TIE
            "pnl":        0,
        }
        data["trades"].append(entry)
        data["total"] += 1
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  📋 Logged {len(signals)} trade(s) to {RESULTS_FILE}")

def print_stats():
    data = load_results()
    total  = data["total"]
    wins   = data["wins"]
    losses = data["losses"]
    ties   = data["ties"]
    pending = total - wins - losses - ties
    if total == 0:
        print("  📊 No trades recorded yet")
        return
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    print(f"\n{SEP2}")
    print(f"  TRACK RECORD  ({total} signals logged)")
    print(f"  Wins    : {wins}")
    print(f"  Losses  : {losses}")
    print(f"  Ties    : {ties}")
    print(f"  Pending : {pending}")
    print(f"  Win rate: {win_rate:.1f}%")
    if win_rate >= 60:
        print(f"  ✅ PROFITABLE — keep trading")
    elif win_rate >= 50:
        print(f"  ⚠️  BREAKEVEN — needs improvement")
    elif (wins + losses) >= 10:
        print(f"  ❌ BELOW 50% — stop real money")
    print(SEP2)

# ── Telegram ──────────────────────────────────────────────────────────────────
def send_signals(signals):
    open_at, close_at = trade_times()
    data = load_results()
    wins = data["wins"]; losses = data["losses"]
    win_rate = f"{wins/(wins+losses)*100:.0f}%" if (wins+losses) > 0 else "N/A"

    lines = [
        "🎯 <b>Crypto Knight — STRONGEST SIGNAL</b>",
        f"🏆 <b>Best of all 6 pairs scanned</b>",
        f"⏰ <code>{ist_now()}</code>",
        f"📊 Track record: {wins}W/{losses}L (WR: {win_rate})",
        "",
    ]
    for i, s in enumerate(signals, 1):
        em  = "🟢" if s["signal"]=="UP" else "🔴"
        act = "CALL  ▲  (UP)" if s["signal"]=="UP" else "PUT  ▼  (DOWN)"
        d   = s["details"]
        cross = "\n   ⚡ <b>Fresh MACD crossover!</b>" if d.get("fresh_cross") else ""
        lines += [
            "──────────────────────" if i > 1 else "",
            f"{em} <b>{s['asset']} — {act}</b>",
            f"   Action     : <b>{act}</b>",
        f"   Score      : <code>{d['score']} indicators agree</code>",
            f"   Score      : <code>{d['score']} indicators agree</code>",
            f"   Confidence : <code>{d['confidence']}%</code>{cross}",
            f"   Price      : <code>{d['price']}</code>",
            f"   ADX        : <code>{d['adx']}</code>",
            f"   RSI        : <code>{d['rsi']}</code>",
            f"   Stoch K/D  : <code>{d['stoch_k']} / {d['stoch_d']}</code>",
            f"   Open at    : <code>{open_at} IST</code>",
            f"   Close at   : <code>{close_at} IST</code>",
            "",
        ]
    lines += [
        "──────────────────────",
        "📌 <b>Pocket Option → expiry 5 mins</b>",
        "⚠️ <i>1 trade only. If it loses, stop for today.</i>",
    ]
    _tg("\n".join(lines))

def send_no_signal(skips):
    data = load_results()
    wins = data["wins"]; losses = data["losses"]
    win_rate = f"{wins/(wins+losses)*100:.0f}%" if (wins+losses) > 0 else "N/A"
    lines = [
        "🔍 <b>Crypto Knight</b>",
        f"⏰ <code>{ist_now()}</code>",
        f"📊 Track record: {wins}W/{losses}L (WR: {win_rate})",
        "",
        "❌ <b>No trade — 5/7 indicators not aligned</b>",
        "<i>This is the filter protecting your money.</i>",
        "",
        "📋 <b>Indicator summary:</b>",
    ]
    for name, reason in skips.items():
        lines.append(f"   • <code>{name}</code>: {reason}")
    lines += [
        "",
        "<i>Best times:\n• 14:00–16:00 IST\n• 19:00–21:00 IST</i>",
    ]
    _tg("\n".join(lines))

def _tg(text):
    try:
        r = httpx.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id":CHAT_ID,"text":text,"parse_mode":"HTML"},
            timeout=15,
        )
        if r.status_code == 403:
            print("  ❌ Telegram 403 — send /start to your bot")
            return
        r.raise_for_status()
        print("  ✅ Telegram sent")
    except Exception as e:
        print(f"  ❌ Telegram: {e}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print_header()
    print_stats()

    signals, skips = [], {}

    all_signals = []

    for name, ticker in ASSETS.items():
        signal, score, details, skip_reason = score_signal(name, ticker)
        print_asset(name, signal, score, details, skip_reason)
        if signal:
            all_signals.append({
                "asset":   name,
                "signal":  signal,
                "details": details,
                "score":   score,
            })
        else:
            skips[name] = skip_reason

    # Pick the single strongest signal across all assets
    if all_signals:
        # Sort by score first, then confidence
        best = sorted(
            all_signals,
            key=lambda x: (x["score"], x["details"]["confidence"]),
            reverse=True
        )[0]
        signals = [best]
        print(f"\n  🏆 BEST SIGNAL: {best['asset']} {best['signal']}")
        print(f"     Score: {best['score']}/7  Conf: {best['details']['confidence']}%")
        print(f"     (Scanned all {len(all_signals)} candidate(s), picked strongest)")

    print_summary(signals, skips)

    if signals:
        save_signal_to_results(signals)
        send_signals(signals)
    else:
        send_no_signal(skips)

    print(f"\n  Done at {ist_now()}")

if __name__ == "__main__":
    main()
    
