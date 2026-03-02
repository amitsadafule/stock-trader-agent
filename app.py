"""
ZERODHA MULTI-STOCK TRADING AGENT
===================================
Run:   python app.py
Open:  http://localhost:8080
"""

import csv, json, time, logging, datetime, threading, traceback, socket
from pathlib import Path
from zoneinfo import ZoneInfo

from flask import Flask, redirect, request, jsonify, render_template, Response
from kiteconnect import KiteConnect

try:
    from pywebpush import webpush, WebPushException
    PUSH_AVAILABLE = True
except ImportError:
    PUSH_AVAILABLE = False

from config import ZERODHA_CONFIG, TRADING_CONFIG, WATCHLIST, PATHS

# ── News sentiment (free RSS scraper — no API key needed) ─────────────────────
try:
    from news import get_news_sentiment, clear_cache as clear_news_cache
    NEWS_AVAILABLE = True
except ImportError:
    NEWS_AVAILABLE = False
    def get_news_sentiment(sym, **kw):
        return {"score": 0, "label": "NEUTRAL", "headlines": [],
                "conviction_delta": 0, "summary": "news.py not found", "error": ""}
    def clear_news_cache(sym=None): pass

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("agent.log")],
)
log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")
app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Global state ──────────────────────────────────────────────────────────────
# Per-stock state stored in stocks_state dict keyed by symbol
state = {
    "logged_in":            False,
    "agent_running":        False,
    "cash":                 "—",
    "last_checked":         "—",
    "last_error":           "",
    "market_open":          False,
    "opening_volatile":     False,   # True = blocked (buffer active, no override granted)
    "opening_early":        False,   # True = override granted, trading early despite buffer
    "opening_reason":       "",      # human-readable reason shown in UI
    "opening_wait_until":   "",      # e.g. "9:45 AM"
    "notifications_enabled": False,
    "dry_run":              TRADING_CONFIG.get("dry_run", True),
    "recovery":             None,
}
# Per-symbol live data updated each loop tick
stocks_state = {
    w["symbol"]: {
        "symbol":           w["symbol"],
        "exchange":         w["exchange"],
        "weight":           w["weight"],
        "signal":           "—",
        "price":            "—",
        "ema_green":        "—",
        "ema_red":          "—",
        "gap_pct":          "—",
        "gap_change":       "—",
        "trend_candles":    "—",
        "vol_ratio":        "—",
        "conviction":       "—",
        "conviction_label": "—",
        "invest_pct":       "—",
        "holdings_qty":     0,
        "avg_price":        "—",
        "news_score":       0,
        "news_label":       "—",
        "news_summary":     "—",
        "news_headlines":   [],
        "news_fetched_at":  "—",
        "error":            "",
    }
    for w in WATCHLIST
}

_stop_flag          = threading.Event()
_agent_thread       = None
_push_subscriptions = []
_buy_date_tracker   = {}   # symbol -> datetime.date
_instrument_cache   = {}   # exchange -> list of instruments (refreshed daily)
_news_lock          = threading.Lock()   # guards stocks_state news fields

kite = KiteConnect(api_key=ZERODHA_CONFIG["api_key"])
VAPID_PRIVATE_KEY = PATHS.get("vapid_private_key", "vapid_private.pem")
VAPID_PUBLIC_KEY  = PATHS.get("vapid_public_key",  "vapid_public.txt")
VAPID_CLAIMS      = {"sub": "mailto:agent@zerodha.local"}

# ── Transaction log ───────────────────────────────────────────────────────────
FIELDS = [
    "timestamp","action","symbol","quantity","price","total_value",
    "available_cash_before","available_cash_after","holdings_before",
    "holdings_after","reason","ema_green","ema_red","holding_days","estimated_tax_note"
]

def init_log():
    p = Path(PATHS["transaction_log"])
    if not p.exists():
        with open(p, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

def write_transaction(data: dict):
    with open(PATHS["transaction_log"], "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow({k: data.get(k, "") for k in FIELDS})
    action = data.get("action", "")
    sym    = data.get("symbol", "")
    send_push(
        title=f"{'🟢 BUY' if 'BUY' in action else '🔴 SELL'} {sym}",
        body=f"{data.get('quantity','')} shares @ ₹{data.get('price','')} | {str(data.get('reason',''))[:80]}",
        tag="trade"
    )

def read_transactions(limit=50):
    p = Path(PATHS["transaction_log"])
    if not p.exists():
        return []
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-limit:][::-1]

# ── Push notifications ────────────────────────────────────────────────────────
def send_push(title: str, body: str, tag: str = "alert", url: str = "/"):
    if not PUSH_AVAILABLE or not _push_subscriptions:
        return
    if not Path(VAPID_PRIVATE_KEY).exists():
        return
    payload = json.dumps({"title": title, "body": body, "tag": tag, "url": url})
    for sub in list(_push_subscriptions):
        try:
            webpush(
                subscription_info=sub, data=payload,
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims={**VAPID_CLAIMS,
                    "aud": "/".join(sub["endpoint"].split("/")[:3]) if sub.get("endpoint") else "https://fcm.googleapis.com"},
            )
        except WebPushException as e:
            log.warning(f"Push failed: {e}")
            if "410" in str(e) or "404" in str(e):
                try: _push_subscriptions.remove(sub)
                except ValueError: pass
        except Exception as e:
            log.warning(f"Push error: {e}")

# ── Auth ──────────────────────────────────────────────────────────────────────
def try_load_saved_token() -> bool:
    p = Path(PATHS["token_file"])
    if p.exists():
        token = p.read_text().strip()
        try:
            kite.set_access_token(token)
            kite.margins(segment="equity")
            ZERODHA_CONFIG["access_token"] = token
            state["logged_in"] = True
            return True
        except Exception:
            p.unlink(missing_ok=True)
            state["logged_in"] = False
    return False

def schedule_daily_login_reminder():
    while True:
        now    = datetime.datetime.now(IST)
        target = now.replace(hour=8, minute=45, second=0, microsecond=0)
        if now >= target:
            target += datetime.timedelta(days=1)
        time.sleep((target - now).total_seconds())
        if not state["logged_in"] or not Path(PATHS["token_file"]).exists():
            send_push("🔑 Zerodha Login Needed", "Market opens at 9:15 AM. Tap to login.", tag="login", url="/")
        elif not try_load_saved_token():
            send_push("🔑 Session Expired", "Tap to re-login before market opens.", tag="login", url="/")

# ── Market data helpers ───────────────────────────────────────────────────────
def get_instruments(exchange: str) -> list:
    today = datetime.date.today()
    cache_key = f"{exchange}_{today}"
    if cache_key not in _instrument_cache:
        _instrument_cache.clear()
        _instrument_cache[cache_key] = kite.instruments(exchange)
    return _instrument_cache[cache_key]

def get_instrument_token(symbol: str, exchange: str) -> int:
    insts = get_instruments(exchange)
    tok = next((i["instrument_token"] for i in insts if i["tradingsymbol"] == symbol), None)
    if tok is None:
        raise ValueError(f"Instrument not found: {exchange}:{symbol}")
    return tok

MAX_DAYS = {
    "minute": 60, "3minute": 60, "5minute": 60,
    "10minute": 60, "15minute": 60, "30minute": 60,
    "60minute": 400, "day": 2000,
}

def get_candles(symbol: str, exchange: str) -> list:
    cfg       = TRADING_CONFIG
    interval  = cfg["candle_interval"]
    safe_days = min(MAX_DAYS.get(interval, 60) - 2, 58)
    to_dt     = datetime.datetime.now()
    fr_dt     = to_dt - datetime.timedelta(days=safe_days)
    tok       = get_instrument_token(symbol, exchange)
    data      = kite.historical_data(tok, fr_dt, to_dt, interval)
    rows = [{"timestamp": r["date"], "open": r["open"], "high": r["high"],
             "low": r["low"], "close": r["close"], "volume": r.get("volume", 0)}
            for r in data]
    rows.sort(key=lambda r: r["timestamp"])
    return rows[-cfg["lookback_candles"]:]

def calc_ema(prices: list, period: int) -> list:
    k = 2.0 / (period + 1)
    ema = [prices[0]]
    for price in prices[1:]:
        ema.append(price * k + ema[-1] * (1 - k))
    return ema

def analyse(candles: list) -> dict:
    """
    Full intelligence analysis of candle history.
    Returns a rich dict used for both signal decision AND position sizing.

    Signals:
      "BUY"      — EMA fast just crossed above EMA slow (fresh crossover)
      "SELL"     — EMA fast just crossed below EMA slow
      "HOLD(↑)"  — EMA fast already above slow (bullish, no new crossover)
      "HOLD(↓)"  — EMA fast already below slow (bearish)

    Intelligence fields:
      gap_pct        — % distance between EMAs (bigger = stronger trend)
      gap_change     — how gap changed vs previous candle (+ve = widening = accelerating)
      trend_candles  — how many consecutive candles EMA fast has been above/below slow
      vol_ratio      — current volume vs 10-candle average (>1.5 = high volume confirmation)
      conviction     — 0–100 score combining all above factors
      conviction_label — "STRONG" / "MODERATE" / "WEAK"
      invest_pct     — fraction of allocated cash to deploy (0.5–1.0)
                       STRONG=100%, MODERATE=75%, WEAK=50%

    Position sizing logic:
      A STRONG BUY uses 100% of its allocated cash.
      A WEAK BUY (tiny gap, shrinking, low volume) uses only 50% — keeps dry powder.
      A STRONG SELL exits the full position.
      A WEAK SELL exits only 25% — the signal may be false/temporary.
    """
    cfg   = TRADING_CONFIG
    close = [r["close"] for r in candles]
    vols  = [r.get("volume", 0) for r in candles]

    eg = calc_ema(close, cfg["ema_green_period"])
    er = calc_ema(close, cfg["ema_red_period"])

    pg, pr = eg[-2], er[-2]
    cg, cr = eg[-1], er[-1]
    gap    = abs(cg - cr) / cr * 100

    # ── Core signal ────────────────────────────────────────────────────────────
    if   pg <= pr and cg > cr: signal = "BUY"
    elif pg >= pr and cg < cr: signal = "SELL"
    else:                       signal = f"HOLD({'↑' if cg > cr else '↓'})"

    bullish = cg > cr

    # ── Gap momentum: is the separation growing or shrinking? ─────────────────
    prev_gap   = abs(pg - pr) / pr * 100
    gap_change = gap - prev_gap   # +ve = widening (accelerating trend)

    # ── Trend duration: consecutive candles in this EMA alignment ────────────
    trend_candles = 0
    for i in range(len(eg) - 1, -1, -1):
        if (eg[i] > er[i]) == bullish:
            trend_candles += 1
        else:
            break

    # ── Volume ratio: current vs 10-candle rolling average ────────────────────
    recent_v = [v for v in vols[-10:] if v > 0]
    avg_vol  = sum(recent_v) / len(recent_v) if recent_v else 1
    curr_vol = vols[-1] if vols[-1] > 0 else avg_vol
    vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0

    # ── Conviction score (0–100) ───────────────────────────────────────────────
    # Four independent factors, each scored and summed.
    score = 0
    parts = []

    # 1. Gap size (0–35 pts) — the primary "steepness" measure you asked about
    #    0.2% gap→7pts  0.5%→17pts  1.0%→28pts  2%+→35pts
    gap_score = min(35, int(gap * 17.5))
    score += gap_score
    parts.append(f"gap {gap:.2f}%→{gap_score}pt")

    # 2. Momentum (±25 pts) — widening gap is strong, narrowing is a warning
    if gap_change > 0:
        mom = min(25, int(gap_change * 50))
        score += mom
        parts.append(f"↑momentum +{gap_change:.3f}%→+{mom}pt")
    else:
        pen = max(-15, int(gap_change * 20))
        score += pen
        parts.append(f"↓fading {gap_change:.3f}%→{pen}pt")

    # 3. Volume confirmation (0–25 pts)
    #    1x avg→0pts  1.5x→6pts  2x→12pts  3x+→25pts
    vol_pts = min(25, int((vol_ratio - 1.0) * 12.5)) if vol_ratio > 1.0 else 0
    score += vol_pts
    parts.append(f"vol {vol_ratio:.1f}x→{vol_pts}pt")

    # 4. Trend age (0–15 pts) — established trends score higher
    #    1 candle→1pt  5→4pt  10→8pt  15+→12pt  but 30+ gets slight penalty (overextended)
    if trend_candles <= 20:
        age_pts = min(15, int(trend_candles * 0.75))
    else:
        age_pts = max(5, 15 - int((trend_candles - 20) * 0.5))
    score += age_pts
    parts.append(f"trend {trend_candles}c→{age_pts}pt")

    score = max(0, min(100, score))

    # ── Conviction label and invest fraction ──────────────────────────────────
    if score >= 65:
        label, invest_pct = "STRONG",   1.00
    elif score >= 38:
        label, invest_pct = "MODERATE", 0.75
    else:
        label, invest_pct = "WEAK",     0.50

    reason_detail = (
        f"Conviction {score}/100 ({label}) | "
        + " | ".join(parts)
        + f" | {trend_candles}c {'bull' if bullish else 'bear'}"
    )

    return {
        "signal":           signal,
        "ema_green":        cg,
        "ema_red":          cr,
        "gap_pct":          gap,
        "gap_change":       gap_change,
        "trend_candles":    trend_candles,
        "vol_ratio":        vol_ratio,
        "conviction":       score,
        "conviction_label": label,
        "invest_pct":       invest_pct,
        "reason_detail":    reason_detail,
    }


def get_signal(candles: list):
    """Backward-compat wrapper. Use analyse() for full intelligence."""
    a = analyse(candles)
    return a["signal"], a["ema_green"], a["ema_red"], a["gap_pct"]

# ── Background news fetcher ───────────────────────────────────────────────────
def _news_worker():
    """
    Runs in a daemon thread. Refreshes news sentiment for all watchlist stocks
    every 30 minutes. News is cached in news.py and only refetched when stale.
    Runs independently of the trading loop — never blocks trade execution.
    """
    log.info("📰 News worker started")
    while True:
        for w in WATCHLIST:
            sym = w["symbol"]
            try:
                result = get_news_sentiment(sym, max_age_hours=24)
                with _news_lock:
                    ss = stocks_state.get(sym)
                    if ss:
                        ss["news_score"]      = result["score"]
                        ss["news_label"]      = result["label"]
                        ss["news_summary"]    = result["summary"]
                        ss["news_headlines"]  = result["headlines"]
                        ss["news_fetched_at"] = result["fetched_at"]
            except Exception as e:
                log.warning(f"📰 {sym} news error: {e}")
            time.sleep(3)   # small gap between stocks to avoid rate limiting
        time.sleep(1800)    # refresh every 30 minutes


def apply_news_to_conviction(intel: dict, symbol: str) -> dict:
    """
    Reads cached news sentiment for symbol and adjusts conviction score.
    Returns a copy of intel with updated conviction, conviction_label,
    invest_pct and reason_detail.

    News contributes up to ±20 pts:
      BULLISH news + BUY/HOLD↑  → +5 to +20 pts (confirms signal)
      BULLISH news + SELL       → +0 pts (don't add conviction to a sell)
      BEARISH news + SELL/HOLD↓ → -5 to -20 pts (confirms bearish signal)
      BEARISH news + BUY        → -5 to -20 pts (news contradicts EMA signal)
      NEUTRAL / no news         → 0 pts
    """
    ss = stocks_state.get(symbol, {})
    news_label = ss.get("news_label", "NEUTRAL")
    news_score = ss.get("news_score", 0)
    news_summary = ss.get("news_summary", "")

    sig = intel.get("signal", "")
    is_bullish_signal = "BUY" in sig or "↑" in sig
    is_bearish_signal = "SELL" in sig or "↓" in sig

    delta = 0
    news_note = ""

    if news_label == "BULLISH":
        if is_bullish_signal:
            # News confirms EMA bullish signal — boost conviction
            delta = ss.get("news_conviction_delta", 0) or (
                20 if news_score >= 60 else 10 if news_score >= 30 else 5
            )
            news_note = f"📰 News BULLISH ({news_score:+d}) confirms signal → +{delta}pt"
        elif is_bearish_signal:
            # News is bullish but EMA says sell — reduce conviction of sell
            delta = -(ss.get("news_conviction_delta", 0) or (
                10 if news_score >= 60 else 5 if news_score >= 30 else 0
            ))
            news_note = f"📰 News BULLISH ({news_score:+d}) conflicts with SELL → {delta}pt"
    elif news_label == "BEARISH":
        if is_bearish_signal:
            # News confirms EMA bearish signal — boost SELL conviction (actually hurts buy)
            delta = -(ss.get("news_conviction_delta", 0) or (
                20 if news_score <= -60 else 10 if news_score <= -30 else 5
            ))
            news_note = f"📰 News BEARISH ({news_score:+d}) confirms bearish signal → {delta}pt"
        elif is_bullish_signal:
            # News is bearish but EMA says buy — reduce conviction
            delta = -(ss.get("news_conviction_delta", 0) or (
                15 if news_score <= -60 else 8 if news_score <= -30 else 3
            ))
            news_note = f"📰 News BEARISH ({news_score:+d}) conflicts with BUY → {delta}pt"

    if delta == 0:
        news_note = f"📰 News NEUTRAL — no conviction impact"

    new_score = max(0, min(100, intel["conviction"] + delta))

    # Recalculate label and invest_pct
    if new_score >= 65:
        new_label, new_invest = "STRONG", 1.00
    elif new_score >= 38:
        new_label, new_invest = "MODERATE", 0.75
    else:
        new_label, new_invest = "WEAK", 0.50

    updated = dict(intel)
    updated["conviction"]       = new_score
    updated["conviction_label"] = new_label
    updated["invest_pct"]       = new_invest
    updated["news_delta"]       = delta
    updated["news_note"]        = news_note
    updated["reason_detail"]    = intel["reason_detail"] + f" | {news_note}"

    if delta != 0:
        log.info(
            f"📰 {symbol}: news adjusted conviction "
            f"{intel['conviction']}→{new_score} ({intel['conviction_label']}→{new_label}) | {news_note}"
        )

    return updated


def is_market_open() -> bool:
    now = datetime.datetime.now(IST)
    if now.weekday() >= 5:
        return False
    o = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    c = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return o <= now <= c

def opening_volatility_status(intel: dict = None) -> dict:
    """
    Smart opening volatility guard — protects against false signals at open
    WITHOUT blocking genuine strong breakouts.

    Problem with a flat time buffer:
      If SILVERIETF rallies strongly from 9:15 and keeps going, a hard 30-min
      block means missing the entire move — that is also a loss (opportunity cost).

    Solution — signal-aware early override:
      During the buffer window, a signal CAN be acted on early if ALL four
      "strong breakout" criteria are met. Otherwise stay blocked.

    BLOCK (default during buffer):
      - Candle range is high/whipsawing (chaotic open)
      - Gap is small or narrowing (weak/fading signal)
      - Volume is below average (thin, no real participation)
      - Conviction is WEAK or MODERATE (uncertain signal)

    ALLOW EARLY (override buffer) only if ALL of:
      1. Conviction is STRONG (score >= 65)
      2. Gap is widening — trend accelerating, not reversing
      3. Volume ratio >= 1.5x — real participation confirms the move
      4. Candle range is NOT excessively wild (not a spike)

    After the buffer, only keeps blocking if candle range is still high.

    Returns:
      blocked       — True = do not execute orders this tick
      early_entry   — True = buffer overridden due to strong breakout signal
      volatile      — True = candle range still wild (post-buffer block)
      reason        — plain-English explanation for UI
      candle_range_pct — measured opening candle swing %
    """
    cfg  = TRADING_CONFIG
    now  = datetime.datetime.now(IST)
    buf  = cfg.get("opening_buffer_minutes", 30)
    vola = cfg.get("opening_volatility_threshold_pct", 1.5)

    # Thresholds for early-entry override
    EARLY_MIN_CONVICTION = 65    # must be STRONG
    EARLY_MIN_VOL_RATIO  = 1.5   # at least 1.5x average volume
    EARLY_MAX_RANGE_PCT  = 2.0   # candle range must NOT exceed this (not a spike)

    market_open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    buffer_end       = market_open_time + datetime.timedelta(minutes=buf)
    minutes_left     = max(0, int((buffer_end - now).total_seconds() / 60))
    in_buffer        = now < buffer_end

    def _get_opening_candle_range():
        """Fetch first 3 five-min candles of today and return high-low range %."""
        try:
            tok = None
            for w in WATCHLIST[:1]:
                try:
                    tok = get_instrument_token(w["symbol"], w["exchange"])
                    break
                except Exception:
                    pass
            if not tok:
                return None
            today_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
            raw = kite.historical_data(tok, today_start, now, "5minute")
            if not raw or len(raw) < 1:
                return None
            first = raw[:3]
            high  = max(c["high"] for c in first)
            low   = min(c["low"]  for c in first)
            open_ = first[0]["open"]
            return (high - low) / open_ * 100 if open_ else None
        except Exception as e:
            log.debug(f"Opening candle range check failed: {e}")
            return None

    candle_range_pct = _get_opening_candle_range()

    # ── PAST buffer: only block if still genuinely wild ──────────────────────
    if not in_buffer:
        still_volatile = (
            candle_range_pct is not None and candle_range_pct > vola
        )
        if still_volatile:
            return {
                "blocked": True, "early_entry": False, "volatile": True,
                "wait_until": (now + datetime.timedelta(minutes=10)).strftime("%I:%M %p"),
                "minutes_left": 10,
                "reason": (
                    f"Opening range {candle_range_pct:.1f}% > {vola}% threshold — "
                    f"market still choppy, re-checking in 10 min"
                ),
                "candle_range_pct": round(candle_range_pct, 2),
            }
        return {
            "blocked": False, "early_entry": False, "volatile": False,
            "wait_until": None, "minutes_left": 0,
            "reason": "Opening period clear — normal trading active",
            "candle_range_pct": round(candle_range_pct, 2) if candle_range_pct else None,
        }

    # ── INSIDE buffer: check for strong breakout override ────────────────────
    if intel is not None:
        conviction   = intel.get("conviction", 0)
        conv_label   = intel.get("conviction_label", "WEAK")
        gap_change   = intel.get("gap_change", 0)
        vol_ratio    = intel.get("vol_ratio", 0)
        signal       = intel.get("signal", "")
        is_bullish   = signal in ("BUY",) or (signal.startswith("HOLD") and "↑" in signal)

        candle_ok    = candle_range_pct is None or candle_range_pct <= EARLY_MAX_RANGE_PCT

        if (is_bullish
                and conviction >= EARLY_MIN_CONVICTION
                and gap_change > 0
                and vol_ratio >= EARLY_MIN_VOL_RATIO
                and candle_ok):
            # ALL criteria met — override the buffer, let this trade through
            log.info(
                f"🚀 Opening early-entry override: conviction={conviction}/100 "
                f"gap_change=+{gap_change:.3f}% vol={vol_ratio:.1f}x "
                f"range={f'{candle_range_pct:.1f}%' if candle_range_pct else 'n/a'} "
                f"— strong breakout, ignoring buffer"
            )
            return {
                "blocked": False, "early_entry": True, "volatile": False,
                "wait_until": None, "minutes_left": 0,
                "reason": (
                    f"🚀 Strong breakout override — conviction {conviction}/100, "
                    f"gap widening +{gap_change:.3f}%, vol {vol_ratio:.1f}x — "
                    f"buying despite opening buffer ({minutes_left}min remaining)"
                ),
                "candle_range_pct": round(candle_range_pct, 2) if candle_range_pct else None,
            }

        # Build a specific reason for why early entry was NOT granted
        reasons = []
        if not is_bullish:
            reasons.append("signal not bullish")
        if conviction < EARLY_MIN_CONVICTION:
            reasons.append(f"conviction {conviction}/100 < {EARLY_MIN_CONVICTION} needed")
        if gap_change <= 0:
            reasons.append(f"gap {'narrowing' if gap_change < 0 else 'flat'} (not accelerating)")
        if vol_ratio < EARLY_MIN_VOL_RATIO:
            reasons.append(f"volume {vol_ratio:.1f}x < {EARLY_MIN_VOL_RATIO}x needed")
        if not candle_ok:
            reasons.append(f"candle range {candle_range_pct:.1f}% too wide (spike risk)")

        log.info(
            f"⏸️  In buffer ({minutes_left}min left). Early entry denied: "
            + "; ".join(reasons)
        )

    return {
        "blocked": True, "early_entry": False, "volatile": False,
        "wait_until": buffer_end.strftime("%I:%M %p"),
        "minutes_left": minutes_left,
        "reason": (
            f"Opening buffer ({minutes_left}min left until {buffer_end.strftime('%I:%M %p')}) — "
            f"signal monitored, waiting for market to settle"
        ),
        "candle_range_pct": round(candle_range_pct, 2) if candle_range_pct else None,
    }

# ── State recovery ────────────────────────────────────────────────────────────
def recover_state() -> dict:
    summary = {
        "recovered": True, "offline_days": 0,
        "per_stock": {}, "warnings": [], "missed_signals": [],
    }
    rows = read_transactions(limit=500)
    for w in WATCHLIST:
        sym       = w["symbol"]
        sym_rows  = [r for r in rows if r.get("symbol") == sym and "[DRY]" not in r.get("action","")]
        info      = {"last_action": None, "last_action_time": None, "buy_date": None,
                     "log_holdings": 0, "actual_holdings": 0, "offline_days": 0}
        if sym_rows:
            last = sym_rows[0]
            info["last_action"]      = "BUY" if "BUY" in last["action"] else "SELL"
            info["last_action_time"] = last["timestamp"]
            info["log_holdings"]     = int(last.get("holdings_after") or 0)
            try:
                last_dt = datetime.datetime.fromisoformat(last["timestamp"])
                if last_dt.tzinfo is None: last_dt = last_dt.replace(tzinfo=IST)
                else: last_dt = last_dt.astimezone(IST)
                info["offline_days"] = (datetime.datetime.now(IST) - last_dt).days
                summary["offline_days"] = max(summary["offline_days"], info["offline_days"])
                if info["last_action"] == "BUY":
                    info["buy_date"] = last_dt.date()
                    _buy_date_tracker[sym] = last_dt.date()
            except Exception:
                pass
        summary["per_stock"][sym] = info

    # Actual holdings from Zerodha
    try:
        h_list = kite.holdings()
        for w in WATCHLIST:
            sym     = w["symbol"]
            holding = next((h for h in h_list if h["tradingsymbol"] == sym), None)
            actual  = holding["quantity"] if holding else 0
            summary["per_stock"][sym]["actual_holdings"] = actual
            logged  = summary["per_stock"][sym]["log_holdings"]
            if logged != actual:
                msg = f"{sym}: log={logged} shares, demat={actual}. Using demat."
                summary["warnings"].append(msg)
                log.warning(f"⚠️  {msg}")
            # Sync stocks_state
            stocks_state[sym]["holdings_qty"] = actual
            if holding:
                stocks_state[sym]["avg_price"] = f"₹{holding['average_price']:.2f}"
    except Exception as e:
        log.warning(f"Recovery: could not fetch holdings — {e}")

    log.info(f"🔄 Recovery done | offline {summary['offline_days']}d | stocks: {list(summary['per_stock'].keys())}")
    return summary

# ── Multi-stock allocation strategy ──────────────────────────────────────────
def allocate_cash(buy_signals: list, available_cash: float) -> dict:
    """
    Allocates investable cash using BOTH watchlist weights AND conviction scores.

    How it works:
    1. Split investable cash proportionally by watchlist weight (only among
       stocks that are actually signalling BUY right now).
    2. Scale each stock's slice by its invest_pct from conviction:
         STRONG (score 65+) → deploy 100% of its slice
         MODERATE (38–64)   → deploy 75% of its slice
         WEAK (<38)         → deploy only 50% of its slice
    3. Cash "saved" by WEAK/MODERATE signals stays as reserve — not redistributed.
       This means a WEAK signal never gets the full weight-based amount.
    4. Smart fallback: if low total cash means any stock can't meet min_trade_amount,
       drop the lowest-conviction stocks first until remaining can each be filled.

    Example with ₹10,000, SILVERIETF(w=40, STRONG) + GOLDETF(w=30, WEAK):
      Weight split:  SILVER=₹5,333  GOLD=₹4,000  (reserve ₹667 always held)
      Conviction:    SILVER×1.0=₹5,333  GOLD×0.5=₹2,000
      → SILVER gets ₹5,333, GOLD gets ₹2,000, ₹2,667 stays as dry powder
    """
    cfg        = TRADING_CONFIG
    strategy   = cfg.get("multi_buy_strategy", "weighted")
    reserve    = cfg.get("min_cash_reserve_pct", 0.10)
    min_order  = cfg.get("min_trade_amount", 200)
    investable = available_cash * (1 - reserve)

    if not buy_signals:
        return {}

    def raw_allocation(signals: list) -> dict:
        """Base allocation by strategy before conviction scaling."""
        alloc = {}
        if strategy == "top1" or len(signals) == 1:
            best = max(signals, key=lambda x: x.get("conviction", 50))
            alloc[best["symbol"]] = investable
        elif strategy == "equal":
            per = investable / len(signals)
            for s in signals: alloc[s["symbol"]] = per
        else:  # weighted
            total_w = sum(s["weight"] for s in signals)
            for s in signals:
                alloc[s["symbol"]] = investable * (s["weight"] / total_w)
        return alloc

    def apply_conviction(raw: dict, signals: list) -> dict:
        """Scale each allocation down by conviction invest_pct."""
        sig_map = {s["symbol"]: s for s in signals}
        return {
            sym: amt * sig_map[sym].get("invest_pct", 1.0)
            for sym, amt in raw.items()
        }

    def all_viable(alloc: dict) -> bool:
        return all(v >= min_order for v in alloc.values() if v > 0)

    # Initial allocation with conviction scaling
    allocation = apply_conviction(raw_allocation(buy_signals), buy_signals)

    # Smart fallback: if low cash, drop lowest-conviction stocks first
    if not all_viable(allocation) and len(buy_signals) > 1:
        ranked = sorted(buy_signals, key=lambda x: x.get("conviction", 0), reverse=True)
        for n in range(len(ranked), 0, -1):
            candidate = apply_conviction(raw_allocation(ranked[:n]), ranked[:n])
            if all_viable(candidate):
                allocation = candidate
                dropped = [s["symbol"] for s in ranked[n:]]
                if dropped:
                    log.info(
                        f"💡 Low cash: dropped {dropped} (lowest conviction), "
                        f"concentrating into "
                        f"{[(s['symbol'], s.get('conviction',0), s.get('conviction_label','?')) for s in ranked[:n]]}"
                    )
                break
        else:
            # Even single stock can't meet min — use everything on highest conviction
            best = max(buy_signals, key=lambda x: x.get("conviction", 0))
            allocation = {best["symbol"]: investable * best.get("invest_pct", 1.0)}
            log.info(
                f"💡 Cash too low to split — all to {best['symbol']} "
                f"conviction={best.get('conviction',0)}/100 ({best.get('conviction_label','?')})"
            )

    # Final log with conviction context per stock
    for s in buy_signals:
        sym  = s["symbol"]
        amt  = allocation.get(sym, 0)
        conv = s.get("conviction", "?")
        lbl  = s.get("conviction_label", "?")
        ipct = int(s.get("invest_pct", 1.0) * 100)
        log.info(
            f"💰 {sym}: ₹{amt:.0f} allocated | "
            f"conviction={conv}/100 ({lbl}) | deploying {ipct}% of weight-share"
        )
    return allocation

# ── Place buy/sell ────────────────────────────────────────────────────────────
def do_buy(symbol: str, exchange: str, cash_to_use: float, price: float,
           eg: float, er: float, gap: float, qty_held: int, catchup: bool = False,
           intel: dict = None):
    cfg = TRADING_CONFIG
    qty = int(cash_to_use // price)
    log.info(
        f"💰 {symbol}: do_buy — allocated=₹{cash_to_use:.2f}, "
        f"price=₹{price:.2f}, qty={qty}, min=₹{cfg['min_trade_amount']}, "
        f"catchup={catchup}, dry={cfg.get('dry_run')}"
    )
    if cash_to_use <= 0:
        reason = f"BUY skipped — ₹0 allocated. Check cash balance and min_cash_reserve_pct"
        log.warning(f"⚠️  {symbol}: {reason}")
        stocks_state[symbol]["error"] = reason
        return
    if qty <= 0:
        reason = f"BUY skipped — qty=0 (price ₹{price:.2f} > allocated ₹{cash_to_use:.2f})"
        log.warning(f"⚠️  {symbol}: {reason}")
        stocks_state[symbol]["error"] = reason
        return
    if qty * price < cfg["min_trade_amount"]:
        reason = f"BUY skipped — order ₹{qty*price:.2f} < min ₹{cfg['min_trade_amount']}"
        log.warning(f"⚠️  {symbol}: {reason}")
        stocks_state[symbol]["error"] = reason
        return

    tag    = "CATCHUP-BUY" if catchup else "BUY"
    dry    = cfg.get("dry_run")
    intel_str = intel.get("reason_detail", "") if intel else ""
    reason = (
        f"{'STARTUP CATCH-UP: ' if catchup else ''}"
        f"EMA{cfg['ema_green_period']} ({eg:.2f}) {'already ABOVE' if catchup else 'crossed ABOVE'} "
        f"EMA{cfg['ema_red_period']} ({er:.2f}). Gap: {gap:.2f}%. "
        f"{intel_str}"
        + (" [DRY RUN]" if dry else "")
    )

    if dry:
        oid = f"DRY-{tag}"
        log.info(f"🧪 DRY RUN — would {tag} {qty}×{symbol} @ ₹{price:.2f}")
    else:
        oid = kite.place_order(
            tradingsymbol=symbol, exchange=exchange,
            transaction_type=kite.TRANSACTION_TYPE_BUY,
            quantity=qty, order_type=kite.ORDER_TYPE_MARKET,
            product=kite.PRODUCT_CNC, variety=kite.VARIETY_REGULAR,
        )
        log.info(f"✅ {tag} {qty}×{symbol} @ ₹{price:.2f} | {oid}")

    _buy_date_tracker[symbol] = datetime.date.today()
    cash = float(kite.margins(segment="equity")["available"]["live_balance"]) if not dry else 0
    write_transaction({
        "timestamp":             datetime.datetime.now(IST).isoformat(),
        "action":                f"{'[DRY] ' if dry else ''}{tag}",
        "symbol":                symbol, "quantity": qty,
        "price":                 round(price, 2), "total_value": round(qty * price, 2),
        "available_cash_before": round(cash + qty * price if not dry else cash_to_use / (1 - TRADING_CONFIG["min_cash_reserve_pct"]), 2),
        "available_cash_after":  round(cash, 2),
        "holdings_before":       qty_held, "holdings_after": qty_held + qty,
        "reason":                reason,
        "ema_green":             round(eg, 4), "ema_red": round(er, 4), "holding_days": 0,
        "estimated_tax_note":    "CNC delivery. STCG 15% if <1yr, LTCG 10% if >1yr.",
    })
    stocks_state[symbol]["holdings_qty"] = qty_held + qty
    stocks_state[symbol]["error"] = ""

def do_sell(symbol: str, exchange: str, qty_held: int, avg_p: float,
            price: float, eg: float, er: float, gap: float, cash: float,
            intel: dict = None):
    cfg          = TRADING_CONFIG
    holding_days = 0
    bd           = _buy_date_tracker.get(symbol)
    if bd:
        holding_days = (datetime.date.today() - bd).days
        if holding_days < cfg["min_holding_days"]:
            log.info(f"⏳ {symbol}: skip sell — held {holding_days}d < min {cfg['min_holding_days']}d")
            return

    profit_pct = (price - avg_p) / avg_p * 100 if avg_p else 0
    if profit_pct < cfg["min_profit_pct_to_sell"]:
        log.info(f"📉 {symbol}: skip sell — profit {profit_pct:.2f}% < threshold")
        return

    # ── Conviction-aware sell sizing ─────────────────────────────────────────
    # STRONG conviction + strong gap  → full exit (trend reversal is real)
    # MODERATE conviction             → sell 50% (may be temporary pullback)
    # WEAK conviction                 → sell only 25% (very cautious, false signal risk)
    # LTCG (held >365d)               → always full exit regardless of conviction
    conv_label = intel.get("conviction_label", "MODERATE") if intel else "MODERATE"
    conv_score = intel.get("conviction", 50) if intel else 50
    is_strong_gap = gap >= cfg["strong_signal_threshold"] * 100
    is_ltcg       = holding_days >= cfg["stcg_holding_days"]

    if is_ltcg:
        sell_qty, sell_type = qty_held, "FULL(LTCG)"
    elif conv_label == "STRONG" or (conv_label == "MODERATE" and is_strong_gap):
        sell_qty, sell_type = qty_held, "FULL"
    elif conv_label == "MODERATE":
        sell_qty  = max(1, int(qty_held * cfg["partial_sell_pct"]))
        sell_type = "PARTIAL-50%"
    else:  # WEAK — very conservative, may be noise
        sell_qty  = max(1, int(qty_held * 0.25))
        sell_type = "PARTIAL-25%"

    tax_note = (f"LTCG 10% — held {holding_days}d." if is_ltcg
                else f"STCG 15% — {sell_type} sell. Gap {gap:.2f}%, conviction={conv_score}/100, held {holding_days}d.")
    dry       = cfg.get("dry_run")
    intel_str = intel.get("reason_detail", "") if intel else ""
    reason    = (f"EMA{cfg['ema_green_period']} ({eg:.2f}) crossed BELOW "
                 f"EMA{cfg['ema_red_period']} ({er:.2f}). Gap: {gap:.2f}%. "
                 f"Profit: {profit_pct:.2f}%. {intel_str}")

    if dry:
        oid = "DRY-SELL"
        log.info(f"🧪 DRY RUN — would SELL {sell_qty}×{symbol} @ ₹{price:.2f}")
    else:
        oid = kite.place_order(
            tradingsymbol=symbol, exchange=exchange,
            transaction_type=kite.TRANSACTION_TYPE_SELL,
            quantity=sell_qty, order_type=kite.ORDER_TYPE_MARKET,
            product=kite.PRODUCT_CNC, variety=kite.VARIETY_REGULAR,
        )
        log.info(f"✅ SELL ({sell_type}) {sell_qty}×{symbol} @ ₹{price:.2f} | {oid}")

    write_transaction({
        "timestamp":             datetime.datetime.now(IST).isoformat(),
        "action":                f"{'[DRY] ' if dry else ''}SELL ({sell_type})",
        "symbol":                symbol, "quantity": sell_qty,
        "price":                 round(price, 2), "total_value": round(sell_qty * price, 2),
        "available_cash_before": round(cash, 2),
        "available_cash_after":  round(cash + sell_qty * price, 2),
        "holdings_before":       qty_held, "holdings_after": qty_held - sell_qty,
        "reason":                reason + (" [DRY RUN]" if dry else ""),
        "ema_green":             round(eg, 4), "ema_red": round(er, 4),
        "holding_days":          holding_days, "estimated_tax_note": tax_note,
    })
    stocks_state[symbol]["holdings_qty"] = qty_held - sell_qty
    stocks_state[symbol]["error"] = ""

# ── Main agent loop ───────────────────────────────────────────────────────────
def agent_loop():
    cfg = TRADING_CONFIG
    init_log()
    log.info(f"🤖 Agent started — watching {[w['symbol'] for w in WATCHLIST]}")

    # Recovery
    recovery = recover_state()
    state["recovery"] = recovery
    if recovery["offline_days"] > 0:
        send_push("🔄 Agent Resumed",
                  f"Offline {recovery['offline_days']}d. {len(WATCHLIST)} stocks watched.", "system")
    else:
        send_push("🤖 Agent Started",
                  f"Watching {', '.join(w['symbol'] for w in WATCHLIST)}", "system")
    if recovery.get("warnings"):
        state["last_error"] = " | ".join(recovery["warnings"])

    # ── Loud dry_run warning so it is never missed ───────────────────────────
    if cfg.get('dry_run'):
        log.warning('=' * 60)
        log.warning('  🧪 DRY RUN MODE — NO REAL ORDERS WILL BE PLACED')
        log.warning('  Set dry_run: False in config.py to trade for real')
        log.warning('=' * 60)
    else:
        log.warning('=' * 60)
        log.warning('  🔴 LIVE TRADING MODE — REAL ORDERS WILL BE PLACED')
        log.warning('=' * 60)

    # ── Start background news fetcher ──────────────────────────────────────────
    if NEWS_AVAILABLE:
        news_thread = threading.Thread(target=_news_worker, daemon=True, name="news-worker")
        news_thread.start()
        log.info("📰 News sentiment worker started (refreshes every 30min)")
    else:
        log.warning("📰 news.py not found — news sentiment disabled")

    _first_run = True

    while not _stop_flag.is_set():
        try:
            market_now = is_market_open()
            state["market_open"] = market_now

            if not market_now:
                # Fetch holdings and cash even when market is closed
                try:
                    cash = float(kite.margins(segment="equity")["available"]["live_balance"])
                    state["cash"]         = f"₹{cash:.2f}"
                    state["last_checked"] = datetime.datetime.now(IST).strftime("%H:%M:%S")
                    mc_holdings = kite.holdings()
                    try:
                        mc_positions = kite.positions().get("day", [])
                    except Exception:
                        mc_positions = []
                    log.info(f"📊 [Market Closed] Cash: ₹{cash:.2f} | Holdings fetched for {len(WATCHLIST)} stocks")
                    for w in WATCHLIST:
                        sym     = w["symbol"]
                        demat   = next((h for h in mc_holdings  if h["tradingsymbol"] == sym), None)
                        pos     = next((p for p in mc_positions if p["tradingsymbol"] == sym), None)
                        qty     = (demat["quantity"] if demat else 0) + (pos["quantity"] if pos else 0)
                        avg_p   = (demat["average_price"] if demat and demat["quantity"] > 0
                                   else pos["average_price"] if pos and pos["quantity"] > 0 else 0)
                        ltp     = demat["last_price"] if demat else 0
                        stocks_state[sym]["signal"]       = "Market Closed"
                        stocks_state[sym]["holdings_qty"] = qty
                        stocks_state[sym]["avg_price"]    = f"₹{avg_p:.2f}" if avg_p else "—"
                        stocks_state[sym]["price"]        = f"₹{ltp:.2f}"   if ltp   else "—"
                        log.info(f"   📦 {sym}: qty={qty} | avg=₹{avg_p:.2f} | ltp=₹{ltp:.2f}")
                except Exception as e:
                    log.warning(f"⚠️  Market closed — could not fetch holdings: {e}")
                    for sym in stocks_state:
                        stocks_state[sym]["signal"] = "Market Closed"
                _stop_flag.wait(300)
                continue

            if not state["logged_in"]:
                log.warning("Token lost mid-session, stopping.")
                state["agent_running"] = False
                send_push("⚠️ Agent Stopped", "Session expired. Please re-login.", "login")
                break

            # ── Fetch margins once per tick ───────────────────────────────────
            cash = float(kite.margins(segment="equity")["available"]["live_balance"])
            state["cash"]         = f"₹{cash:.2f}"
            state["last_checked"] = datetime.datetime.now(IST).strftime("%H:%M:%S")
            state["last_error"]   = ""

            # ── Fetch holdings + positions once per tick (not per stock) ─────
            # kite.holdings() = settled demat (T+2). Stocks bought today won't
            # appear here until settlement. kite.positions() covers today's buys.
            try:
                h_list = kite.holdings()
            except Exception as e:
                log.warning(f"⚠️  Could not fetch holdings: {e}")
                h_list = []
            try:
                pos_day = kite.positions().get("day", [])
            except Exception as e:
                log.warning(f"⚠️  Could not fetch positions: {e}")
                pos_day = []

            # ── Scan all stocks first (need intel before volatility decision) ─
            buy_signals  = []
            sell_signals = []

            for w in WATCHLIST:
                sym   = w["symbol"]
                exch  = w["exchange"]
                ss    = stocks_state[sym]
                try:
                    df    = get_candles(sym, exch)
                    intel = analyse(df)          # ← full intelligence analysis
                    intel = apply_news_to_conviction(intel, sym)  # ← news adjustment
                    sig   = intel["signal"]
                    eg    = intel["ema_green"]
                    er    = intel["ema_red"]
                    gap   = intel["gap_pct"]

                    quote    = kite.quote(f"{exch}:{sym}")
                    price    = float(quote[f"{exch}:{sym}"]["last_price"])
                    qty_held, avg_p = get_holding(sym)

                    log.info(
                        f"📈 {sym} | ₹{price:.2f} | "
                        f"EMA{cfg['ema_green_period']}={eg:.4f} EMA{cfg['ema_red_period']}={er:.4f} "
                        f"gap={gap:.2f}% Δ{intel['gap_change']:+.3f}% | "
                        f"vol={intel['vol_ratio']:.1f}x | trend={intel['trend_candles']}c | "
                        f"conviction={intel['conviction']}/100({intel['conviction_label']}) | "
                        f"signal={sig} | qty={qty_held} avg=₹{avg_p:.2f}"
                    )

                    # ── Update all live state including new intelligence fields ─
                    ss["signal"]           = sig
                    ss["price"]            = f"₹{price:.2f}"
                    ss["ema_green"]        = f"{eg:.4f}"
                    ss["ema_red"]          = f"{er:.4f}"
                    ss["gap_pct"]          = f"{gap:.2f}%"
                    ss["gap_change"]       = f"{intel['gap_change']:+.3f}%"
                    ss["trend_candles"]    = intel["trend_candles"]
                    ss["vol_ratio"]        = f"{intel['vol_ratio']:.2f}x"
                    ss["conviction"]       = intel["conviction"]
                    ss["conviction_label"] = intel["conviction_label"]
                    ss["invest_pct"]       = f"{int(intel['invest_pct']*100)}%"
                    ss["holdings_qty"]     = qty_held
                    ss["avg_price"]        = f"₹{avg_p:.2f}" if avg_p else "—"
                    ss["error"]            = ""
                    # News fields (written by background thread, just copy to response)
                    # We also store the delta for the why-box
                    ss["news_delta"]       = intel.get("news_delta", 0)
                    ss["news_note"]        = intel.get("news_note", "")

                    is_catchup = (_first_run and eg > er
                                  and sig.startswith("HOLD") and qty_held == 0)
                    if is_catchup:
                        log.info(f"🚀 {sym}: CATCH-UP BUY | {intel['reason_detail']}")
                    elif _first_run and sig.startswith("HOLD"):
                        log.info(
                            f"ℹ️  {sym}: first-run HOLD, no catch-up — "
                            f"bearish={eg<=er} qty={qty_held}"
                        )

                    if sig == "BUY" or is_catchup:
                        buy_signals.append({
                            "symbol":     sym,  "exchange": exch,
                            "weight":     w["weight"], "gap_pct": gap,
                            "price":      price, "eg": eg, "er": er,
                            "qty_held":   qty_held, "avg_p": avg_p,
                            "catchup":    is_catchup,
                            "conviction": intel["conviction"],
                            "invest_pct": intel["invest_pct"],
                            "intel":      intel,
                        })
                    elif sig == "SELL" and qty_held > 0:
                        sell_signals.append({
                            "symbol":     sym,  "exchange": exch,
                            "price":      price, "eg": eg, "er": er,
                            "gap_pct":    gap, "qty_held": qty_held, "avg_p": avg_p,
                            "conviction": intel["conviction"],
                            "intel":      intel,
                        })

                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    ss["error"]  = err
                    ss["signal"] = "ERROR"
                    log.error(f"❌ {sym}: {err}")

            # ── Opening volatility guard ──────────────────────────────────────
            # Run AFTER scan so we can pass the strongest buy signal's intel
            # into the check — strong breakouts can override the buffer.
            strongest_intel = None
            if buy_signals:
                strongest_intel = max(
                    buy_signals, key=lambda x: x.get("conviction", 0)
                ).get("intel")

            ov = opening_volatility_status(intel=strongest_intel)
            state["opening_volatile"]   = ov["blocked"]
            state["opening_reason"]     = ov["reason"]
            state["opening_wait_until"] = ov.get("wait_until") or ""
            state["opening_early"]      = ov.get("early_entry", False)

            if ov["blocked"]:
                # Signal detected but market not safe — show in UI, don't trade
                detected = [s["symbol"] for s in sell_signals + buy_signals]
                if detected:
                    log.info(
                        f"⏸️  {detected} signal(s) detected but paused — {ov['reason']}"
                    )

            else:
                # Safe to trade — either past buffer, or early-entry override granted
                if ov.get("early_entry"):
                    log.info(f"🚀 Early-entry override granted — executing signals now")

                # Sells first (frees cash before buying)
                for s in sell_signals:
                    do_sell(s["symbol"], s["exchange"], s["qty_held"], s["avg_p"],
                            s["price"], s["eg"], s["er"], s["gap_pct"], cash,
                            intel=s.get("intel"))

                if sell_signals:
                    time.sleep(2)
                    cash = float(kite.margins(segment="equity")["available"]["live_balance"])
                    state["cash"] = f"₹{cash:.2f}"

                if buy_signals:
                    alloc = allocate_cash(buy_signals, cash)
                    for s in buy_signals:
                        sym        = s["symbol"]
                        cash_alloc = alloc.get(sym, 0)
                        if cash_alloc <= 0:
                            log.info(f"⏭️  {sym}: ₹0 allocated, skipping")
                            continue
                        do_buy(sym, s["exchange"], cash_alloc, s["price"],
                               s["eg"], s["er"], s["gap_pct"], s["qty_held"], s["catchup"],
                               intel=s.get("intel"))

                    if len(buy_signals) > 1:
                        names = [s["symbol"] for s in buy_signals]
                        strat = cfg.get("multi_buy_strategy", "weighted")
                        log.info(f"📊 Multi-buy ({strat}): {names} | alloc={alloc}")
                        send_push(
                            f"📊 {len(buy_signals)} BUY signals",
                            f"{', '.join(names)} | strategy: {strat}",
                            tag="trade"
                        )

            _first_run = False

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            state["last_error"] = err
            log.error(f"❌ Loop error: {err}\n{traceback.format_exc()}")
            send_push("❌ Agent Error", err[:120], "error")

        _stop_flag.wait(cfg["check_interval_seconds"])

    state["agent_running"] = False
    log.info("🛑 Agent stopped")

# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    try_load_saved_token()
    state["dry_run"] = TRADING_CONFIG.get("dry_run", True)
    return render_template("index.html",
        state=state,
        stocks=list(stocks_state.values()),
        cfg=TRADING_CONFIG,
        watchlist=WATCHLIST,
        recovery=state.get("recovery"),
        push_available=PUSH_AVAILABLE and Path(VAPID_PUBLIC_KEY).exists(),
        vapid_public_key=Path(VAPID_PUBLIC_KEY).read_text().strip() if Path(VAPID_PUBLIC_KEY).exists() else "",
    )

@app.route("/login")
def login():
    return redirect(kite.login_url())

@app.route("/callback")
def callback():
    req_token = request.args.get("request_token")
    if request.args.get("status") != "success" or not req_token:
        return render_template("error.html", msg="Login was cancelled or failed.")
    try:
        data  = kite.generate_session(req_token, api_secret=ZERODHA_CONFIG["api_secret"])
        token = data["access_token"]
        kite.set_access_token(token)
        ZERODHA_CONFIG["access_token"] = token
        Path(PATHS["token_file"]).write_text(token)
        state["logged_in"] = True
        log.info("✅ Login successful")
        send_push("✅ Logged in", "Zerodha session active. You can start the agent.", "login")
        return redirect("/")
    except Exception as e:
        return render_template("error.html", msg=f"Token error: {e}")

@app.route("/start")
def start():
    global _agent_thread, _stop_flag
    if not state["logged_in"] or state["agent_running"]:
        return redirect("/")
    _stop_flag    = threading.Event()
    _agent_thread = threading.Thread(target=agent_loop, daemon=True)
    _agent_thread.start()
    state["agent_running"] = True
    return redirect("/")

@app.route("/stop")
def stop():
    global _stop_flag
    _stop_flag.set()
    state["agent_running"] = False
    send_push("⏹ Agent Stopped", "Trading agent has been stopped.", "system")
    return redirect("/")

@app.route("/status")
def status():
    """Full live state polled by JS every 15s — includes ALL per-stock data."""
    return jsonify({
        **{k: v for k, v in state.items() if k != "recovery"},
        "stocks": list(stocks_state.values()),
    })

@app.route("/api/news")
def api_news():
    """Return live news sentiment for all watchlist stocks."""
    result = {}
    for sym, ss in stocks_state.items():
        result[sym] = {
            "score":       ss.get("news_score", 0),
            "label":       ss.get("news_label", "—"),
            "summary":     ss.get("news_summary", "—"),
            "headlines":   ss.get("news_headlines", []),
            "fetched_at":  ss.get("news_fetched_at", "—"),
            "delta":       ss.get("news_delta", 0),
            "note":        ss.get("news_note", ""),
        }
    return jsonify(result)

@app.route("/api/news/refresh")
def api_news_refresh():
    """Force-clear news cache and trigger re-fetch on next worker cycle."""
    clear_news_cache()
    return jsonify({"ok": True, "message": "News cache cleared — will re-fetch within 30s"})

@app.route("/api/transactions")
def api_transactions():
    return jsonify(read_transactions(50))

@app.route("/subscribe", methods=["POST"])
def subscribe():
    sub = request.get_json()
    if sub and sub not in _push_subscriptions:
        _push_subscriptions.append(sub)
        state["notifications_enabled"] = True
    return jsonify({"ok": True})

@app.route("/unsubscribe", methods=["POST"])
def unsubscribe():
    sub = request.get_json()
    if sub in _push_subscriptions:
        _push_subscriptions.remove(sub)
    if not _push_subscriptions:
        state["notifications_enabled"] = False
    return jsonify({"ok": True})

@app.route("/test-push")
def test_push():
    send_push("🔔 Test", "Push notifications are working!", "test")
    return jsonify({"sent": True})

@app.route("/manifest.json")
def manifest():
    return jsonify({
        "name": "Zerodha Agent", "short_name": "ZAgent",
        "description": "Automated multi-stock EMA trading agent",
        "start_url": "/", "display": "standalone",
        "background_color": "#0f0f14", "theme_color": "#0f0f14",
        "orientation": "portrait",
        "icons": [
            {"src": "/static/icons/icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "/static/icons/icon-512.png", "sizes": "512x512", "type": "image/png"},
        ]
    })

@app.route("/sw.js")
def service_worker():
    return Response(r"""
const CACHE='zagent-v2';
self.addEventListener('install',e=>{e.waitUntil(caches.open(CACHE).then(c=>c.addAll(['/'])));self.skipWaiting();});
self.addEventListener('activate',e=>{e.waitUntil(clients.claim());});
self.addEventListener('fetch',e=>{e.respondWith(fetch(e.request).catch(()=>caches.match(e.request)));});
self.addEventListener('push',e=>{
  let d={title:'Zerodha Agent',body:'Update',tag:'alert',url:'/'};
  try{d={...d,...e.data.json()};}catch{}
  e.waitUntil(self.registration.showNotification(d.title,{body:d.body,tag:d.tag,icon:'/static/icons/icon-192.png',data:{url:d.url},vibrate:[200,100,200]}));
});
self.addEventListener('notificationclick',e=>{
  e.notification.close();
  const url=e.notification.data?.url||'/';
  e.waitUntil(clients.matchAll({type:'window',includeUncontrolled:true}).then(list=>{
    for(const c of list){if(c.url.includes(self.location.origin)){c.focus();c.navigate(url);return;}}
    return clients.openWindow(url);
  }));
});
""", mimetype="application/javascript")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=schedule_daily_login_reminder, daemon=True).start()
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        local_ip = "localhost"
    print(f"\n{'═'*56}")
    print(f"  📱 ZERODHA AGENT — {len(WATCHLIST)} stocks watched")
    print(f"{'═'*56}")
    print(f"  Local:    http://localhost:8080")
    print(f"  On phone: http://{local_ip}:8080")
    print(f"  Kite redirect URL → http://{local_ip}:8080/callback")
    print(f"{'═'*56}\n")
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)
