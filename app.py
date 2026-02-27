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
import pandas as pd

try:
    from pywebpush import webpush, WebPushException
    PUSH_AVAILABLE = True
except ImportError:
    PUSH_AVAILABLE = False

from config import ZERODHA_CONFIG, TRADING_CONFIG, WATCHLIST, PATHS

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
    "notifications_enabled": False,
    "dry_run":              TRADING_CONFIG.get("dry_run", True),
    "recovery":             None,
}
# Per-symbol live data updated each loop tick
stocks_state = {
    w["symbol"]: {
        "symbol":       w["symbol"],
        "exchange":     w["exchange"],
        "weight":       w["weight"],
        "signal":       "—",
        "price":        "—",
        "ema_green":    "—",
        "ema_red":      "—",
        "gap_pct":      "—",
        "holdings_qty": 0,
        "avg_price":    "—",
        "error":        "",
    }
    for w in WATCHLIST
}

_stop_flag          = threading.Event()
_agent_thread       = None
_push_subscriptions = []
_buy_date_tracker   = {}   # symbol -> datetime.date
_instrument_cache   = {}   # exchange -> list of instruments (refreshed daily)

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

def get_candles(symbol: str, exchange: str) -> pd.DataFrame:
    cfg       = TRADING_CONFIG
    interval  = cfg["candle_interval"]
    safe_days = min(MAX_DAYS.get(interval, 60) - 2, 58)
    to_dt     = datetime.datetime.now()
    fr_dt     = to_dt - datetime.timedelta(days=safe_days)
    tok       = get_instrument_token(symbol, exchange)
    data      = kite.historical_data(tok, fr_dt, to_dt, interval)
    df        = pd.DataFrame(data).rename(columns={"date": "timestamp"})
    return df.sort_values("timestamp").tail(cfg["lookback_candles"]).reset_index(drop=True)

def calc_ema(prices: pd.Series, period: int) -> pd.Series:
    return prices.ewm(span=period, adjust=False).mean()

def get_signal(df: pd.DataFrame):
    """Returns (signal_str, ema_green_val, ema_red_val, gap_pct)"""
    cfg   = TRADING_CONFIG
    close = df["close"]
    eg    = calc_ema(close, cfg["ema_green_period"])
    er    = calc_ema(close, cfg["ema_red_period"])
    pg, pr = eg.iloc[-2], er.iloc[-2]
    cg, cr = eg.iloc[-1], er.iloc[-1]
    gap   = abs(cg - cr) / cr * 100
    if pg <= pr and cg > cr: return "BUY",  cg, cr, gap
    if pg >= pr and cg < cr: return "SELL", cg, cr, gap
    return f"HOLD({'↑' if cg > cr else '↓'})", cg, cr, gap

def is_market_open() -> bool:
    now = datetime.datetime.now(IST)
    if now.weekday() >= 5:
        return False
    o = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    c = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return o <= now <= c

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
    Allocates investable cash across buy signals.

    Smart fallback: if splitting cash results in any allocation below
    min_trade_amount, automatically concentrates into the fewest stocks
    (strongest signals first) that can each receive a viable amount.
    """
    cfg        = TRADING_CONFIG
    strategy   = cfg.get("multi_buy_strategy", "weighted")
    reserve    = cfg.get("min_cash_reserve_pct", 0.10)
    min_order  = cfg.get("min_trade_amount", 500)
    investable = available_cash * (1 - reserve)

    if not buy_signals:
        return {}

    # ── Helper: check if all allocations are above min_order ─────────────────
    def all_viable(alloc: dict) -> bool:
        return all(v >= min_order for v in alloc.values())

    # ── Initial allocation by strategy ────────────────────────────────────────
    def make_allocation(signals: list) -> dict:
        alloc = {}
        if strategy == "top1" or len(signals) == 1:
            best = max(signals, key=lambda x: x["gap_pct"])
            alloc[best["symbol"]] = investable
        elif strategy == "equal":
            per = investable / len(signals)
            for s in signals: alloc[s["symbol"]] = per
        else:  # weighted
            total_w = sum(s["weight"] for s in signals)
            for s in signals: alloc[s["symbol"]] = investable * (s["weight"] / total_w)
        return alloc

    allocation = make_allocation(buy_signals)

    # ── Smart fallback: drop weakest signals until all get viable amounts ─────
    if not all_viable(allocation) and len(buy_signals) > 1:
        # Sort by signal strength (biggest EMA gap = strongest)
        ranked = sorted(buy_signals, key=lambda x: x["gap_pct"], reverse=True)
        for n in range(len(ranked), 0, -1):
            candidate = make_allocation(ranked[:n])
            if all_viable(candidate):
                allocation = candidate
                dropped = [s["symbol"] for s in ranked[n:]]
                if dropped:
                    log.info(
                        f"💡 Low cash fallback: dropped {dropped} (weak signals), "
                        f"concentrating ₹{investable:.0f} into {[s['symbol'] for s in ranked[:n]]}"
                    )
                break
        else:
            # Even 1 stock can't meet min — put everything into the strongest
            best = max(buy_signals, key=lambda x: x["gap_pct"])
            allocation = {best["symbol"]: investable}
            log.info(
                f"💡 Cash too low to split — putting ₹{investable:.0f} into "
                f"{best['symbol']} (strongest gap {best['gap_pct']:.2f}%)"
            )

    log.info(f"💰 Allocation: total=₹{investable:.0f} | {allocation}")
    return allocation

# ── Place buy/sell ────────────────────────────────────────────────────────────
def do_buy(symbol: str, exchange: str, cash_to_use: float, price: float,
           eg: float, er: float, gap: float, qty_held: int, catchup: bool = False):
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
    reason = (
        f"{'STARTUP CATCH-UP: ' if catchup else ''}"
        f"EMA{cfg['ema_green_period']} ({eg:.2f}) {'already ABOVE' if catchup else 'crossed ABOVE'} "
        f"EMA{cfg['ema_red_period']} ({er:.2f}). Gap: {gap:.2f}%."
        f"{' Multi-stock allocation.' if not catchup else ''}"
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
            price: float, eg: float, er: float, gap: float, cash: float):
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

    is_strong = gap >= cfg["strong_signal_threshold"] * 100
    is_ltcg   = holding_days >= cfg["stcg_holding_days"]
    sell_qty  = qty_held if (is_strong or is_ltcg) else max(1, int(qty_held * cfg["partial_sell_pct"]))
    sell_type = "FULL" if sell_qty == qty_held else "PARTIAL"
    tax_note  = (f"LTCG 10% — held {holding_days}d." if is_ltcg
                 else f"STCG 15% — {sell_type} sell. Gap {gap:.2f}%, held {holding_days}d.")
    dry       = cfg.get("dry_run")
    reason    = (f"EMA{cfg['ema_green_period']} ({eg:.2f}) crossed BELOW "
                 f"EMA{cfg['ema_red_period']} ({er:.2f}). Gap: {gap:.2f}%. Profit: {profit_pct:.2f}%.")

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
                    h_list = kite.holdings()
                    log.info(f"📊 [Market Closed] Cash: ₹{cash:.2f} | Holdings fetched for {len(WATCHLIST)} stocks")
                    for w in WATCHLIST:
                        sym     = w["symbol"]
                        holding = next((h for h in h_list if h["tradingsymbol"] == sym), None)
                        qty     = holding["quantity"]      if holding else 0
                        avg_p   = holding["average_price"] if holding else 0
                        ltp     = holding["last_price"]    if holding else 0
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

            # ── Scan all stocks ───────────────────────────────────────────────
            buy_signals  = []   # stocks with active BUY signal this tick
            sell_signals = []   # stocks with active SELL signal

            for w in WATCHLIST:
                sym   = w["symbol"]
                exch  = w["exchange"]
                ss    = stocks_state[sym]
                try:
                    df                = get_candles(sym, exch)
                    sig, eg, er, gap  = get_signal(df)
                    quote             = kite.quote(f"{exch}:{sym}")
                    price             = float(quote[f"{exch}:{sym}"]["last_price"])
                    h_list            = kite.holdings()
                    holding           = next((h for h in h_list if h["tradingsymbol"] == sym), None)
                    qty_held          = holding["quantity"]      if holding else 0
                    avg_p             = holding["average_price"] if holding else 0

                    log.info(
                        f"📈 {sym} | price=₹{price:.2f} | "
                        f"EMA{cfg['ema_green_period']}={eg:.4f} EMA{cfg['ema_red_period']}={er:.4f} "
                        f"gap={gap:.2f}% | signal={sig} | "
                        f"qty={qty_held} avg=₹{avg_p:.2f} | candles={len(df)}"
                    )

                    # Update per-stock live state
                    ss["signal"]       = sig
                    ss["price"]        = f"₹{price:.2f}"
                    ss["ema_green"]    = f"{eg:.4f}"
                    ss["ema_red"]      = f"{er:.4f}"
                    ss["gap_pct"]      = f"{gap:.2f}%"
                    ss["holdings_qty"] = qty_held
                    ss["avg_price"]    = f"₹{avg_p:.2f}" if avg_p else "—"
                    ss["error"]        = ""

                    is_catchup = (_first_run and eg > er
                                  and sig.startswith("HOLD") and qty_held == 0)
                    if is_catchup:
                        log.info(
                            f"🚀 {sym}: CATCH-UP BUY eligible — "                            f"EMA{cfg['ema_green_period']}={eg:.4f} > EMA{cfg['ema_red_period']}={er:.4f}, "                            f"gap={gap:.2f}%, holdings=0, cash=₹{cash:.2f}"
                        )
                    elif _first_run and sig.startswith("HOLD"):
                        log.info(
                            f"ℹ️  {sym}: first-run HOLD, no catch-up — "                            f"eg={eg:.4f} er={er:.4f} bearish={eg<=er} qty={qty_held}"
                        )

                    if sig == "BUY" or is_catchup:
                        buy_signals.append({
                            "symbol": sym, "exchange": exch,
                            "weight": w["weight"], "gap_pct": gap,
                            "price": price, "eg": eg, "er": er,
                            "qty_held": qty_held, "avg_p": avg_p,
                            "catchup": is_catchup,
                        })
                    elif sig == "SELL" and qty_held > 0:
                        sell_signals.append({
                            "symbol": sym, "exchange": exch,
                            "price": price, "eg": eg, "er": er,
                            "gap_pct": gap, "qty_held": qty_held, "avg_p": avg_p,
                        })

                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    ss["error"]  = err
                    ss["signal"] = "ERROR"
                    log.error(f"❌ {sym}: {err}")

            # ── Execute sells first (free up cash before buying) ──────────────
            for s in sell_signals:
                do_sell(s["symbol"], s["exchange"], s["qty_held"], s["avg_p"],
                        s["price"], s["eg"], s["er"], s["gap_pct"], cash)

            # ── Refresh cash after sells ──────────────────────────────────────
            if sell_signals:
                time.sleep(2)
                cash = float(kite.margins(segment="equity")["available"]["live_balance"])
                state["cash"] = f"₹{cash:.2f}"

            # ── Allocate cash across buy signals ──────────────────────────────
            if buy_signals:
                alloc = allocate_cash(buy_signals, cash)
                for s in buy_signals:
                    sym        = s["symbol"]
                    cash_alloc = alloc.get(sym, 0)
                    if cash_alloc <= 0:
                        log.info(f"⏭️  {sym}: ₹0 allocated (dropped by low-cash fallback), skipping")
                        continue
                    # Note: do_buy itself checks if qty*price >= min_trade_amount
                    do_buy(sym, s["exchange"], cash_alloc, s["price"],
                           s["eg"], s["er"], s["gap_pct"], s["qty_held"], s["catchup"])

            _first_run = False  # only clear AFTER buys executed this tick

            if len(buy_signals) > 1:
                names = [s["symbol"] for s in buy_signals]
                strat = cfg.get("multi_buy_strategy", "weighted")
                log.info(f"📊 Multi-buy ({strat}): {names} | alloc={alloc}")
                send_push(
                    f"📊 {len(buy_signals)} BUY signals",
                    f"{', '.join(names)} | strategy: {strat}",
                    tag="trade"
                )

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
