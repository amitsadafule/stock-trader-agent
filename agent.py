"""
ZERODHA EMA CROSSOVER TRADING AGENT
=====================================
Strategy: Buy when 7-EMA crosses above 14-EMA, Sell when it crosses below.
Tax-smart: Avoids frequent trades, prefers long-term holding (>1 year = LTCG @ 10%).
Runs as a continuous loop — suitable for phone via Termux or server.

Requirements:
    pip install kiteconnect pandas requests schedule
"""

import os
import csv
import json
import time
import logging
import datetime
import traceback
from pathlib import Path

import requests

try:
    from kiteconnect import KiteConnect
except ImportError:
    raise ImportError("Install kiteconnect: pip install kiteconnect")

from config import ZERODHA_CONFIG, TRADING_CONFIG, WATCHLIST, PATHS

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent.log"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# TRANSACTION LOGGER
# ─────────────────────────────────────────────
TRANSACTION_FIELDS = [
    "timestamp", "action", "symbol", "quantity", "price",
    "total_value", "available_cash_before", "available_cash_after",
    "holdings_before", "holdings_after", "reason",
    "ema_green", "ema_red", "holding_days", "estimated_tax_note"
]


def init_transaction_log():
    path = PATHS["transaction_log"]
    if not Path(path).exists():
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRANSACTION_FIELDS)
            writer.writeheader()
    log.info(f"Transaction log: {path}")


def log_transaction(data: dict):
    with open(PATHS["transaction_log"], "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRANSACTION_FIELDS)
        writer.writerow({k: data.get(k, "") for k in TRANSACTION_FIELDS})
    log.info(f"📝 Logged transaction: {data.get('action')} {data.get('quantity')} "
             f"{data.get('symbol')} @ ₹{data.get('price')}")


# ─────────────────────────────────────────────
# STATE PERSISTENCE  (fixes holdings across restarts)
# ─────────────────────────────────────────────
def save_state(state: dict):
    """Persist agent state to disk so it survives restarts."""
    with open(PATHS["state_file"], "w") as f:
        json.dump(state, f, indent=2)


def load_state() -> dict:
    """Load persisted agent state (buy dates etc.) from disk."""
    path = PATHS["state_file"]
    if Path(path).exists():
        try:
            with open(path) as f:
                state = json.load(f)
            log.info(f"📂 Loaded state: {state}")
            return state
        except Exception as e:
            log.warning(f"Could not load state file ({e}), starting fresh.")
    state = {"buy_dates": {}}
    save_state(state)  # Create the file immediately so it's always present
    return state


# ─────────────────────────────────────────────
# KITE CONNECT WRAPPER
# ─────────────────────────────────────────────
class ZerodhaAgent:
    def __init__(self):
        self.cfg = TRADING_CONFIG
        self.kite = KiteConnect(api_key=ZERODHA_CONFIG["api_key"])
        self._load_or_refresh_token()
        self._instrument_cache = {}  # exchange -> list of instruments

    # ── AUTH ──────────────────────────────────
    def _load_or_refresh_token(self):
        token_file = PATHS["token_file"]
        if Path(token_file).exists():
            token = Path(token_file).read_text().strip()
            self.kite.set_access_token(token)
            ZERODHA_CONFIG["access_token"] = token
            log.info("✅ Loaded saved access token.")
        else:
            self._login_flow()

    def _login_flow(self):
        """Interactive login — required once per day."""
        login_url = self.kite.login_url()
        print("\n" + "="*60)
        print("ZERODHA LOGIN REQUIRED")
        print("="*60)
        print(f"1. Open this URL in browser:\n   {login_url}")
        print("2. Login with your Zerodha credentials")
        print("3. After redirect, copy the 'request_token' from the URL")
        print("   URL looks like: https://127.0.0.1/?request_token=XXXX&action=login&status=success")
        request_token = input("\nPaste the request_token here: ").strip()
        data = self.kite.generate_session(request_token, api_secret=ZERODHA_CONFIG["api_secret"])
        access_token = data["access_token"]
        self.kite.set_access_token(access_token)
        ZERODHA_CONFIG["access_token"] = access_token
        Path(PATHS["token_file"]).write_text(access_token)
        log.info("✅ Login successful. Token saved.")

    # ── MARKET DATA ───────────────────────────
    def get_candles(self, symbol: str, exchange: str) -> list:
        interval = self.cfg["candle_interval"]
        lookback = self.cfg["lookback_candles"]

        to_date = datetime.datetime.now()
        from_date = to_date - datetime.timedelta(days=lookback * 3)

        try:
            data = self.kite.historical_data(
                instrument_token=self._get_instrument_token(symbol, exchange),
                from_date=from_date,
                to_date=to_date,
                interval=interval,
            )
            rows = [{"timestamp": r["date"], "open": r["open"], "high": r["high"],
                     "low": r["low"], "close": r["close"], "volume": r.get("volume", 0)}
                    for r in data]
            rows.sort(key=lambda r: r["timestamp"])
            return rows[-lookback:]
        except Exception as e:
            log.error(f"Failed to fetch candles for {symbol}: {e}")
            raise

    def _get_instrument_token(self, symbol: str, exchange: str) -> int:
        if exchange not in self._instrument_cache:
            self._instrument_cache[exchange] = self.kite.instruments(exchange)
        for inst in self._instrument_cache[exchange]:
            if inst["tradingsymbol"] == symbol:
                return inst["instrument_token"]
        raise ValueError(f"Instrument not found: {exchange}:{symbol}")

    def get_available_cash(self) -> float:
        margins = self.kite.margins(segment="equity")
        return float(margins["available"]["live_balance"])

    def get_holdings(self, symbol: str) -> dict:
        """Returns holdings for a specific stock symbol."""
        holdings = self.kite.holdings()
        for h in holdings:
            if h["tradingsymbol"] == symbol:
                return {
                    "qty": h["quantity"],
                    "avg_price": h["average_price"],
                    "current_price": h["last_price"],
                }
        return {"qty": 0, "avg_price": 0, "current_price": 0}

    def get_all_holdings(self) -> dict:
        """Returns a dict of symbol -> holdings for all watchlist stocks."""
        result = {}
        api_holdings = self.kite.holdings()
        watchlist_symbols = {s["symbol"] for s in WATCHLIST}
        for h in api_holdings:
            if h["tradingsymbol"] in watchlist_symbols:
                result[h["tradingsymbol"]] = {
                    "qty": h["quantity"],
                    "avg_price": h["average_price"],
                    "current_price": h["last_price"],
                }
        # Ensure every watchlist symbol is represented
        for stock in WATCHLIST:
            if stock["symbol"] not in result:
                result[stock["symbol"]] = {"qty": 0, "avg_price": 0, "current_price": 0}
        return result

    def get_current_price(self, symbol: str, exchange: str) -> float:
        quote = self.kite.quote(f"{exchange}:{symbol}")
        return float(quote[f"{exchange}:{symbol}"]["last_price"])

    # ── EMA CALCULATION ───────────────────────
    @staticmethod
    def calculate_ema(prices: list, period: int) -> list:
        k = 2.0 / (period + 1)
        ema = [prices[0]]
        for price in prices[1:]:
            ema.append(price * k + ema[-1] * (1 - k))
        return ema

    def get_signal(self, candles: list):
        """
        Returns: ('BUY', ema_green, ema_red, gap_pct)
                 ('SELL', ema_green, ema_red, gap_pct)
                 ('HOLD', ema_green, ema_red, gap_pct)
        """
        close = [r["close"] for r in candles]
        ema_green = self.calculate_ema(close, self.cfg["ema_green_period"])
        ema_red   = self.calculate_ema(close, self.cfg["ema_red_period"])

        prev_green = ema_green[-2]
        prev_red   = ema_red[-2]
        curr_green = ema_green[-1]
        curr_red   = ema_red[-1]

        gap_pct = abs(curr_green - curr_red) / curr_red * 100

        # Crossover detection
        crossed_up   = prev_green <= prev_red and curr_green > curr_red
        crossed_down = prev_green >= prev_red and curr_green < curr_red

        if crossed_up:
            return "BUY", curr_green, curr_red, gap_pct
        elif crossed_down:
            return "SELL", curr_green, curr_red, gap_pct
        else:
            direction = "ABOVE" if curr_green > curr_red else "BELOW"
            return f"HOLD({direction})", curr_green, curr_red, gap_pct

    # ── TRADE EXECUTION ───────────────────────
    def place_buy(self, symbol: str, exchange: str, cash: float, price: float,
                  ema_green: float, ema_red: float, gap_pct: float,
                  action_label: str = "BUY") -> float:
        """
        Places a buy order. Returns updated cash after the order (deducted locally).
        Cash is also re-fetched from Kite after the order to keep it accurate.
        """
        cfg = self.cfg

        investable = cash * cfg["max_invest_pct"]
        qty = int(investable // price)

        if qty <= 0:
            reason = f"Insufficient funds (₹{cash:.2f} available, need ₹{price:.2f} for 1 share)"
            log.warning(f"⚠️  SKIP BUY {symbol} — {reason}")
            return cash

        order_value = qty * price
        if order_value < cfg["min_trade_amount"]:
            log.warning(f"⚠️  SKIP BUY {symbol} — Order value ₹{order_value:.2f} below minimum ₹{cfg['min_trade_amount']}")
            return cash

        reason = (
            f"{action_label}: EMA7 ({ema_green:.2f}) {'crossed ABOVE' if action_label == 'BUY' else 'already ABOVE'} "
            f"EMA14 ({ema_red:.2f}). Gap: {gap_pct:.2f}%. "
            f"Using {cfg['max_invest_pct']*100:.0f}% of allocated cash ₹{cash:.2f}. "
            f"CNC product to prefer LTCG tax treatment."
        )

        if cfg.get("dry_run"):
            log.info(f"[DRY RUN] Would BUY {qty} x {symbol} @ ~₹{price:.2f}")
            cash_after = cash - order_value
        else:
            try:
                order_id = self.kite.place_order(
                    tradingsymbol=symbol,
                    exchange=exchange,
                    transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                    quantity=qty,
                    order_type=self.kite.ORDER_TYPE_MARKET,
                    product=self.kite.PRODUCT_CNC,
                    variety=self.kite.VARIETY_REGULAR,
                )
                log.info(f"✅ BUY ORDER placed: {qty} x {symbol} @ ~₹{price:.2f} | Order ID: {order_id}")
                # Re-fetch actual cash from Kite after the order
                time.sleep(2)
                cash_after = self.get_available_cash()
            except Exception as e:
                log.error(f"❌ BUY ORDER FAILED for {symbol}: {e}")
                return cash

        tax_note = "CNC delivery. STCG 15% if <1yr, LTCG 10% if >1yr."
        log_transaction({
            "timestamp": datetime.datetime.now().astimezone().isoformat(),
            "action": action_label,
            "symbol": symbol,
            "quantity": qty,
            "price": price,
            "total_value": order_value,
            "available_cash_before": round(cash, 2),
            "available_cash_after": round(cash_after, 2),
            "holdings_before": 0,
            "holdings_after": qty,
            "reason": reason,
            "ema_green": round(ema_green, 4),
            "ema_red": round(ema_red, 4),
            "holding_days": 0,
            "estimated_tax_note": tax_note,
        })

        return cash_after

    def place_sell(self, symbol: str, exchange: str, holdings: dict, price: float,
                   ema_green: float, ema_red: float, gap_pct: float,
                   buy_date: datetime.date = None):
        cfg = self.cfg
        qty_held = holdings["qty"]
        avg_price = holdings["avg_price"]

        if qty_held <= 0:
            log.info(f"📭 No holdings to sell for {symbol}.")
            return

        # ── Minimum holding days check ────────
        holding_days = 0
        if buy_date:
            holding_days = (datetime.date.today() - buy_date).days
            if holding_days < cfg["min_holding_days"]:
                log.info(f"⏳ SKIP SELL {symbol} — Held only {holding_days} day(s). Minimum: {cfg['min_holding_days']} days.")
                return

        # ── Profit check ──────────────────────
        profit_pct = (price - avg_price) / avg_price * 100
        if profit_pct < cfg["min_profit_pct_to_sell"]:
            log.info(f"📉 SKIP SELL {symbol} — Profit {profit_pct:.2f}% below threshold {cfg['min_profit_pct_to_sell']}%.")
            return

        # ── Partial vs full sell decision ─────
        if gap_pct >= cfg["strong_signal_threshold"] * 100 or holding_days >= cfg["stcg_holding_days"]:
            sell_qty = qty_held
            sell_type = "FULL"
            tax_note = (
                f"LTCG (10%) applies — held {holding_days} days (>365)."
                if holding_days >= 365
                else f"STCG (15%) applies — held {holding_days} days. Strong signal ({gap_pct:.2f}% gap) justifies full exit."
            )
        else:
            sell_qty = max(1, int(qty_held * cfg["partial_sell_pct"]))
            sell_type = "PARTIAL"
            tax_note = (
                f"Partial sell ({cfg['partial_sell_pct']*100:.0f}%) to reduce STCG burden. "
                f"Weak crossover signal (gap {gap_pct:.2f}%). Keeping {qty_held - sell_qty} shares."
            )

        reason = (
            f"7-EMA ({ema_green:.2f}) crossed BELOW 14-EMA ({ema_red:.2f}). "
            f"Gap: {gap_pct:.2f}%. Profit: {profit_pct:.2f}%. "
            f"{sell_type} sell of {sell_qty}/{qty_held} shares. Held {holding_days} days."
        )

        if cfg.get("dry_run"):
            log.info(f"[DRY RUN] Would SELL {sell_qty} x {symbol} @ ~₹{price:.2f}")
            return

        try:
            order_id = self.kite.place_order(
                tradingsymbol=symbol,
                exchange=exchange,
                transaction_type=self.kite.TRANSACTION_TYPE_SELL,
                quantity=sell_qty,
                order_type=self.kite.ORDER_TYPE_MARKET,
                product=self.kite.PRODUCT_CNC,
                variety=self.kite.VARIETY_REGULAR,
            )
            # Re-fetch cash after sell
            time.sleep(2)
            cash = self.get_available_cash()
            log_transaction({
                "timestamp": datetime.datetime.now().astimezone().isoformat(),
                "action": f"SELL ({sell_type})",
                "symbol": symbol,
                "quantity": sell_qty,
                "price": price,
                "total_value": round(sell_qty * price, 2),
                "available_cash_before": round(cash - sell_qty * price, 2),
                "available_cash_after": round(cash, 2),
                "holdings_before": qty_held,
                "holdings_after": qty_held - sell_qty,
                "reason": reason,
                "ema_green": round(ema_green, 4),
                "ema_red": round(ema_red, 4),
                "holding_days": holding_days,
                "estimated_tax_note": tax_note,
            })
            log.info(f"✅ SELL ORDER placed: {sell_qty} x {symbol} @ ~₹{price:.2f} | Order ID: {order_id}")
        except Exception as e:
            log.error(f"❌ SELL ORDER FAILED for {symbol}: {e}")


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def is_market_open() -> bool:
    now = datetime.datetime.now()
    # NSE market: Mon-Fri, 9:15 AM to 3:30 PM IST
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    market_open  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def print_holdings_summary(all_holdings: dict, cash: float):
    """Display current portfolio state clearly in the logs."""
    log.info("─" * 55)
    log.info("📊 CURRENT PORTFOLIO")
    log.info(f"   💰 Available Cash : ₹{cash:.2f}")
    for symbol, h in all_holdings.items():
        if h["qty"] > 0:
            value = h["qty"] * h["current_price"]
            pnl = (h["current_price"] - h["avg_price"]) * h["qty"]
            log.info(
                f"   📦 {symbol:15s}: {h['qty']} units | "
                f"Avg ₹{h['avg_price']:.2f} | LTP ₹{h['current_price']:.2f} | "
                f"Value ₹{value:.2f} | P&L ₹{pnl:+.2f}"
            )
        else:
            log.info(f"   📭 {symbol:15s}: No holdings")
    log.info("─" * 55)


def run_agent():
    log.info("🚀 Zerodha EMA Crossover Agent starting...")
    init_transaction_log()

    # Load state FIRST — before agent init, so file is always created on startup
    state = load_state()
    buy_dates = state.get("buy_dates", {})  # symbol -> "YYYY-MM-DD"

    agent = ZerodhaAgent()

    interval = TRADING_CONFIG["check_interval_seconds"]
    log.info(f"📋 Watchlist: {[s['symbol'] for s in WATCHLIST]}")
    log.info(f"📐 EMA({TRADING_CONFIG['ema_green_period']}) vs EMA({TRADING_CONFIG['ema_red_period']})")
    log.info(f"🔁 Checking every {interval} seconds during market hours\n")

    first_run = True

    while True:
        try:
            # ── Always fetch and show portfolio (market open or closed) ──
            cash = agent.get_available_cash()
            all_holdings = agent.get_all_holdings()
            print_holdings_summary(all_holdings, cash)

            if not is_market_open():
                log.info("🕐 Market closed. No trading.")
                time.sleep(300)  # Check every 5 min during off-hours
                continue

            # ── Startup catch-up: buy if EMA already bullish ──
            if first_run:
                first_run = False
                for stock in WATCHLIST:
                    sym = stock["symbol"]
                    exch = stock["exchange"]
                    h = all_holdings[sym]
                    if h["qty"] > 0:
                        continue  # Already holding — skip catch-up

                    try:
                        df = agent.get_candles(sym, exch)
                        signal, ema_g, ema_r, gap = agent.get_signal(df)
                        if "ABOVE" in signal or signal == "BUY":
                            price = agent.get_current_price(sym, exch)
                            # Allocate cash by weight
                            total_weight = sum(s["weight"] for s in WATCHLIST)
                            alloc = cash * (stock["weight"] / total_weight) * TRADING_CONFIG["max_invest_pct"]
                            log.info(f"🔄 STARTUP CATCH-UP: {sym} EMA already bullish (gap {gap:.2f}%). Buying...")
                            cash = agent.place_buy(
                                sym, exch, alloc, price, ema_g, ema_r, gap,
                                action_label="CATCHUP-BUY"
                            )
                            buy_dates[sym] = datetime.date.today().isoformat()
                            save_state({"buy_dates": buy_dates})
                    except Exception as e:
                        log.error(f"❌ Startup catch-up error for {sym}: {e}")

                # Re-fetch cash after any catch-up buys
                cash = agent.get_available_cash()

            # ── Per-stock signal loop ─────────────
            for stock in WATCHLIST:
                sym = stock["symbol"]
                exch = stock["exchange"]

                try:
                    df = agent.get_candles(sym, exch)
                    signal, ema_g, ema_r, gap = agent.get_signal(df)
                    price = agent.get_current_price(sym, exch)
                    holdings = agent.get_holdings(sym)

                    log.info(
                        f"📈 {sym} ₹{price:.2f} | EMA7={ema_g:.2f} EMA14={ema_r:.2f} "
                        f"Gap={gap:.2f}% | Holdings={holdings['qty']} | Cash=₹{cash:.2f} | Signal={signal}"
                    )

                    if signal == "BUY":
                        # Allocate cash by watchlist weight
                        total_weight = sum(s["weight"] for s in WATCHLIST)
                        alloc = cash * (stock["weight"] / total_weight) * TRADING_CONFIG["max_invest_pct"]
                        new_cash = agent.place_buy(sym, exch, alloc, price, ema_g, ema_r, gap)
                        if new_cash != alloc:  # Order actually went through
                            cash = new_cash   # ← fixes cash not updating between stocks
                            buy_dates[sym] = datetime.date.today().isoformat()
                            save_state({"buy_dates": buy_dates})

                    elif signal == "SELL":
                        bd_str = buy_dates.get(sym)
                        buy_date = datetime.date.fromisoformat(bd_str) if bd_str else None
                        agent.place_sell(sym, exch, holdings, price, ema_g, ema_r, gap, buy_date)
                        # If fully sold, remove buy date from state
                        updated = agent.get_holdings(sym)
                        if updated["qty"] == 0:
                            buy_dates.pop(sym, None)
                            save_state({"buy_dates": buy_dates})

                except Exception as e:
                    log.error(f"❌ Error processing {sym}: {e}\n{traceback.format_exc()}")

            time.sleep(interval)

        except KeyboardInterrupt:
            log.info("\n🛑 Agent stopped by user.")
            break
        except Exception as e:
            log.error(f"❌ Error in main loop: {e}\n{traceback.format_exc()}")
            time.sleep(60)  # Back off on error


if __name__ == "__main__":
    run_agent()
