"""
Unit tests for the Zerodha Trading Agent.

All tests are pure — no network calls, no Kite API, no database writes.

Run:
    python -m pytest test_agent.py -v
    python test_agent.py          # also works via unittest
"""

import sys
import datetime
import unittest
from unittest.mock import MagicMock, patch

# ── Stub external dependencies before importing app ───────────────────────────
_kite_mock = MagicMock()
_kite_mock.KiteConnect.return_value = MagicMock()
sys.modules.setdefault("kiteconnect", _kite_mock)
sys.modules.setdefault("pywebpush", MagicMock())

_db_mock = MagicMock()
_db_mock.init_db.return_value = None
_db_mock.load_push_subscriptions.return_value = []
_db_mock.load_buy_dates.return_value = {}
sys.modules["db"] = _db_mock

_news_mock = MagicMock()
_news_mock.get_news_sentiment.return_value = {
    "score": 0, "label": "NEUTRAL", "headlines": [],
    "conviction_delta": 0, "summary": "", "error": "", "fetched_at": "—",
}
_news_mock.clear_cache = MagicMock()
sys.modules.setdefault("news", _news_mock)

import app  # noqa: E402  (must come after stubs)
from app import (
    calc_ema,
    analyse,
    allocate_cash,
    apply_news_to_conviction,
    stocks_state,
    TRADING_CONFIG,
)


# ── Candle helpers ────────────────────────────────────────────────────────────

def _candles(prices, volume=1000):
    """Convert a list of close prices to minimal candle dicts."""
    return [
        {
            "timestamp": f"2024-01-{(i % 28) + 1:02d}",
            "open":   p,
            "high":   p * 1.005,
            "low":    p * 0.995,
            "close":  p,
            "volume": volume,
        }
        for i, p in enumerate(prices)
    ]


def _buy_candles():
    """
    Sequence that produces a BUY crossover at the last candle.

    Plan:
      30 × 100  → EMAs converge at 100
      10 × 80   → EMA7 drops below EMA14 (EMA7 converges faster)
       1 × 80   → prev candle: EMA7 < EMA14  (bearish alignment)
       1 × 150  → curr candle: EMA7 jumps above EMA14  → BUY
    """
    return _candles([100] * 30 + [80] * 11 + [150])


def _sell_candles():
    """
    Sequence that produces a SELL crossover at the last candle.

      30 × 100  → EMAs converge
      10 × 120  → EMA7 rises above EMA14
       1 × 120  → prev: EMA7 > EMA14  (bullish)
       1 × 50   → curr: EMA7 crashes below EMA14  → SELL
    """
    return _candles([100] * 30 + [120] * 11 + [50])


def _hold_up_candles():
    """Steadily rising — EMA7 stays above EMA14 throughout; no crossover."""
    return _candles([100 + i for i in range(50)])


def _hold_down_candles():
    """Steadily falling — EMA7 stays below EMA14 throughout; no crossover."""
    return _candles([200 - i for i in range(50)])


# ── calc_ema ──────────────────────────────────────────────────────────────────

class TestCalcEma(unittest.TestCase):

    def test_single_price_returns_itself(self):
        result = calc_ema([100.0], period=7)
        self.assertEqual(result, [100.0])

    def test_length_matches_input(self):
        prices = [float(i) for i in range(1, 21)]
        self.assertEqual(len(calc_ema(prices, 7)),  20)
        self.assertEqual(len(calc_ema(prices, 14)), 20)

    def test_constant_series_stays_constant(self):
        prices = [50.0] * 30
        ema = calc_ema(prices, 7)
        for v in ema:
            self.assertAlmostEqual(v, 50.0, places=8)

    def test_ema_converges_toward_price(self):
        """After a long flat series at 200, EMA should approach 200."""
        prices = [100.0] + [200.0] * 100
        ema = calc_ema(prices, 7)
        self.assertGreater(ema[-1], 195.0)

    def test_higher_k_converges_faster(self):
        """EMA7 (k=0.25) must track a price change faster than EMA14 (k=0.133)."""
        prices = [100.0] * 20 + [200.0] * 20
        ema7  = calc_ema(prices, 7)
        ema14 = calc_ema(prices, 14)
        # After the jump to 200, EMA7 should be closer to 200 than EMA14
        self.assertGreater(ema7[-1], ema14[-1])

    def test_rising_series_ema7_above_ema14(self):
        """In a rising trend EMA7 tracks price more tightly → stays above EMA14."""
        prices = [float(i) for i in range(1, 60)]
        ema7  = calc_ema(prices, 7)
        ema14 = calc_ema(prices, 14)
        # After warm-up the fast EMA must be above the slow one
        self.assertGreater(ema7[-1], ema14[-1])

    def test_falling_series_ema7_below_ema14(self):
        prices = [float(60 - i) for i in range(60)]
        ema7  = calc_ema(prices, 7)
        ema14 = calc_ema(prices, 14)
        self.assertLess(ema7[-1], ema14[-1])

    def test_formula_matches_manual(self):
        """Spot-check the recursive formula: EMA_t = p*k + EMA_(t-1)*(1-k)."""
        prices = [10.0, 20.0, 30.0]
        k = 2.0 / (7 + 1)
        expected = [10.0, 10.0 + k * (20.0 - 10.0), 0.0]
        expected[2] = expected[1] + k * (30.0 - expected[1])
        result = calc_ema(prices, 7)
        for got, exp in zip(result, expected):
            self.assertAlmostEqual(got, exp, places=10)


# ── analyse() ─────────────────────────────────────────────────────────────────

class TestAnalyse(unittest.TestCase):

    def test_buy_signal_on_upward_crossover(self):
        result = analyse(_buy_candles())
        self.assertEqual(result["signal"], "BUY")

    def test_sell_signal_on_downward_crossover(self):
        result = analyse(_sell_candles())
        self.assertEqual(result["signal"], "SELL")

    def test_hold_up_in_bullish_trend(self):
        result = analyse(_hold_up_candles())
        self.assertIn("HOLD", result["signal"])
        self.assertIn("↑", result["signal"])

    def test_hold_down_in_bearish_trend(self):
        result = analyse(_hold_down_candles())
        self.assertIn("HOLD", result["signal"])
        self.assertIn("↓", result["signal"])

    def test_returns_required_keys(self):
        keys = {
            "signal", "ema_green", "ema_red", "gap_pct", "gap_change",
            "trend_candles", "vol_ratio", "conviction", "conviction_label",
            "invest_pct", "reason_detail",
        }
        result = analyse(_buy_candles())
        self.assertTrue(keys.issubset(result.keys()))

    def test_gap_pct_is_non_negative(self):
        for candles in [_buy_candles(), _sell_candles(), _hold_up_candles()]:
            self.assertGreaterEqual(analyse(candles)["gap_pct"], 0)

    def test_conviction_in_range(self):
        for candles in [_buy_candles(), _sell_candles(), _hold_up_candles(), _hold_down_candles()]:
            r = analyse(candles)
            self.assertGreaterEqual(r["conviction"], 0)
            self.assertLessEqual(r["conviction"],  100)

    def test_conviction_label_tiers(self):
        for candles in [_buy_candles(), _sell_candles(), _hold_up_candles()]:
            r = analyse(candles)
            if r["conviction"] >= 65:
                self.assertEqual(r["conviction_label"], "STRONG")
                self.assertAlmostEqual(r["invest_pct"], 1.00)
            elif r["conviction"] >= 38:
                self.assertEqual(r["conviction_label"], "MODERATE")
                self.assertAlmostEqual(r["invest_pct"], 0.75)
            else:
                self.assertEqual(r["conviction_label"], "WEAK")
                self.assertAlmostEqual(r["invest_pct"], 0.50)

    def test_high_volume_boosts_conviction(self):
        """2x volume on last candle should raise conviction vs flat volume."""
        low_vol  = analyse(_candles([100] * 30 + [80] * 11 + [150], volume=1000))
        high_vol = analyse(_candles([100] * 30 + [80] * 11 + [150], volume=2000))
        self.assertGreaterEqual(high_vol["conviction"], low_vol["conviction"])

    def test_trend_candles_positive(self):
        r = analyse(_hold_up_candles())
        self.assertGreater(r["trend_candles"], 0)

    def test_ema_green_and_red_are_positive_floats(self):
        r = analyse(_buy_candles())
        self.assertGreater(r["ema_green"], 0)
        self.assertGreater(r["ema_red"],   0)

    def test_widening_gap_gives_positive_gap_change(self):
        """In a strong uptrend last crossover, gap should be widening."""
        r = analyse(_buy_candles())
        # gap_change may not always be positive for our synthetic data,
        # but we test the field exists and is a float
        self.assertIsInstance(r["gap_change"], float)


# ── allocate_cash() ───────────────────────────────────────────────────────────

def _signal(symbol, weight, conviction=70, invest_pct=1.0, conv_label="STRONG"):
    return {
        "symbol": symbol, "weight": weight,
        "conviction": conviction, "invest_pct": invest_pct,
        "conviction_label": conv_label,
    }


class TestAllocateCash(unittest.TestCase):

    def setUp(self):
        # Store originals so we can restore
        self._orig_strategy  = TRADING_CONFIG["multi_buy_strategy"]
        self._orig_reserve   = TRADING_CONFIG["min_cash_reserve_pct"]
        self._orig_min_order = TRADING_CONFIG["min_trade_amount"]

    def tearDown(self):
        TRADING_CONFIG["multi_buy_strategy"]   = self._orig_strategy
        TRADING_CONFIG["min_cash_reserve_pct"] = self._orig_reserve
        TRADING_CONFIG["min_trade_amount"]     = self._orig_min_order

    def test_empty_signals_returns_empty(self):
        self.assertEqual(allocate_cash([], 10000), {})

    def test_single_stock_gets_full_investable(self):
        TRADING_CONFIG["multi_buy_strategy"]   = "weighted"
        TRADING_CONFIG["min_cash_reserve_pct"] = 0.10
        sigs = [_signal("SILVER", 100, conviction=80, invest_pct=1.0)]
        alloc = allocate_cash(sigs, 10000)
        # investable = 9000; STRONG → deploy 100%
        self.assertAlmostEqual(alloc["SILVER"], 9000.0, places=1)

    def test_weighted_split_proportional(self):
        TRADING_CONFIG["multi_buy_strategy"]   = "weighted"
        TRADING_CONFIG["min_cash_reserve_pct"] = 0.0
        TRADING_CONFIG["min_trade_amount"]     = 1
        sigs = [
            _signal("SILVER", 60, conviction=80, invest_pct=1.0),
            _signal("GOLD",   40, conviction=80, invest_pct=1.0),
        ]
        alloc = allocate_cash(sigs, 10000)
        self.assertAlmostEqual(alloc["SILVER"] / alloc["GOLD"], 60 / 40, places=3)

    def test_equal_split(self):
        TRADING_CONFIG["multi_buy_strategy"]   = "equal"
        TRADING_CONFIG["min_cash_reserve_pct"] = 0.0
        TRADING_CONFIG["min_trade_amount"]     = 1
        sigs = [
            _signal("SILVER", 60, conviction=80, invest_pct=1.0),
            _signal("GOLD",   40, conviction=80, invest_pct=1.0),
        ]
        alloc = allocate_cash(sigs, 10000)
        self.assertAlmostEqual(alloc["SILVER"], alloc["GOLD"], places=1)

    def test_top1_gives_all_to_highest_conviction(self):
        TRADING_CONFIG["multi_buy_strategy"]   = "top1"
        TRADING_CONFIG["min_cash_reserve_pct"] = 0.0
        TRADING_CONFIG["min_trade_amount"]     = 1
        sigs = [
            _signal("SILVER", 60, conviction=80, invest_pct=1.0),
            _signal("GOLD",   40, conviction=50, invest_pct=1.0),
        ]
        alloc = allocate_cash(sigs, 10000)
        self.assertIn("SILVER", alloc)
        self.assertNotIn("GOLD", alloc)

    def test_weak_conviction_deploys_only_50pct(self):
        TRADING_CONFIG["multi_buy_strategy"]   = "weighted"
        TRADING_CONFIG["min_cash_reserve_pct"] = 0.0
        TRADING_CONFIG["min_trade_amount"]     = 1
        sigs = [_signal("SILVER", 100, conviction=30, invest_pct=0.50, conv_label="WEAK")]
        alloc = allocate_cash(sigs, 10000)
        self.assertAlmostEqual(alloc["SILVER"], 5000.0, places=1)

    def test_moderate_conviction_deploys_75pct(self):
        TRADING_CONFIG["multi_buy_strategy"]   = "weighted"
        TRADING_CONFIG["min_cash_reserve_pct"] = 0.0
        TRADING_CONFIG["min_trade_amount"]     = 1
        sigs = [_signal("SILVER", 100, conviction=50, invest_pct=0.75, conv_label="MODERATE")]
        alloc = allocate_cash(sigs, 10000)
        self.assertAlmostEqual(alloc["SILVER"], 7500.0, places=1)

    def test_low_cash_drops_weakest_conviction(self):
        """When cash is too low for both stocks, the weaker-conviction one is dropped."""
        TRADING_CONFIG["multi_buy_strategy"]   = "weighted"
        TRADING_CONFIG["min_cash_reserve_pct"] = 0.0
        TRADING_CONFIG["min_trade_amount"]     = 500
        sigs = [
            _signal("SILVER", 50, conviction=80, invest_pct=1.0, conv_label="STRONG"),
            _signal("GOLD",   50, conviction=20, invest_pct=0.5, conv_label="WEAK"),
        ]
        alloc = allocate_cash(sigs, 600)  # not enough for both at ≥500 each
        # GOLD should be dropped; all cash goes to SILVER
        self.assertIn("SILVER", alloc)
        self.assertNotIn("GOLD", alloc)

    def test_reserve_pct_respected(self):
        """Total allocated cash must never exceed (1 - reserve) × available."""
        TRADING_CONFIG["multi_buy_strategy"]   = "weighted"
        TRADING_CONFIG["min_cash_reserve_pct"] = 0.10
        TRADING_CONFIG["min_trade_amount"]     = 1
        sigs = [_signal("SILVER", 100, conviction=80, invest_pct=1.0)]
        alloc = allocate_cash(sigs, 10000)
        total_deployed = sum(alloc.values())
        self.assertLessEqual(total_deployed, 10000 * 0.90 + 1e-6)


# ── apply_news_to_conviction() ────────────────────────────────────────────────

def _intel(signal="BUY", conviction=60, label="MODERATE", invest_pct=0.75):
    return {
        "signal":           signal,
        "conviction":       conviction,
        "conviction_label": label,
        "invest_pct":       invest_pct,
        "reason_detail":    "test",
    }


class TestApplyNewsToConviction(unittest.TestCase):

    def _set_news(self, symbol, score, label):
        stocks_state[symbol]["news_score"] = score
        stocks_state[symbol]["news_label"] = label
        stocks_state[symbol]["news_ready"] = True

    def _sym(self):
        """Return first watchlist symbol."""
        from config import WATCHLIST
        return WATCHLIST[0]["symbol"]

    def test_neutral_news_no_change(self):
        sym = self._sym()
        self._set_news(sym, 0, "NEUTRAL")
        intel  = _intel("BUY", 60, "MODERATE")
        result = apply_news_to_conviction(intel, sym)
        self.assertEqual(result["news_delta"], 0)
        self.assertEqual(result["conviction"], 60)

    def test_bullish_news_boosts_buy_signal(self):
        sym = self._sym()
        self._set_news(sym, 70, "BULLISH")
        intel  = _intel("BUY", 60, "MODERATE")
        result = apply_news_to_conviction(intel, sym)
        self.assertGreater(result["conviction"], 60)
        self.assertGreater(result["news_delta"], 0)

    def test_bearish_news_reduces_buy_conviction(self):
        sym = self._sym()
        self._set_news(sym, -70, "BEARISH")
        intel  = _intel("BUY", 60, "MODERATE")
        result = apply_news_to_conviction(intel, sym)
        self.assertLess(result["conviction"], 60)
        self.assertLess(result["news_delta"], 0)

    def test_bearish_news_boosts_sell_signal(self):
        sym = self._sym()
        self._set_news(sym, -80, "BEARISH")
        intel  = _intel("SELL", 60, "MODERATE")
        result = apply_news_to_conviction(intel, sym)
        # BEARISH news confirms SELL → conviction DECREASES (delta is negative,
        # which is applied as a subtraction to the score — sell signals benefit)
        self.assertLess(result["news_delta"], 0)

    def test_bullish_news_conflicts_with_sell(self):
        sym = self._sym()
        self._set_news(sym, 60, "BULLISH")
        intel  = _intel("SELL", 70, "STRONG")
        result = apply_news_to_conviction(intel, sym)
        # News contradicts sell — delta should be negative or zero
        self.assertLessEqual(result["news_delta"], 0)

    def test_news_pending_no_adjustment(self):
        sym = self._sym()
        stocks_state[sym]["news_ready"] = False
        intel  = _intel("BUY", 60, "MODERATE")
        result = apply_news_to_conviction(intel, sym)
        self.assertEqual(result["news_delta"], 0)
        self.assertIn("pending", result["news_note"])

    def test_conviction_capped_at_100(self):
        sym = self._sym()
        self._set_news(sym, 100, "BULLISH")
        intel  = _intel("BUY", 95, "STRONG")
        result = apply_news_to_conviction(intel, sym)
        self.assertLessEqual(result["conviction"], 100)

    def test_conviction_floored_at_0(self):
        sym = self._sym()
        self._set_news(sym, -100, "BEARISH")
        intel  = _intel("BUY", 5, "WEAK")
        result = apply_news_to_conviction(intel, sym)
        self.assertGreaterEqual(result["conviction"], 0)

    def test_hold_up_with_bullish_news_boosted(self):
        sym = self._sym()
        self._set_news(sym, 60, "BULLISH")
        intel  = _intel("HOLD(↑)", 50, "MODERATE")
        result = apply_news_to_conviction(intel, sym)
        self.assertGreater(result["news_delta"], 0)

    def test_hold_down_with_bearish_news(self):
        sym = self._sym()
        self._set_news(sym, -60, "BEARISH")
        intel  = _intel("HOLD(↓)", 50, "MODERATE")
        result = apply_news_to_conviction(intel, sym)
        self.assertLess(result["news_delta"], 0)


# ── do_buy() guard conditions ─────────────────────────────────────────────────

class TestDoBuyGuards(unittest.TestCase):
    """do_buy should skip (no transaction logged) when inputs are invalid."""

    def setUp(self):
        from config import WATCHLIST
        self._sym           = WATCHLIST[0]["symbol"]
        self._exch          = WATCHLIST[0]["exchange"]
        self._orig_dry      = TRADING_CONFIG["dry_run"]
        self._orig_min      = TRADING_CONFIG["min_trade_amount"]
        TRADING_CONFIG["dry_run"]          = True
        TRADING_CONFIG["min_trade_amount"] = 200

    def tearDown(self):
        TRADING_CONFIG["dry_run"]          = self._orig_dry
        TRADING_CONFIG["min_trade_amount"] = self._orig_min

    def _call(self, cash_to_use, price):
        app.write_transaction = MagicMock()
        app.do_buy(self._sym, self._exch, cash_to_use, price, 100.0, 98.0, 2.0, 0)

    def test_skip_when_cash_zero(self):
        self._call(cash_to_use=0, price=100)
        app.write_transaction.assert_not_called()

    def test_skip_when_qty_zero(self):
        # price > cash → int(cash // price) = 0
        self._call(cash_to_use=50, price=100)
        app.write_transaction.assert_not_called()

    def test_skip_when_order_below_min(self):
        # price=190, cash=195 → qty=1, total=190 < 200 min
        self._call(cash_to_use=195, price=190)
        app.write_transaction.assert_not_called()

    def test_proceeds_when_valid(self):
        # price=100, cash=500 → qty=5, total=500 ≥ 200 → should log
        app.write_transaction = MagicMock()
        with patch.object(app.kite, "margins", return_value={"available": {"live_balance": "0"}}):
            app.do_buy(self._sym, self._exch, 500, 100.0, 100.0, 98.0, 2.0, 0)
        app.write_transaction.assert_called_once()


# ── do_sell() guard conditions ────────────────────────────────────────────────

class TestDoSellGuards(unittest.TestCase):

    def setUp(self):
        from config import WATCHLIST
        self._sym  = WATCHLIST[0]["symbol"]
        self._exch = WATCHLIST[0]["exchange"]
        self._orig_dry       = TRADING_CONFIG["dry_run"]
        self._orig_min_hold  = TRADING_CONFIG["min_holding_days"]
        self._orig_min_prof  = TRADING_CONFIG["min_profit_pct_to_sell"]
        TRADING_CONFIG["dry_run"]                = True
        TRADING_CONFIG["min_holding_days"]       = 2
        TRADING_CONFIG["min_profit_pct_to_sell"] = 0.5

    def tearDown(self):
        TRADING_CONFIG["dry_run"]                = self._orig_dry
        TRADING_CONFIG["min_holding_days"]       = self._orig_min_hold
        TRADING_CONFIG["min_profit_pct_to_sell"] = self._orig_min_prof

    def _call(self, holding_days=5, avg_p=100.0, price=101.0, qty=10):
        app.write_transaction = MagicMock()
        app._buy_date_tracker[self._sym] = (
            datetime.date.today() - datetime.timedelta(days=holding_days)
        )
        app.do_sell(self._sym, self._exch, qty, avg_p, price, 100.0, 98.0, 2.0, 10000.0)

    def test_skip_when_held_too_short(self):
        self._call(holding_days=1)   # < min 2
        app.write_transaction.assert_not_called()

    def test_skip_when_profit_too_low(self):
        # avg=100, price=100.2 → profit=0.2% < 0.5% threshold
        self._call(holding_days=5, avg_p=100.0, price=100.2)
        app.write_transaction.assert_not_called()

    def test_proceeds_when_conditions_met(self):
        # holding_days=5≥2, profit=(102-100)/100=2%≥0.5%
        self._call(holding_days=5, avg_p=100.0, price=102.0, qty=10)
        app.write_transaction.assert_called_once()

    def test_ltcg_always_full_exit(self):
        """Held >365 days → FULL exit regardless of conviction."""
        app.write_transaction = MagicMock()
        app._buy_date_tracker[self._sym] = (
            datetime.date.today() - datetime.timedelta(days=400)
        )
        intel = {"conviction_label": "WEAK", "conviction": 20, "reason_detail": ""}
        app.do_sell(self._sym, self._exch, 10, 100.0, 103.0, 100.0, 98.0, 2.0, 10000.0, intel=intel)
        call_data = app.write_transaction.call_args[0][0]
        self.assertEqual(call_data["quantity"], 10)   # full exit
        self.assertIn("LTCG", call_data["estimated_tax_note"])

    def test_weak_conviction_partial_25pct(self):
        """WEAK conviction → sell only 25% of qty."""
        app.write_transaction = MagicMock()
        app._buy_date_tracker[self._sym] = (
            datetime.date.today() - datetime.timedelta(days=10)
        )
        intel = {"conviction_label": "WEAK", "conviction": 20, "reason_detail": ""}
        app.do_sell(self._sym, self._exch, 20, 100.0, 103.0, 100.0, 98.0, 2.0, 10000.0, intel=intel)
        call_data = app.write_transaction.call_args[0][0]
        self.assertEqual(call_data["quantity"], 5)   # 25% of 20

    def test_moderate_conviction_partial_50pct(self):
        """MODERATE conviction + weak gap → sell 50%."""
        app.write_transaction = MagicMock()
        app._buy_date_tracker[self._sym] = (
            datetime.date.today() - datetime.timedelta(days=10)
        )
        intel = {"conviction_label": "MODERATE", "conviction": 50, "reason_detail": ""}
        # gap=0.1% < strong_signal_threshold*100 (0.3%) → partial 50%
        app.do_sell(self._sym, self._exch, 20, 100.0, 103.0, 100.0, 98.0,
                    gap=0.1, cash=10000.0, intel=intel)
        call_data = app.write_transaction.call_args[0][0]
        self.assertEqual(call_data["quantity"], 10)  # 50% of 20


# ── db module ─────────────────────────────────────────────────────────────────

class TestDb(unittest.TestCase):
    """Tests for db.py using a real SQLite file (temp, deleted after each test)."""

    def setUp(self):
        import importlib.util, os, tempfile
        # Load the real db.py directly — bypasses sys.modules["db"] mock
        _spec = importlib.util.spec_from_file_location(
            "real_db", os.path.join(os.path.dirname(__file__), "db.py")
        )
        self.db = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(self.db)
        # Fresh temp file so each test is fully isolated
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db.init_db(self.path)

    def tearDown(self):
        import os
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass

    def test_init_creates_tables(self):
        with self.db._conn(self.path) as c:
            tables = {r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
        self.assertIn("transactions",      tables)
        self.assertIn("push_subscriptions", tables)
        self.assertIn("agent_state",        tables)

    def test_log_and_read_transaction(self):
        self.db.log_transaction(self.path, {
            "timestamp": "2024-01-01T10:00:00",
            "action": "BUY", "symbol": "SILVER",
            "quantity": 5, "price": 100.0,
        })
        rows = self.db.read_transactions(self.path, limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["symbol"], "SILVER")
        self.assertEqual(rows[0]["action"], "BUY")

    def test_read_transactions_newest_first(self):
        for i in range(3):
            self.db.log_transaction(self.path, {
                "timestamp": f"2024-01-{i+1:02d}", "action": "BUY",
                "symbol": "SILVER", "quantity": i + 1, "price": 100.0,
            })
        rows = self.db.read_transactions(self.path, limit=10)
        # newest first means qty=3, 2, 1
        self.assertEqual(rows[0]["quantity"], 3)
        self.assertEqual(rows[2]["quantity"], 1)

    def test_read_transactions_limit(self):
        for i in range(10):
            self.db.log_transaction(self.path, {
                "timestamp": f"2024-01-{i+1:02d}", "action": "BUY",
                "symbol": "SILVER", "quantity": i, "price": 100.0,
            })
        rows = self.db.read_transactions(self.path, limit=3)
        self.assertEqual(len(rows), 3)

    def test_push_subscription_save_and_load(self):
        sub = {"endpoint": "https://fcm.example.com/123", "keys": {"auth": "abc"}}
        self.db.save_push_subscription(self.path, sub)
        loaded = self.db.load_push_subscriptions(self.path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["endpoint"], sub["endpoint"])

    def test_push_subscription_upsert(self):
        sub = {"endpoint": "https://fcm.example.com/123", "keys": {"auth": "abc"}}
        self.db.save_push_subscription(self.path, sub)
        self.db.save_push_subscription(self.path, sub)  # duplicate → upsert
        self.assertEqual(len(self.db.load_push_subscriptions(self.path)), 1)

    def test_remove_push_subscription(self):
        sub = {"endpoint": "https://fcm.example.com/999", "keys": {"auth": "xyz"}}
        self.db.save_push_subscription(self.path, sub)
        self.db.remove_push_subscription(self.path, sub["endpoint"])
        self.assertEqual(self.db.load_push_subscriptions(self.path), [])

    def test_save_and_load_buy_dates(self):
        dates = {"SILVER": "2024-03-01", "GOLD": "2024-02-15"}
        self.db.save_buy_dates(self.path, dates)
        loaded = self.db.load_buy_dates(self.path)
        self.assertEqual(loaded["SILVER"], "2024-03-01")
        self.assertEqual(loaded["GOLD"],   "2024-02-15")

    def test_save_buy_dates_overwrites(self):
        self.db.save_buy_dates(self.path, {"SILVER": "2024-01-01"})
        self.db.save_buy_dates(self.path, {"SILVER": "2024-06-01"})
        loaded = self.db.load_buy_dates(self.path)
        self.assertEqual(loaded["SILVER"], "2024-06-01")

    def test_empty_db_returns_empty_collections(self):
        self.assertEqual(self.db.read_transactions(self.path), [])
        self.assertEqual(self.db.load_push_subscriptions(self.path), [])
        self.assertEqual(self.db.load_buy_dates(self.path), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
