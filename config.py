# ============================================================
# ZERODHA TRADING AGENT — CONFIGURATION
# ============================================================

ZERODHA_CONFIG = {
    "api_key":    "your_kite_api_key",
    "api_secret": "your_kite_api_secret",
    "user_id":    "your_kite_user_id",
    "access_token": None,
}

# ── Multi-stock watchlist ─────────────────────────────────
# Add as many stocks as you want. Each gets its own EMA signal.
# Allocation weights decide how cash is split when multiple BUY signals fire.
# Weights are relative — {60, 40} means 60% to first, 40% to second.
WATCHLIST = [
    {"symbol": "SILVERIETF",  "exchange": "NSE", "weight": 40},
    {"symbol": "GOLDETF",      "exchange": "NSE", "weight": 30},
]

TRADING_CONFIG = {
    # ── EMA crossover (applies to all stocks) ──────────────
    "ema_green_period": 7,
    "ema_red_period":   14,
    # Candle interval — MUST match your Kite chart timeframe.
    # Options: "minute" "3minute" "5minute" "10minute" "15minute"
    #          "30minute" "60minute" "day"
    "candle_interval":  "day",

    "lookback_candles": 60,

    # ── Capital allocation ─────────────────────────────────
    # When multiple stocks signal BUY at the same time, total cash is split
    # by their relative weights (only among the signalling stocks).
    # Strategy options:
    #   "weighted"    — split by watchlist weight (recommended)
    #   "equal"       — split equally among all buy signals
    #   "top1"        — only buy the stock with the strongest EMA gap
    "multi_buy_strategy": "weighted",

    # Reserve at least this % of cash — never go fully all-in
    "min_cash_reserve_pct": 0.10,    # keep 10% cash as reserve always
    "max_invest_pct":        0.90,   # invest at most 90% of available cash
    "min_trade_amount":      200,    # skip if order < ₹500

    # ── Sell rules ──────────────────────────────────────────
    "min_holding_days":        1,
    "min_profit_pct_to_sell":  0.5,
    "partial_sell_pct":        0.50,
    "strong_signal_threshold": 0.003,
    "stcg_holding_days":       365,

    # ── Timing ─────────────────────────────────────────────
    "check_interval_seconds": 1800,

    # ── Dry run ─────────────────────────────────────────────
    "dry_run": True,   # Set False to place real orders
}

PATHS = {
    "transaction_log":   "transactions.csv",
    "token_file":        "access_token.txt",
    "vapid_private_key": "vapid_private.pem",
    "vapid_public_key":  "vapid_public.txt",
    "state_file":        "agent_state.json",
}
