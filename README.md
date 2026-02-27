# 📱 Zerodha Trading Agent — Android PWA

A fully automated multi-stock EMA crossover trading bot that installs as a native app on your Android phone, with push notifications for trades, errors, and daily login reminders.

---

## 🗂️ Project Files

```
zerodha_app/
├── app.py               ← Main server (run this)
├── config.py            ← Your credentials, watchlist & settings
├── generate_vapid.py    ← One-time push notification setup
├── requirements.txt
├── templates/
│   ├── index.html       ← PWA dashboard UI
│   └── error.html
└── static/
    └── icons/           ← App icons
```

---

## ⚙️ Setup (One Time)

### 1. Install Termux on Android
Download from **F-Droid** (NOT Play Store — Play Store version is outdated):
→ https://f-droid.org/packages/com.termux/

```bash
pkg update && pkg upgrade -y
pkg install python -y
pip install -r requirements.txt
```

### 2. Create a Kite Developer App
1. Go to https://developers.kite.trade → create an app
2. Enable the **Historical Data** permission (required for EMA calculation)
3. Set redirect URL to: `http://YOUR_PHONE_IP:8080/callback`
   (the app prints your IP when it starts)
4. Copy your `api_key` and `api_secret`

### 3. Configure `config.py`

```python
ZERODHA_CONFIG = {
    "api_key":    "your_kite_api_key",
    "api_secret": "your_kite_api_secret",
    "user_id":    "AB1234",
}

WATCHLIST = [
    {"symbol": "SILVERIETF", "exchange": "NSE", "weight": 60},
    {"symbol": "GOLDETF",    "exchange": "NSE", "weight": 40},
]
```

### 4. Match chart settings to config

> ⚠️ **Critical** — the agent's EMA values will only match your Kite chart if both use the same candle interval and EMA periods.

On your **Kite chart**, add two Moving Averages:
- MA 1: Period = `ema_green_period` (default 7), Type = EMA, Field = Close → set colour **Green**
- MA 2: Period = `ema_red_period` (default 14), Type = EMA, Field = Close → set colour **Red**

Set `candle_interval` in `config.py` to **match your chart's timeframe**:

| Kite chart timeframe | config.py value |
|---|---|
| 1 minute | `"minute"` |
| 5 minutes | `"5minute"` |
| 15 minutes | `"15minute"` |
| 1 hour | `"60minute"` |
| Daily | `"day"` ← recommended for ETFs |

For **daily candles** also set `check_interval_seconds: 1800` (check every 30 min — no point checking every minute when a new candle only forms at end of day).

### 5. Generate push notification keys (once)
```bash
python generate_vapid.py
```

### 6. Run the server
```bash
python app.py
```

### 7. Install as Android app
1. Open **Chrome** on your phone → go to `http://localhost:8080`
2. Chrome shows an "Add to Home Screen" banner — tap it
3. Or: Chrome menu (⋮) → "Add to Home Screen"
4. App icon appears on home screen and opens without browser UI

---

## 📊 Dashboard Tabs

**Dashboard** — Agent start/stop, live signals overview for all stocks, cash balance, allocation weights chart

**Stocks** — Per-stock detail: EMA7 value, EMA14 value, gap %, current price, holdings, avg buy price

**Trades** — Full transaction log: action, quantity, price, cash before/after, reason, tax note

**Settings** — All active config values (read-only — edit config.py to change)

---

## 🔔 Push Notifications

| Event | When |
|-------|------|
| 🔑 Login Reminder | 8:45 AM every day if session expired |
| ✅ Login Success | After completing Zerodha login |
| 🤖 Agent Started | On start (includes recovery summary if offline) |
| 🟢 BUY executed | Every buy order (real or dry run) |
| 🔴 SELL executed | Every sell order |
| 📊 Multi-buy | When 2+ stocks signal BUY simultaneously |
| 🔄 Agent Resumed | On restart after being offline |
| ❌ Error | Any exception in the trading loop |

Enable by tapping **"Enable Notifications"** on the dashboard.

---

## 🤖 Trading Strategy

### EMA Crossover Signal
- **BUY**: EMA7 crosses **above** EMA14 (fast line crosses up through slow line)
- **SELL**: EMA7 crosses **below** EMA14
- **HOLD**: No crossover — agent waits

### Startup Catch-Up Buy
If you start the agent with **zero holdings** and EMA7 is already above EMA14 (bullish trend already in progress), the agent buys immediately rather than waiting for the next crossover. This is logged as `CATCHUP-BUY`.

If EMA7 is below EMA14 on startup (bearish), it waits for a proper upward crossover before buying.

### State Recovery After Restart
When the app restarts after being offline (2–4+ days):
1. Reads `transactions.csv` to find your last action and buy date
2. Calls `kite.holdings()` to get actual shares in demat (source of truth)
3. Warns if log and demat disagree
4. Scans candle history to find any crossover signals missed while offline (shown in UI — not auto-traded)
5. Restores `buy_date` so min-holding-days and LTCG logic still work correctly

---

## 💰 Multi-Stock Allocation

When multiple stocks signal BUY at the same time, cash is allocated using `multi_buy_strategy`:

| Strategy | Behaviour |
|---|---|
| `weighted` | Split proportionally by watchlist `weight` (recommended) |
| `equal` | Split equally among all signalling stocks |
| `top1` | Put all cash into the stock with the biggest EMA gap |

### Smart Low-Cash Fallback
If splitting cash would give any stock less than `min_trade_amount` (₹500 by default), the agent **automatically concentrates** into fewer stocks — dropping the weakest signal first — until every remaining stock gets a viable amount.

Example with ₹496 cash and two BUY signals:
```
SILVERIETF weight=60 gap=0.45%  → would get ₹255  ← below min ₹500
GOLDETF    weight=40 gap=0.84%  → would get ₹192  ← below min ₹500

→ Fallback: drop SILVERIETF (weaker gap), put ₹447 into GOLDETF (stronger gap)
→ GOLDETF: qty=2 @ ₹152 = ₹304 ✅
```

**Minimum cash needed to trade:**
- Single stock: `price_per_share` + 10% reserve. E.g. SILVERIETF at ₹263 → need ~₹300+
- Two stocks: enough for each to get ≥ ₹500 allocation after 10% reserve. E.g. ₹1,200+ for two stocks

---

## 💸 Sell Rules & Tax Logic

| Condition | Action | Reason |
|-----------|--------|--------|
| EMA crossover gap ≥ 0.3% | Sell **100%** | Strong signal |
| EMA crossover gap < 0.3% | Sell **50%** | Weak signal — reduce STCG exposure |
| Held ≥ 365 days | Sell **100%** | LTCG @ 10% applies |
| Profit < 0.5% | **Skip sell** | Would not cover STT + brokerage |
| Held < `min_holding_days` | **Skip sell** | Prevents wash trades |

Tax notes are recorded on every transaction:
- **STCG (Short Term Capital Gain)**: 15% if sold within 1 year
- **LTCG (Long Term Capital Gain)**: 10% on gains above ₹1L if held over 1 year

All trades use **CNC (delivery)** product — not MIS/intraday.

---

## 🧪 Dry Run Mode

`dry_run: True` in `config.py` (default) — signals and logic run fully, but `kite.place_order()` is never called. All transactions are logged with `[DRY]` prefix. A yellow banner shows in the app.

Set `dry_run: False` only when you've verified the signals match your chart.

---

## 🏠 Keep Running in Background (Termux)

```bash
pkg install tmux
tmux new -s trader
python app.py
# Detach: press Ctrl+B then D
# Re-attach later: tmux attach -t trader
```

---

## 🐛 Debugging

**EMA values don't match Kite chart:**
- Ensure `candle_interval` in config matches your chart's timeframe exactly
- Ensure you have **both** EMA7 and EMA14 plotted on your chart (not just one)
- After changing interval, restart app.py

**Catch-up buy not firing:**
- Check `agent.log` for lines starting with `🚀` (eligible) or `ℹ️` (not eligible + reason)
- Common reasons: market closed, EMA bearish at startup, cash too low

**Buy skipped — cash too low:**
- Check `agent.log` for `💡 Low cash fallback` or `⚠️ BUY skipped`
- Minimum needed: price of 1 share × 1.1 (10% reserve). Add more funds.
- Or lower `min_cash_reserve_pct` to `0.05` in config.py

**PermissionException on historical data:**
- Enable "Historical Data" permission on your Kite developer app
- Delete `access_token.txt` and re-login

**Session expired mid-day:**
- Zerodha tokens expire daily at midnight
- Agent stops and sends a push notification
- Re-login the next morning before 9:15 AM

---

## ⚠️ Notes

- Zerodha sessions expire daily — push notification sent at 8:45 AM if re-login needed
- `access_token.txt` is auto-validated on startup and deleted if expired
- Tested on Android Chrome; Firefox for Android also works
- iOS: Safari supports "Add to Home Screen" but **does not support push notifications** (Apple limitation)
