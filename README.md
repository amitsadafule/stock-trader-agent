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

On your **Kite chart**, add **two** Moving Averages:
- MA 1: Period = `ema_green_period` (default 7), Type = EMA, Field = Close → colour **Green**
- MA 2: Period = `ema_red_period` (default 14), Type = EMA, Field = Close → colour **Red**

Set `candle_interval` in `config.py` to **match your chart's timeframe**:

| Kite chart timeframe | config.py value |
|---|---|
| 1 minute | `"minute"` |
| 5 minutes | `"5minute"` |
| 15 minutes | `"15minute"` |
| 1 hour | `"60minute"` |
| Daily | `"day"` ← recommended for ETFs |

For **daily candles** also set `check_interval_seconds: 1800` (check every 30 min — daily candles only finalise at end of day, no point checking every minute).

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

**Stocks** — Full per-stock detail with live updating conviction meter, intelligence grid, and plain-English "Why this signal?" reasoning box

**Trades** — Full transaction log: action, quantity, price, cash before/after, reason, conviction score, tax note

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
- **BUY**: EMA7 crosses **above** EMA14 (fast line breaks through slow line upward)
- **SELL**: EMA7 crosses **below** EMA14
- **HOLD(↑)**: EMA7 already above EMA14 — bullish trend in progress, no fresh crossover
- **HOLD(↓)**: EMA7 below EMA14 — bearish, waiting for reversal

### Signal Intelligence & Conviction Scoring

The agent doesn't treat all signals equally. Every tick it runs a full `analyse()` on each stock and computes a **conviction score (0–100)** from four factors:

| Factor | What it measures | Max pts |
|---|---|---|
| **Gap size** | How far apart are EMA7 and EMA14? Bigger = stronger trend | 35 |
| **Gap momentum** | Is the gap widening (accelerating) or narrowing (fading)? | ±25 |
| **Volume ratio** | Current volume vs 10-candle average. High vol = real move | 25 |
| **Trend age** | How many consecutive candles in this EMA alignment? | 15 |

Conviction drives position sizing automatically:

| Score | Label | Cash deployed on BUY | Shares sold on SELL |
|---|---|---|---|
| 65–100 | **STRONG** | 100% of allocated share | Full position (100%) |
| 38–64 | **MODERATE** | 75% — holds 25% back | Partial sell (50%) |
| 0–37 | **WEAK** | 50% — cautious sizing | Very cautious (25%) |

This means a steep, high-volume, established trend deploys full capital, while a thin crossover on low volume only risks half — automatically, with no manual intervention.

### "Why This Signal?" Explanation

Every stock card in the Stocks tab shows a live **plain-English reasoning box** that explains:
- Why the current signal (BUY/SELL/HOLD) was generated in terms of EMA positions
- Whether the gap is widening or narrowing — and what that means for trend strength
- How many candles the trend has been active and the confidence level that implies
- Whether volume confirms or undermines the signal
- Exactly what will happen if the signal triggers (how much cash/shares)
- What price movement would flip the signal to something different

This updates every 15 seconds alongside all other live data.

Example (HOLD↑ on SILVERIETF):
```
✅  Fast EMA (263.48) is above slow EMA (255.92) — bullish trend ongoing, no fresh crossover
📏  Gap: 0.46% and widening ↑ (+0.012% this candle) — trend is accelerating
📅  EMA alignment held for 5 candles — young trend, moderate confidence
📊  Volume: 1.8x above average — moderate confirmation
🎯  Conviction 58/100 (MODERATE) → deploying 75% of cash if signal triggers
🔮  Next: Signal becomes SELL if fast EMA crosses below slow EMA
```

### Startup Catch-Up Buy
If you start the agent with **zero holdings** and EMA7 is already above EMA14 (bullish trend already in progress), the agent buys immediately on startup rather than waiting for the next crossover. This is logged as `CATCHUP-BUY`. Conviction scoring still applies — a weak catch-up signal uses only 50% of cash.

If EMA7 is below EMA14 on startup (bearish), it waits for a proper upward crossover before buying.

### State Recovery After Restart
When the app restarts after being offline:
1. Reads `transactions.csv` to find your last action and buy date
2. Calls `kite.holdings()` to get actual shares in demat (source of truth)
3. Warns if log and demat disagree
4. Restores `buy_date` so min-holding-days and LTCG logic still work correctly

---

## 💰 Multi-Stock Allocation

When multiple stocks signal BUY at the same time, cash is allocated using `multi_buy_strategy`:

| Strategy | Behaviour |
|---|---|
| `weighted` | Split proportionally by watchlist `weight` (recommended) |
| `equal` | Split equally among all signalling stocks |
| `top1` | Put all cash into the stock with the highest conviction score |

### Conviction Scaling on Top of Weights

Even after the weight-based split, each stock's slice is scaled down by its conviction level:

```
Example: ₹10,000 cash | SILVERIETF (w=60, STRONG) + GOLDETF (w=30, WEAK)

Weight split:   SILVER=₹5,333   GOLD=₹4,000   (₹667 reserve always held)
Conviction:     SILVER × 1.0    GOLD × 0.5
Final:          SILVER=₹5,333   GOLD=₹2,000   (₹2,667 saved as dry powder)
```

A WEAK signal never gets the full weight-share — it's automatically de-risked.

### Smart Low-Cash Fallback
If splitting cash would give any stock less than `min_trade_amount`, the agent **automatically concentrates** into fewer stocks — dropping the lowest-conviction stock first — until every remaining stock gets a viable allocation.

Example with ₹496 cash and two BUY signals:
```
SILVERIETF weight=60 gap=0.45% conviction=45 (MODERATE) → gets ₹255 ← below min
GOLDETF    weight=40 gap=0.84% conviction=62 (MODERATE) → gets ₹192 ← below min

→ Fallback: drop SILVERIETF (lower conviction), put ₹447 into GOLDETF
→ GOLDETF: qty=2 @ ₹152 = ₹304 ✅
```

**Minimum cash needed to trade:**
- Single stock: `price_per_share` + 10% reserve. E.g. SILVERIETF at ₹263 → need ~₹300+
- Two stocks split by weight: each must get ≥ `min_trade_amount` after conviction scaling. Around ₹1,200+ for two typical ETFs.

---

## 💸 Sell Rules & Tax Logic

Sells are conviction-aware. The same SELL crossover produces a different action depending on signal strength:

| Condition | Action | Reason |
|-----------|--------|--------|
| SELL signal — STRONG conviction | Sell **100%** | Trend reversal is confirmed |
| SELL signal — MODERATE conviction + gap ≥ 0.3% | Sell **100%** | Strong gap reinforces conviction |
| SELL signal — MODERATE conviction | Sell **50%** | May be temporary pullback |
| SELL signal — WEAK conviction | Sell **25%** | Likely noise, very conservative |
| Held ≥ 365 days | Sell **100%** regardless | LTCG @ 10% applies |
| Profit < 0.5% | **Skip sell** | Would not cover STT + brokerage |
| Held < `min_holding_days` | **Skip sell** | Prevents wash trades |

Tax notes are recorded on every transaction:
- **STCG (Short Term Capital Gain)**: 15% if sold within 1 year
- **LTCG (Long Term Capital Gain)**: 10% on gains above ₹1L if held over 1 year

All trades use **CNC (delivery)** product — not MIS/intraday.

---

## 🧪 Dry Run Mode

`dry_run: True` in `config.py` — signals and logic run fully, but `kite.place_order()` is never called. All transactions are logged with `[DRY]` prefix. A yellow banner shows in the app. The conviction scoring, position sizing, and "Why?" reasoning all still work exactly as in live mode.

Set `dry_run: False` only when you've verified:
- EMA values in the app match your Kite chart
- Signals fire at expected crossover points
- Conviction scores look reasonable for recent candles

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
- Ensure you have **both** EMA7 and EMA14 plotted on your Kite chart (not just one)
- After changing interval, restart app.py
- Common mistake: chart is on daily candles but config has `5minute` (or vice versa)

**Catch-up buy not firing:**
- Check `agent.log` for lines starting with `🚀` (eligible) or `ℹ️` (not eligible + reason)
- Common reasons: market closed at startup, EMA bearish, cash too low
- If eligible but still skipped: check for `⚠️ BUY skipped` with the exact reason

**Buy skipped — cash too low:**
- Check `agent.log` for `💡 Low cash fallback` or `⚠️ BUY skipped`
- Minimum needed: `price_per_share × 1.1` (10% reserve). Add more funds.
- Or lower `min_cash_reserve_pct` to `0.05` in config.py
- Or lower `min_trade_amount` to match your actual order sizes

**Conviction score seems wrong:**
- Check `agent.log` — every tick logs the full score breakdown:
  `conviction=58/100(MODERATE) | gap 0.46%→16pt | ↑momentum +0.012%→+1pt | vol 1.8x→10pt | trend 5c→4pt`
- Low volume (ETFs outside market hours) can suppress the score significantly
- Very long trend ages (30+ candles) get a slight penalty for potential exhaustion

**Sell was too small / too large:**
- Conviction-aware sizing: STRONG → 100%, MODERATE → 50%, WEAK → 25%
- Check `agent.log` for `SELL (FULL)` / `SELL (PARTIAL-50%)` / `SELL (PARTIAL-25%)`
- If you want more aggressive sells, lower the conviction thresholds in `analyse()` in app.py

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
- This is not financial advice. All trading decisions are made by the algorithm based on EMA crossovers and conviction scoring. Past performance does not guarantee future results.
