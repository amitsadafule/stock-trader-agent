# Trading Agent — Exhaustive Case Reference

Every decision path the agent can take, with the exact conditions and outcome.
Config defaults referenced: `ema_green=7`, `ema_red=14`, `min_holding_days=1`,
`min_profit_pct_to_sell=0.5%`, `strong_signal_threshold=0.003` (=0.3%),
`min_trade_amount=200`, `min_cash_reserve_pct=10%`.

---

## 1. Signal Detection (`analyse()`)

The core signal is determined by the last two EMA values.

| # | Condition (prev candle → curr candle) | Signal | Notes |
|---|---|---|---|
| 1.1 | EMA7 was ≤ EMA14, now EMA7 > EMA14 | **BUY** | Fresh upward crossover |
| 1.2 | EMA7 was ≥ EMA14, now EMA7 < EMA14 | **SELL** | Fresh downward crossover |
| 1.3 | EMA7 > EMA14 both candles (no crossover) | **HOLD(↑)** | Bullish trend, no new entry |
| 1.4 | EMA7 < EMA14 both candles (no crossover) | **HOLD(↓)** | Bearish trend, no new entry |
| 1.5 | EMA7 == EMA14 both candles | **HOLD(↓)** | Treated as non-bullish |

---

## 2. Conviction Scoring (`analyse()`)

Scored 0–100 from four independent factors. Label and `invest_pct` derived from final score.

### Factor breakdown

| Factor | Max pts | Rule |
|---|---|---|
| Gap size | 35 | `min(35, int(gap_pct × 17.5))` — 0.2%→7pt, 0.5%→17pt, 1%→28pt, 2%+→35pt |
| Momentum | +25 / −15 | Gap widening → +pts; narrowing → penalty |
| Volume | 25 | `min(25, (vol_ratio−1) × 12.5)` — needs >1× avg to score |
| Trend age | 15 | 1 candle→1pt, 10c→7pt, 20c→15pt, 30c+→decay to 5pt (overextended) |

### Conviction tiers

| Score | Label | `invest_pct` | Effect |
|---|---|---|---|
| 65–100 | **STRONG** | 1.00 (100%) | Full position for BUY; full exit for SELL |
| 38–64 | **MODERATE** | 0.75 (75%) | 75% of allocated cash for BUY; 50% exit for SELL (unless strong gap) |
| 0–37 | **WEAK** | 0.50 (50%) | Half position for BUY; only 25% exit for SELL |

---

## 3. BUY Cases

### 3A. Signal fires and trade executes

| # | Scenario | Conviction | Cash | Outcome |
|---|---|---|---|---|
| 3.1 | EMA7 crosses above EMA14 | STRONG (65+) | Sufficient | BUY 100% of allocated slice |
| 3.2 | EMA7 crosses above EMA14 | MODERATE (38–64) | Sufficient | BUY 75% of allocated slice |
| 3.3 | EMA7 crosses above EMA14 | WEAK (<38) | Sufficient | BUY 50% of allocated slice |
| 3.4 | Startup: EMA7 > EMA14, zero holdings | Any | Sufficient | **CATCHUP-BUY** — buys immediately, no need to wait for fresh crossover |
| 3.5 | Startup: EMA7 > EMA14, already holding shares | Any | — | No catchup (already invested) |
| 3.6 | Startup: EMA7 < EMA14, zero holdings | Any | — | No catchup — waits for a real BUY crossover |

### 3B. BUY skipped (guards)

| # | Guard | Condition | Log message |
|---|---|---|---|
| 3.7 | Zero cash allocated | `cash_to_use ≤ 0` | `BUY skipped — ₹0 allocated` |
| 3.8 | Quantity rounds to zero | `price > cash_to_use` | `BUY skipped — qty=0` |
| 3.9 | Order below minimum | `qty × price < min_trade_amount` | `BUY skipped — order ₹X < min ₹200` |

### 3C. Opening buffer blocks BUY

| # | Scenario | Outcome |
|---|---|---|
| 3.10 | BUY signal fires within first 30 min after 9:15 AM | Signal noted in UI, **trade blocked** |
| 3.11 | BUY during buffer + STRONG conviction + gap widening + vol ≥1.5× + candle range ≤2% | **Early-entry override** — trade executes despite buffer |
| 3.12 | BUY during buffer + any condition NOT met (WEAK, narrowing gap, low vol, spike) | Blocked with specific reason logged |
| 3.13 | BUY after buffer, opening candle range still >1.5% | Blocked 10 more min ("still choppy") |
| 3.14 | BUY after buffer, opening range calm | Normal trade execution |

---

## 4. Multi-Stock BUY Allocation

When two or more stocks signal BUY simultaneously.

### 4A. Allocation strategies

| # | Strategy (`multi_buy_strategy`) | Cash split rule |
|---|---|---|
| 4.1 | `weighted` (default) | Split investable cash by watchlist `weight` ratio among buying stocks |
| 4.2 | `equal` | Split evenly among buying stocks |
| 4.3 | `top1` | All investable cash to the stock with highest conviction score |

### 4B. Conviction scaling (all strategies)

After the base split, each stock's slice is multiplied by its `invest_pct`:

| # | conviction_label | invest_pct | Example: ₹10,000 base slice |
|---|---|---|---|
| 4.4 | STRONG | 1.00 | ₹10,000 deployed |
| 4.5 | MODERATE | 0.75 | ₹7,500 deployed, ₹2,500 held back |
| 4.6 | WEAK | 0.50 | ₹5,000 deployed, ₹5,000 held back |

### 4C. Low-cash fallback

| # | Scenario | Outcome |
|---|---|---|
| 4.7 | Split would give any stock < `min_trade_amount` | **Drop lowest-conviction stock(s)** until remaining can each get ≥ min |
| 4.8 | Even one stock can't reach min after all others dropped | **Concentrate entire investable cash** into highest-conviction stock |
| 4.9 | Cash ≤ 0 | All BUY signals skipped via guard 3.7 |

### 4D. Reserve

| # | Rule |
|---|---|
| 4.10 | Total deployed never exceeds `(1 − min_cash_reserve_pct) × available_cash` (default: 90%) |

---

## 5. SELL Cases

### 5A. Sell executes

| # | Scenario | Conviction | Outcome |
|---|---|---|---|
| 5.1 | EMA7 crosses below EMA14 | STRONG | Full exit (100% of holdings) |
| 5.2 | EMA7 crosses below EMA14, gap ≥ 0.3% | MODERATE | Full exit (strong gap confirms signal) |
| 5.3 | EMA7 crosses below EMA14, gap < 0.3% | MODERATE | Partial exit — **50%** of holdings |
| 5.4 | EMA7 crosses below EMA14 | WEAK | Partial exit — **25%** of holdings (signal may be noise) |
| 5.5 | Any SELL signal, holding days ≥ 365 | Any | **Full exit** regardless of conviction (LTCG applies, book gains) |

### 5B. SELL skipped (guards)

| # | Guard | Condition | Log message |
|---|---|---|---|
| 5.6 | Not held long enough | `holding_days < min_holding_days` | `skip sell — held Xd < min Yd` |
| 5.7 | Profit insufficient | `(price − avg_p) / avg_p < 0.5%` | `skip sell — profit X% < threshold` |
| 5.8 | Zero holdings | `qty_held == 0` | No SELL added to signal list (filtered before `do_sell`) |

### 5C. Opening buffer also blocks SELL

| # | Scenario | Outcome |
|---|---|---|
| 5.9 | SELL fires during opening buffer | Signal noted in UI, **sell also blocked** (prevents selling into an opening spike) |

---

## 6. News Sentiment Adjustments

News runs in a background thread, never blocking trades. It adjusts conviction after `analyse()`.

### 6A. Impact matrix

| # | News label | Signal direction | Delta | Effect |
|---|---|---|---|---|
| 6.1 | BULLISH (score ≥ 60) | BUY / HOLD(↑) | **+20 pt** | Strong confirmation |
| 6.2 | BULLISH (score 30–59) | BUY / HOLD(↑) | **+10 pt** | Moderate confirmation |
| 6.3 | BULLISH (score < 30) | BUY / HOLD(↑) | **+5 pt** | Weak confirmation |
| 6.4 | BULLISH | SELL | **−5 to −10 pt** | News contradicts sell (reduce exit size) |
| 6.5 | BEARISH (score ≤ −60) | SELL / HOLD(↓) | **−20 pt** | Confirms bearish signal |
| 6.6 | BEARISH (score −30 to −59) | SELL / HOLD(↓) | **−10 pt** | Moderate confirmation |
| 6.7 | BEARISH (score > −30) | SELL / HOLD(↓) | **−5 pt** | Weak confirmation |
| 6.8 | BEARISH | BUY | **−3 to −15 pt** | News contradicts buy (reduce position size) |
| 6.9 | NEUTRAL | Any | **0 pt** | No effect |
| 6.10 | News not yet fetched | Any | **0 pt** | Labelled "pending" — no stale NEUTRAL applied |

### 6B. Score bounds

Conviction is always clamped `[0, 100]` after news adjustment.
A conviction label change (e.g. MODERATE→WEAK after bearish news) directly reduces `invest_pct`.

---

## 7. HOLD Cases

| # | Scenario | Outcome |
|---|---|---|
| 7.1 | HOLD(↑), zero holdings | No action. Catchup logic did not fire (already allocated, or bearish at startup). |
| 7.2 | HOLD(↑), already holding | No action. Position maintained. |
| 7.3 | HOLD(↓), zero holdings | No action. Waiting for BUY crossover. |
| 7.4 | HOLD(↓), already holding | No action. Sell only fires on a SELL crossover. |
| 7.5 | Market closed | All signals show "Market Closed". Holdings and cash still refreshed every 5 min. |

---

## 8. State Recovery (restart after being offline)

Runs once at startup before the first trading tick.

| # | Step | Source of truth |
|---|---|---|
| 8.1 | Read last `BUY`/`SELL` per symbol from SQLite | `transactions` table |
| 8.2 | Restore `buy_date` for each symbol | Used for `min_holding_days` and LTCG logic |
| 8.3 | Fetch actual holdings from `kite.holdings()` | Demat (T+2 settled) |
| 8.4 | Compare log holdings vs demat holdings | Warn if mismatch; demat is used as truth |
| 8.5 | Scan candle history for missed signals while offline | Shown in UI under "Recovery" — **not auto-traded** |
| 8.6 | Push notification sent | "Agent Resumed — offline Xd" |

---

## 9. Tax Notes (logged on every transaction)

| # | Condition | Tax note logged |
|---|---|---|
| 9.1 | Sold, held < 365 days | `STCG 15% — [sell type]. Gap X%, conviction=Y/100, held Zd.` |
| 9.2 | Sold, held ≥ 365 days | `LTCG 10% — held Xd.` |
| 9.3 | Bought (any) | `CNC delivery. STCG 15% if <1yr, LTCG 10% if >1yr.` |

---

## 10. Dry Run vs Live

| # | Setting | Behaviour |
|---|---|---|
| 10.1 | `dry_run: True` (default) | `kite.place_order()` is **never called**. All orders logged with `[DRY]` prefix. Yellow UI banner shown. |
| 10.2 | `dry_run: False` | Real CNC market orders placed via Kite. Push notification sent on every trade. |

---

## 11. Decision Flow Diagram

```
Each trading tick (every check_interval_seconds):
│
├─ Market closed? → fetch holdings/cash, wait 5min, continue
│
├─ Not logged in? → stop agent, push "session expired"
│
├─ Fetch cash + holdings + positions
│
├─ For each watchlist stock:
│   ├─ get_candles() → analyse() → conviction score
│   ├─ apply_news_to_conviction() → adjusted score
│   ├─ Signal == BUY or CATCHUP-BUY?  → add to buy_signals[]
│   └─ Signal == SELL and qty > 0?    → add to sell_signals[]
│
├─ opening_volatility_status():
│   ├─ blocked=True?  → log signals, skip execution
│   └─ blocked=False? → proceed
│       ├─ Execute sell_signals (frees cash first)
│       └─ allocate_cash(buy_signals) → execute each do_buy()
│
└─ Wait check_interval_seconds, repeat
```

---

## 12. Example Walkthroughs

### Example A — SILVERIETF STRONG BUY, GOLDETF WEAK BUY, ₹10,000 cash

```
SILVERIETF: BUY signal, conviction=72 (STRONG), invest_pct=1.0, weight=40
GOLDETF:    BUY signal, conviction=30 (WEAK),   invest_pct=0.5, weight=30

Investable = ₹10,000 × 0.90 = ₹9,000

Weight split (among buying stocks, total weight=70):
  SILVER = ₹9,000 × 40/70 = ₹5,143
  GOLD   = ₹9,000 × 30/70 = ₹3,857

Conviction scaling:
  SILVER × 1.0 = ₹5,143  (STRONG → deploy all)
  GOLD   × 0.5 = ₹1,929  (WEAK → deploy half)

Result: SILVER gets ₹5,143 | GOLD gets ₹1,929 | ₹2,928 kept as dry powder
```

### Example B — Low cash fallback, ₹600 cash

```
SILVERIETF: BUY, conviction=80 (STRONG), weight=50
GOLDETF:    BUY, conviction=25 (WEAK),   weight=50
min_trade_amount = ₹500

Investable = ₹600 × 0.90 = ₹540

Equal split with conviction:
  SILVER = ₹270 × 1.0 = ₹270  ← below ₹500 min
  GOLD   = ₹270 × 0.5 = ₹135  ← below ₹500 min

Fallback: drop GOLD (lowest conviction).
  SILVER alone = ₹540 × 1.0 = ₹540 ≥ ₹500 → viable ✓

Result: SILVER gets ₹540 | GOLD dropped
```

### Example C — SELL with partial conviction

```
SILVERIETF: SELL signal, held 45 days, bought @ ₹250, now ₹260
Profit = (260−250)/250 = 4% ≥ 0.5% threshold ✓
Holding days = 45 ≥ 1 min ✓
Conviction = 42 (MODERATE)
Gap = 0.15% < 0.3% strong_signal_threshold

→ Partial sell (MODERATE + weak gap): sell 50% of holdings
→ Tax note: STCG 15% (held <365 days)
```

### Example D — SELL blocked by minimum hold

```
GOLDETF: SELL signal, held 0 days (bought same day), profit 2%
min_holding_days = 1

→ Skip sell — "held 0d < min 1d"
→ Prevents wash trades and brokerage waste
```
