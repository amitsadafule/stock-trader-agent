"""
SQLite persistence layer for the Zerodha Trading Agent.

Replaces:
  - transactions.csv  → 'transactions' table
  - agent_state.json  → 'agent_state' table (key-value)
  - in-memory list    → 'push_subscriptions' table

Thread-safe: each call opens its own connection; WAL mode prevents
lock contention between the Flask thread and the agent loop thread.
"""

import csv
import json
import logging
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)

# Column order must match the CSV header exactly for migration to work.
FIELDS = [
    "timestamp", "action", "symbol", "quantity", "price", "total_value",
    "available_cash_before", "available_cash_after", "holdings_before",
    "holdings_after", "reason", "ema_green", "ema_red", "holding_days",
    "estimated_tax_note",
]

_CREATE_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS transactions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp             TEXT    NOT NULL,
    action                TEXT,
    symbol                TEXT,
    quantity              INTEGER,
    price                 REAL,
    total_value           REAL,
    available_cash_before REAL,
    available_cash_after  REAL,
    holdings_before       INTEGER,
    holdings_after        INTEGER,
    reason                TEXT,
    ema_green             REAL,
    ema_red               REAL,
    holding_days          INTEGER,
    estimated_tax_note    TEXT
);

CREATE TABLE IF NOT EXISTS push_subscriptions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint          TEXT    UNIQUE NOT NULL,
    subscription_json TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _conn(db_path: str) -> sqlite3.Connection:
    c = sqlite3.connect(db_path, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db(db_path: str, csv_path: str = None) -> None:
    """Create tables (idempotent). Migrate from CSV on first run if present."""
    with _conn(db_path) as c:
        c.executescript(_CREATE_SQL)

    if csv_path and Path(csv_path).exists():
        with _conn(db_path) as c:
            count = c.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        if count == 0:
            _migrate_csv(db_path, csv_path)


def _migrate_csv(db_path: str, csv_path: str) -> None:
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return
        ph = ", ".join(["?"] * len(FIELDS))
        col = ", ".join(FIELDS)
        with _conn(db_path) as c:
            c.executemany(
                f"INSERT OR IGNORE INTO transactions ({col}) VALUES ({ph})",
                [[r.get(f, "") for f in FIELDS] for r in rows],
            )
        log.info(f"🗄️  Migrated {len(rows)} transactions from {csv_path} → SQLite")
    except Exception as e:
        log.warning(f"CSV migration skipped: {e}")


# ── Transactions ──────────────────────────────────────────────────────────────

def log_transaction(db_path: str, data: dict) -> None:
    ph  = ", ".join(["?"] * len(FIELDS))
    col = ", ".join(FIELDS)
    with _conn(db_path) as c:
        c.execute(
            f"INSERT INTO transactions ({col}) VALUES ({ph})",
            [data.get(f, "") for f in FIELDS],
        )


def read_transactions(db_path: str, limit: int = 50) -> list[dict]:
    """Return the most recent `limit` transactions, newest first."""
    if not Path(db_path).exists():
        return []
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT * FROM transactions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ── Push subscriptions ────────────────────────────────────────────────────────

def save_push_subscription(db_path: str, sub: dict) -> None:
    endpoint = sub.get("endpoint", "")
    if not endpoint:
        return
    with _conn(db_path) as c:
        c.execute(
            "INSERT OR REPLACE INTO push_subscriptions (endpoint, subscription_json)"
            " VALUES (?, ?)",
            (endpoint, json.dumps(sub)),
        )


def remove_push_subscription(db_path: str, endpoint: str) -> None:
    if not endpoint:
        return
    with _conn(db_path) as c:
        c.execute("DELETE FROM push_subscriptions WHERE endpoint = ?", (endpoint,))


def load_push_subscriptions(db_path: str) -> list[dict]:
    if not Path(db_path).exists():
        return []
    with _conn(db_path) as c:
        rows = c.execute("SELECT subscription_json FROM push_subscriptions").fetchall()
    result = []
    for row in rows:
        try:
            result.append(json.loads(row[0]))
        except Exception:
            pass
    return result


# ── Agent state (buy dates) ───────────────────────────────────────────────────

def save_buy_dates(db_path: str, dates: dict) -> None:
    """Persist {symbol: date_str_or_date} mapping."""
    payload = json.dumps({k: str(v) for k, v in dates.items()})
    with _conn(db_path) as c:
        c.execute(
            "INSERT OR REPLACE INTO agent_state (key, value) VALUES ('buy_dates', ?)",
            (payload,),
        )


def load_buy_dates(db_path: str) -> dict:
    """Return {symbol: date_str} or {} if nothing saved yet."""
    if not Path(db_path).exists():
        return {}
    with _conn(db_path) as c:
        row = c.execute(
            "SELECT value FROM agent_state WHERE key = 'buy_dates'"
        ).fetchone()
    if not row:
        return {}
    try:
        return json.loads(row[0])
    except Exception:
        return {}
