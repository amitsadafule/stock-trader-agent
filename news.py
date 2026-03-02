"""
news.py — Free news scraper + sentiment scorer for watchlist stocks
====================================================================
Sources used (all free, no API key):
  1. Google News RSS       — broad coverage, India-specific
  2. Yahoo Finance RSS     — reliable, global commodity/ETF news
  3. Economic Times RSS    — India financial news (topic-filtered URLs)
  4. Moneycontrol RSS      — India market focused

Returns per-symbol:
  score      — -100 to +100 (negative=bearish, positive=bullish, 0=neutral)
  label      — "BULLISH" / "BEARISH" / "NEUTRAL"
  headlines  — list of {title, source, url, age_hours, sentiment} dicts
  summary    — one-line human readable summary

Score feeds into conviction:
  BULLISH news → +5 to +20 pts added to conviction
  BEARISH news → -5 to -20 pts deducted
  NEUTRAL / no news → 0 pts
"""

import re
import time
import logging
import datetime
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Optional

log = logging.getLogger(__name__)

# ── Sentiment keyword dictionaries ────────────────────────────────────────────
BULLISH_KEYWORDS = {
    "surge": 3, "surges": 3, "soar": 3, "soars": 3, "rally": 3, "rallies": 3,
    "breakout": 3, "all-time high": 3, "record high": 3, "multi-year high": 3,
    "52-week high": 3, "upgrade": 3, "strong buy": 3, "outperform": 3,
    "beats": 3, "beat estimates": 3, "dividend": 3, "buyback": 3,
    "rise": 2, "rises": 2, "gain": 2, "gains": 2, "climbs": 2,
    "positive": 2, "bullish": 2, "upside": 2, "growth": 2, "profit": 2,
    "recovery": 2, "bounce": 2, "accumulate": 2, "inflow": 2, "demand": 2,
    "steady": 1, "stable": 1, "firm": 1, "higher": 1, "increase": 1,
    "support": 1, "optimistic": 1, "holds": 1,
}

BEARISH_KEYWORDS = {
    "crash": 3, "crashes": 3, "collapse": 3, "plunge": 3, "plunges": 3,
    "downgrade": 3, "underperform": 3, "avoid": 3, "sell off": 3,
    "52-week low": 3, "all-time low": 3, "record low": 3,
    "below estimates": 3, "default": 3, "fraud": 3,
    "fall": 2, "falls": 2, "drop": 2, "drops": 2, "decline": 2, "declines": 2,
    "bearish": 2, "loss": 2, "losses": 2, "weak": 2, "outflow": 2,
    "pressure": 2, "concern": 2, "warning": 2, "oversupply": 2, "tariff": 2,
    "lower": 1, "dip": 1, "slip": 1, "slips": 1, "cautious": 1,
    "uncertain": 1, "headwind": 1, "volatile": 1,
}

# ── ETF → search terms mapping ───────────────────────────────────────────────
# Multiple terms tried in order — first match that returns results wins.
# Keep the most specific terms first, broader fallbacks last.
SYMBOL_KEYWORDS = {
    "SILVERIETF":  ["silver price India", "silver ETF India", "silver commodity", "MCX silver"],
    "GOLDBEES":    ["gold price India", "gold ETF India", "gold commodity", "MCX gold"],
    "GOLDETF":     ["gold price India", "gold ETF India", "gold commodity", "MCX gold"],
    "LIQUIDBEES":  ["liquid fund India", "overnight fund", "money market India"],
    "NIFTYBEES":   ["Nifty 50", "NSE Nifty", "Indian stock market"],
    "JUNIORBEES":  ["Nifty midcap", "midcap India"],
    "BANKBEES":    ["Bank Nifty", "Indian banking sector"],
    "CPSEETF":     ["CPSE ETF", "PSU stocks India"],
    "ITBEES":      ["IT sector India", "Nifty IT"],
    "PHARMABEES":  ["pharma sector India", "Nifty Pharma"],
}

# ── Relevance keywords — headline must contain at least one ───────────────────
# Broader than SYMBOL_KEYWORDS — used to filter generic headlines
RELEVANCE_KEYWORDS = {
    "SILVERIETF":  ["silver", "silverietf", "mcx silver", "precious metal"],
    "GOLDBEES":    ["gold", "goldbees", "bullion", "mcx gold", "precious metal"],
    "GOLDETF":     ["gold", "goldetf", "bullion", "mcx gold", "precious metal"],
    "LIQUIDBEES":  ["liquid", "liquidbees", "overnight", "money market"],
    "NIFTYBEES":   ["nifty", "niftybees", "sensex", "indian market"],
    "JUNIORBEES":  ["midcap", "juniorbees", "nifty junior"],
    "BANKBEES":    ["bank nifty", "bankbees", "banking", "bank"],
    "CPSEETF":     ["cpse", "psu", "public sector"],
    "ITBEES":      ["it sector", "itbees", "infotech", "technology"],
    "PHARMABEES":  ["pharma", "pharmabees", "pharmaceutical"],
}

# ── RSS source definitions ────────────────────────────────────────────────────
def _google_news_url(query: str) -> str:
    q = urllib.parse.quote(query)
    return f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"

def _yahoo_finance_url(query: str) -> str:
    # Yahoo Finance news search RSS — reliable for commodities/ETFs
    q = urllib.parse.quote(query)
    return f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={q}&region=IN&lang=en-IN"

def _et_markets_url(query: str) -> str:
    # Economic Times Markets RSS — uses their search, not just top stories
    q = urllib.parse.quote(query)
    return f"https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"

def _moneycontrol_url(query: str) -> str:
    return "https://www.moneycontrol.com/rss/marketreports.xml"

def _investing_com_url(query: str) -> str:
    # Investing.com commodity news RSS
    q = urllib.parse.quote(query)
    return f"https://www.investing.com/rss/news_14.rss"  # commodities feed

SOURCES = [
    ("Google News",    _google_news_url,    1.2),
    ("Yahoo Finance",  _yahoo_finance_url,  1.0),
    ("Economic Times", _et_markets_url,     1.1),
    ("Moneycontrol",   _moneycontrol_url,   1.1),
]

# ── HTTP fetch ────────────────────────────────────────────────────────────────
# Rotate User-Agent strings to reduce blocking
_UA_LIST = [
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.144 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Dalvik/2.1.0 (Linux; U; Android 13; Pixel 7 Build/TQ3A.230901.001)",
]
_ua_idx = 0

def _fetch(url: str, timeout: int = 10) -> Optional[str]:
    global _ua_idx
    ua = _UA_LIST[_ua_idx % len(_UA_LIST)]
    _ua_idx += 1
    headers = {
        "User-Agent": ua,
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
        "Accept-Language": "en-IN,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Cache-Control": "no-cache",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            # Handle gzip if needed
            if resp.info().get("Content-Encoding") == "gzip":
                import gzip
                raw = gzip.decompress(raw)
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw.decode("latin-1", errors="replace")
    except Exception as e:
        log.debug(f"Fetch failed [{url[:60]}]: {type(e).__name__}: {e}")
        return None


# ── RSS parser ────────────────────────────────────────────────────────────────
def _parse_rss(xml_text: str, source_name: str, max_age_hours: int) -> list:
    articles = []
    try:
        # Strip any BOM or leading whitespace
        xml_text = xml_text.strip().lstrip("\ufeff")
        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        items = root.findall(".//item") or root.findall(".//atom:entry", ns)
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=max_age_hours)

        for item in items[:25]:
            title_el = item.find("title") or item.find("atom:title", ns)
            title = ""
            if title_el is not None:
                # Google News wraps titles in CDATA — .text handles it
                title = (title_el.text or "").strip()
                # Strip HTML tags if any
                title = re.sub(r"<[^>]+>", "", title).strip()
            if not title or len(title) < 10:
                continue

            link_el = item.find("link")
            url = ""
            if link_el is not None:
                url = (link_el.text or link_el.get("href", "")).strip()

            pub_el = (item.find("pubDate") or item.find("published")
                      or item.find("atom:published", ns))
            age_hours = 0.0
            if pub_el is not None and pub_el.text:
                try:
                    from email.utils import parsedate_to_datetime
                    pub_dt  = parsedate_to_datetime(pub_el.text)
                    pub_utc = pub_dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
                    age_hours = (datetime.datetime.utcnow() - pub_utc).total_seconds() / 3600
                    if pub_utc < cutoff:
                        continue
                except Exception:
                    pass

            articles.append({
                "title":     title,
                "url":       url,
                "source":    source_name,
                "age_hours": round(age_hours, 1),
            })
    except ET.ParseError as e:
        log.debug(f"RSS parse error [{source_name}]: {e}")
    return articles


# ── Relevance filter ──────────────────────────────────────────────────────────
def _is_relevant(title: str, symbol: str) -> bool:
    """
    Check if a headline is about this stock or its underlying asset.
    Uses RELEVANCE_KEYWORDS which are broader than search terms.
    Prevents generic market news from polluting sentiment.
    """
    t = title.lower()
    # Direct symbol match
    if symbol.lower() in t:
        return True
    # Mapped relevance keywords
    for kw in RELEVANCE_KEYWORDS.get(symbol.upper(), [symbol.lower()]):
        if kw.lower() in t:
            return True
    return False


# ── Sentiment scorer ──────────────────────────────────────────────────────────
def _score_headline(title: str) -> int:
    t = title.lower()
    score = 0
    for word, weight in BULLISH_KEYWORDS.items():
        if word in t:
            score += weight
    for word, weight in BEARISH_KEYWORDS.items():
        if word in t:
            score -= weight
    return score


# ── Main public function ──────────────────────────────────────────────────────
_news_cache: dict = {}
NEWS_CACHE_TTL = 1800  # 30 minutes


def get_news_sentiment(symbol: str, max_age_hours: int = 24) -> dict:
    """
    Fetch and score news for a stock/ETF symbol from multiple free RSS sources.

    Strategy:
    1. Try each search term from SYMBOL_KEYWORDS against each source
    2. Filter returned articles to only those mentioning the asset
    3. Score each headline with BULLISH/BEARISH keyword weights
    4. Normalise to -100..+100 and compute conviction delta

    Returns dict with: score, label, headlines, conviction_delta, summary,
                       fetched_at, error
    """
    # Cache check
    cached = _news_cache.get(symbol)
    if cached:
        ts, result = cached
        if time.time() - ts < NEWS_CACHE_TTL:
            log.debug(f"📰 {symbol}: news from cache ({int((time.time()-ts)/60)}min old)")
            return result

    log.info(f"📰 {symbol}: fetching news from {len(SOURCES)} sources…")
    all_articles = []
    search_terms  = SYMBOL_KEYWORDS.get(symbol.upper(), [symbol])

    for source_name, url_fn, _weight in SOURCES:
        source_articles = []
        for term in search_terms:
            try:
                url  = url_fn(term)
                text = _fetch(url)
                if not text:
                    continue
                arts = _parse_rss(text, source_name, max_age_hours)
                relevant = [a for a in arts if _is_relevant(a["title"], symbol)]
                source_articles.extend(relevant)
                log.debug(f"  {source_name} [{term!r}]: {len(arts)} parsed → {len(relevant)} relevant")
                if source_articles:
                    break   # found relevant articles — no need to try next term
            except Exception as e:
                log.debug(f"  {source_name} error: {e}")
        all_articles.extend(source_articles)
        time.sleep(0.3)   # tiny gap to avoid rate limiting across sources

    # Deduplicate by title similarity (avoid same story from multiple sources)
    seen_titles = set()
    unique = []
    for a in all_articles:
        key = re.sub(r"[^a-z0-9]", "", a["title"].lower())[:60]
        if key not in seen_titles:
            seen_titles.add(key)
            unique.append(a)

    log.info(f"📰 {symbol}: {len(all_articles)} fetched → {len(unique)} unique relevant")

    # Score each headline
    scored = []
    raw_total = 0
    for art in unique:
        s = _score_headline(art["title"])
        raw_total += s
        art["sentiment"] = "positive" if s > 0 else "negative" if s < 0 else "neutral"
        art["raw_score"] = s
        scored.append(art)

    # Sort: most impactful first, then most recent
    scored.sort(key=lambda x: (-abs(x.get("raw_score", 0)), x.get("age_hours", 99)))

    # Normalise raw score to -100..+100
    if not scored:
        norm_score = 0
    else:
        capped     = max(-30, min(30, raw_total))
        norm_score = int(capped / 30 * 100)

    # Conviction delta: ±20 pts max, only kicks in at score ≥ 20
    if norm_score >= 60:   conviction_delta = 20
    elif norm_score >= 30: conviction_delta = 10
    elif norm_score >= 20: conviction_delta = 5
    elif norm_score <= -60: conviction_delta = -20
    elif norm_score <= -30: conviction_delta = -10
    elif norm_score <= -20: conviction_delta = -5
    else:                   conviction_delta = 0

    # Label
    if norm_score >= 20:    label = "BULLISH"
    elif norm_score <= -20: label = "BEARISH"
    else:                   label = "NEUTRAL"

    # Summary
    n   = len(scored)
    pos = sum(1 for a in scored if a["sentiment"] == "positive")
    neg = sum(1 for a in scored if a["sentiment"] == "negative")
    if n == 0:
        summary = "No relevant news found — sentiment unavailable"
    elif label == "BULLISH":
        summary = f"{pos}/{n} positive headlines — news supports BUY signal"
    elif label == "BEARISH":
        summary = f"{neg}/{n} negative headlines — news cautions against buying"
    else:
        summary = f"{n} headlines, mixed or neutral sentiment"

    result = {
        "score":            norm_score,
        "label":            label,
        "headlines":        scored[:5],
        "conviction_delta": conviction_delta,
        "summary":          summary,
        "fetched_at":       datetime.datetime.now().isoformat(timespec="seconds"),
        "error":            "",
    }

    _news_cache[symbol] = (time.time(), result)
    log.info(
        f"📰 {symbol}: {label} score={norm_score:+d} delta={conviction_delta:+d}pt "
        f"| {n} headlines | {summary}"
    )
    return result


def clear_cache(symbol: str = None):
    """Force re-fetch on next call. Pass symbol=None to clear all."""
    if symbol:
        _news_cache.pop(symbol, None)
    else:
        _news_cache.clear()
