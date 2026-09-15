"""News source protocol, the `NewsItem` record, and the query builders.

DESIGN §3. One protocol, two keyless implementations, and one query per move
per routing bucket with a company baseline. This module imports nothing from
the rest of `stock_moves`: settings values arrive as arguments.

`Article` in DESIGN §3's protocol signature is realised here as `NewsItem`, a
plain record. Mapping a `NewsItem` onto the `articles` table happens in
`scoring.py`, not here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Protocol

__all__ = [
    "MACRO_QUERY_TERMS",
    "NewsItem",
    "NewsSource",
    "build_query",
    "clean_company_name",
    "get_news_source",
    "queries_for_move",
    "window_for",
]


@dataclass(frozen=True)
class NewsItem:
    """One headline from a news source, before it is scored or stored.

    `bucket` is the routing bucket the query came from (`company`, `industry`
    or `macro`). A source's `search` does not know its caller's bucket, so it
    leaves the field empty and the caller sets it with `dataclasses.replace`.
    """

    url: str
    title: str
    source: str | None
    published_at: datetime | None
    language: str | None
    news_source: str
    bucket: str


class NewsSource(Protocol):
    """A keyless, historical headline search over a date window."""

    name: str

    def search(self, query: str, start: date, end: date, limit: int = 30) -> list[NewsItem]:
        """Return at most `limit` items published between `start` and `end`."""
        ...


# The fixed macro vocabulary of DESIGN §3. Deliberately duplicated as
# `MACRO_TERMS` in `providers/base.py` so wave-1 modules import nothing from
# each other.
MACRO_QUERY_TERMS: tuple[str, ...] = (
    "Federal Reserve",
    "interest rates",
    "inflation",
    "CPI",
    "tariffs",
    "jobs report",
    "Treasury yields",
    "recession",
    "stock market",
)

# Legal-form tokens dropped from a `yfinance` company name. Compared after
# lower-casing and removing "." and "," so "Inc." and "N.V." are covered.
_NAME_SUFFIXES: frozenset[str] = frozenset(
    {
        "inc",
        "corp",
        "corporation",
        "ltd",
        "plc",
        "co",
        "company",
        "holdings",
        "group",
        "nv",
        "sa",
    }
)

_VALID_BUCKETS: tuple[str, ...] = ("company", "industry", "macro")


def clean_company_name(name: str) -> str:
    """Strip legal-form suffixes and a leading "The" from a company name.

    `"Apple Inc."` -> `"Apple"`, `"NVIDIA Corporation"` -> `"NVIDIA"`,
    `"The Coca-Cola Company"` -> `"Coca-Cola"`. The last token is never
    removed, so a name that is only a suffix survives unchanged.
    """
    cleaned = name.strip()
    if not cleaned:
        return ""
    if cleaned[:4].lower() == "the ":
        cleaned = cleaned[4:].strip()

    tokens = cleaned.split()
    while len(tokens) > 1:
        tail = tokens[-1].replace(".", "").replace(",", "").lower()
        if tail not in _NAME_SUFFIXES:
            break
        tokens.pop()

    # "JPMorgan Chase & Co." loses "Co." and would otherwise keep a dangling "&".
    return " ".join(tokens).strip().rstrip(",.&").strip()


def window_for(move_date: date, before: int = 1, after: int = 1) -> tuple[date, date]:
    """Return the inclusive calendar window around a move date (DESIGN §3: ±1 day)."""
    return move_date - timedelta(days=before), move_date + timedelta(days=after)


def build_query(
    bucket: str,
    company_name: str,
    ticker: str,
    peers: Sequence[str] = (),
    industry: str | None = None,
) -> str:
    """Build the search query for one routing bucket.

    Every query is anchored to the market ("stock" / "stock market") so a
    common company name does not return obituaries and sports.
    """
    if bucket not in _VALID_BUCKETS:
        raise ValueError(f"unknown bucket {bucket!r}; expected one of {_VALID_BUCKETS}")

    name = clean_company_name(company_name)
    ticker = ticker.strip()

    if bucket == "company":
        query = f'"{name}" stock'
        # Two-letter tickers ("GE", "GM") are ordinary words in a headline
        # search, so they are not worth the false positives.
        if len(ticker) >= 3:
            query = f"{query} OR {ticker}"
        return query

    if bucket == "industry":
        parts: list[str] = []
        if name:
            parts.append(f'"{name}"')
        parts.extend(peer.strip() for peer in peers if peer and peer.strip())
        if industry and industry.strip():
            parts.append(f'"{industry.strip()}"')
        if not parts:
            raise ValueError(
                "industry query needs a company name, at least one peer, or an industry"
            )
        return f"({' OR '.join(parts)}) stock"

    terms = " OR ".join(f'"{term}"' for term in MACRO_QUERY_TERMS)
    return f"({terms}) stock market"


def queries_for_move(
    routing: str,
    company_name: str,
    ticker: str,
    peers: Sequence[str] = (),
    industry: str | None = None,
) -> list[tuple[str, str]]:
    """Return `[(bucket, query)]` for a move: the company baseline, plus its bucket.

    The company query always runs, so even a macro-routed move is checked for
    company-specific news. An `industry` routing with no peers and no industry
    term falls back to the company query alone.
    """
    queries: list[tuple[str, str]] = [
        ("company", build_query("company", company_name, ticker, peers, industry))
    ]

    if routing == "industry":
        try:
            queries.append(
                (
                    "industry",
                    build_query("industry", company_name, ticker, peers, industry),
                )
            )
        except ValueError:
            return queries
    elif routing == "macro":
        queries.append(("macro", build_query("macro", company_name, ticker, peers, industry)))

    return queries


def get_news_source(
    name: str = "google_rss",
    *,
    timeout_s: float = 20.0,
    throttle_s: float = 5.0,
) -> NewsSource:
    """Return the configured news source by name (DESIGN §3: RSS primary, GDELT second)."""
    # Imported here so `base` stays importable without touching httpx and so
    # the two implementations can import `base` without a cycle.
    from .gdelt import GDELTSource
    from .google_rss import GoogleNewsRSS

    if name == "google_rss":
        return GoogleNewsRSS(timeout_s)
    if name == "gdelt":
        return GDELTSource(timeout_s, throttle_s)
    raise ValueError(f"unknown news source {name!r}; expected 'google_rss' or 'gdelt'")
