"""SQLModel tables — the whole data model of v1.

Columns follow `docs/architecture-v1.md`. Derived quantities (returns, z-scores,
the OLS decomposition, routing, regimes, event proximity) are *stored as
columns* on `prices` and copied onto `moves`, so every API filter is a `WHERE`
on a stored value and changing a threshold never means recomputing anything.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

from sqlmodel import Field, SQLModel, UniqueConstraint

__all__ = [
    "Article",
    "ChatMessage",
    "Company",
    "Explanation",
    "Move",
    "MoveArticle",
    "Price",
    "utcnow",
]

# `date` is itself a column name on `prices` and `moves`, which shadows the
# imported `date` type inside those class bodies; annotate with this alias.
DateT = date


def utcnow() -> datetime:
    """Naive UTC timestamp — SQLite stores no offset, so never hand it one."""
    return datetime.now(UTC).replace(tzinfo=None)


class Company(SQLModel, table=True):
    """Company ontology: identity, sector mapping and peers (v2 knowledge-graph seed)."""

    __tablename__ = "companies"

    ticker: str = Field(primary_key=True)
    name: str
    sector: str | None = None
    industry: str | None = None
    sector_etf: str | None = None
    peers_json: str = "[]"
    peers_source: str | None = None
    updated_at: datetime = Field(default_factory=utcnow)

    @property
    def peers(self) -> list[str]:
        """Peer tickers decoded from `peers_json`; `[]` if unset or malformed."""
        try:
            parsed = json.loads(self.peers_json or "[]")
        except (TypeError, ValueError):
            return []
        if not isinstance(parsed, list):
            return []
        return [str(peer) for peer in parsed]


class Price(SQLModel, table=True):
    """One trading day of OHLCV plus everything derived from it."""

    __tablename__ = "prices"
    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_price_ticker_date"),)

    id: int | None = Field(default=None, primary_key=True)
    ticker: str = Field(index=True)
    date: DateT = Field(index=True)

    open: float
    high: float
    low: float
    close: float
    volume: float

    ret: float | None = None
    gap_ret: float | None = None
    intraday_ret: float | None = None
    ret_z: float | None = None
    vol_z: float | None = None

    mkt_component: float | None = None
    sector_component: float | None = None
    idio_component: float | None = None
    routing: str | None = None

    regime_mkt: str | None = None
    regime_sector: str | None = None

    near_earnings: bool = False
    near_fomc: bool = False
    near_cpi: bool = False


class Move(SQLModel, table=True):
    """A day flagged as major; the decomposition is copied so reads need no join."""

    __tablename__ = "moves"
    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_move_ticker_date"),)

    id: int | None = Field(default=None, primary_key=True)
    ticker: str = Field(index=True)
    date: DateT = Field(index=True)

    ret: float
    ret_z: float
    gap_ret: float | None = None
    intraday_ret: float | None = None
    vol_z: float | None = None

    mkt_component: float | None = None
    sector_component: float | None = None
    idio_component: float | None = None
    routing: str

    direction: str  # "up" | "down"

    regime_mkt: str | None = None
    regime_sector: str | None = None

    near_earnings: bool = False
    near_fomc: bool = False
    near_cpi: bool = False

    peer_comove: float | None = None
    created_at: datetime = Field(default_factory=utcnow)


class Article(SQLModel, table=True):
    """A headline from a `NewsSource`; deduped on url, stored once, linked to many moves."""

    __tablename__ = "articles"

    id: int | None = Field(default=None, primary_key=True)
    url: str = Field(unique=True, index=True)
    title: str
    source: str | None = None
    language: str | None = None
    published_at: datetime | None = None
    news_source: str  # "google_rss" | "gdelt"
    fetched_at: datetime = Field(default_factory=utcnow)


class MoveArticle(SQLModel, table=True):
    """Move-to-article link with the scored relevance and its five components.

    `relevance` is the weighted sum of the components (0.35 bucket_match,
    0.20 entity_match, 0.15 timing, 0.15 source_tier, 0.15 coverage), or the
    mean of that sum and `model_score` when a model key was present.
    """

    __tablename__ = "move_articles"

    move_id: int = Field(foreign_key="moves.id", primary_key=True)
    article_id: int = Field(foreign_key="articles.id", primary_key=True)

    relevance: float
    category: str  # "company" | "industry" | "macro"
    provider: str  # "anthropic" | "heuristic"

    bucket_match: float = 0.0
    entity_match: float = 0.0
    timing: float = 0.0
    source_tier: float = 0.0
    coverage: float = 0.0
    timing_kind: str | None = None  # "cause" | "report"
    model_score: float | None = None


class Explanation(SQLModel, table=True):
    """The cached, cited explanation of one move — at most one per move."""

    __tablename__ = "explanations"

    id: int | None = Field(default=None, primary_key=True)
    move_id: int = Field(foreign_key="moves.id", unique=True, index=True)
    summary: str
    primary_category: str  # "company" | "industry" | "macro" | "unexplained"
    confidence: float
    cited_article_ids_json: str = "[]"
    unexplained: bool = False
    provider: str  # "anthropic" | "heuristic"
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def cited_article_ids(self) -> list[int]:
        """Cited `articles.id` values decoded from json; `[]` if unset or malformed."""
        try:
            parsed = json.loads(self.cited_article_ids_json or "[]")
        except (TypeError, ValueError):
            return []
        if not isinstance(parsed, list):
            return []
        return [int(article_id) for article_id in parsed]


class ChatMessage(SQLModel, table=True):
    """One turn of a `/chat` session."""

    __tablename__ = "chat_messages"

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    role: str  # "user" | "assistant" | "tool"
    content: str
    tool_calls_json: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
