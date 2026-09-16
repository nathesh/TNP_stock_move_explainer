"""SQLModel tables — the whole data model of v1 and the v1.5 relationship layer.

Columns follow `docs/architecture-v1.md`. Derived quantities (returns, z-scores,
the OLS decomposition, routing, regimes, event proximity) are *stored as
columns* on `prices` and copied onto `moves`, so every API filter is a `WHERE`
on a stored value and changing a threshold never means recomputing anything.

v1.5 (`docs/v1.5-plan.md`) adds two tables — `company_edges`, the one typed-fact
table per company, and `geo_events`, dated geopolitical headline counts per
country — plus the columns the relationship layer stores on existing rows:
`macro_driver` / `macro_driver_component` on `prices` and `moves`, `sub_routing`
/ `rival_comove` / `chain_comove` on `moves`, and `geo_gate` on `move_articles`.

**An older database upgrades in place.** `SQLModel.metadata.create_all` creates
missing *tables* but never adds a column to a table that already exists, so
`db.init_db` follows it with `db.migrate_schema`, which compares this module's
columns against `PRAGMA table_info` and adds every missing nullable column with
`ALTER TABLE ... ADD COLUMN` on open. A `data/app.db` written by v1 therefore
gains the v1.5 columns the first time any entry point opens it — the app
lifespan, the scripts, a seeded cold start — with the old rows left null, and
re-ingesting a ticker with `?refresh=true` fills them in. The same applies to
the deployment snapshot `data/snapshot.db.gz`, which is rebuilt at the end of
v1.5. Only a `NOT NULL` column with no default would need a rebuild instead:
SQLite cannot add one to a table that already has rows, and `migrate_schema`
raises `SchemaMigrationError` rather than skip it in silence.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

from sqlmodel import Field, SQLModel, UniqueConstraint

__all__ = [
    "EDGE_RELATIONS",
    "EDGE_SOURCES",
    "Article",
    "ChatMessage",
    "Company",
    "CompanyEdge",
    "Explanation",
    "GeoEvent",
    "Move",
    "MoveArticle",
    "Price",
    "utcnow",
]

# `company_edges.relation` — `dst` is a ticker for the first three, an ISO-3166
# alpha-2 code for "country", and a factor name (oil/dollar/rates/gold) for
# "factor". Documented here rather than enforced: SQLite has no enum type and a
# CHECK constraint would need a migration to widen.
EDGE_RELATIONS: tuple[str, ...] = ("competitor", "supplier", "customer", "country", "factor")

# `company_edges.source` — "model" is one `suggest_relations` provider call,
# "etf_holdings" the keyless competitor fallback, "prices" the fitted factor betas.
EDGE_SOURCES: tuple[str, ...] = ("model", "etf_holdings", "prices")

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


class CompanyEdge(SQLModel, table=True):
    """One typed fact about a company: `(src, dst, relation)` is the whole key.

    v1.5 decision 1 — every relationship is a row here, not a column anywhere
    else. `relation` is one of `EDGE_RELATIONS` and `source` one of
    `EDGE_SOURCES`; `weight` is the strength the source reported (a model
    confidence, an ETF holding share, or a fitted beta), defaulting to 1.0 for
    sources that state a fact without a number. `competitor` replaces the idea
    of a peer: `companies.peers_json` stays only as the keyless ETF fallback.
    """

    __tablename__ = "company_edges"

    src: str = Field(primary_key=True, index=True)
    dst: str = Field(primary_key=True)
    relation: str = Field(primary_key=True)  # one of EDGE_RELATIONS

    weight: float = 1.0
    source: str  # one of EDGE_SOURCES
    updated_at: datetime = Field(default_factory=utcnow)


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

    # v1.5: the factor-proxy attribution, a second and separate step from the
    # SPY-plus-sector OLS above. `macro_driver` is the proxy with the largest
    # absolute contribution among proxies that themselves moved that day
    # ("oil" | "dollar" | "rates" | "gold" | "country:XX"), `None` if none did.
    macro_driver: str | None = None
    macro_driver_component: float | None = None

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

    # v1.5: routing keeps its three buckets; the sub-bucket says which story.
    # "share_shift" | "supply_chain" | "oil" | "dollar" | "rates" | "gold" |
    # "country:XX", or None when no rule fires.
    sub_routing: str | None = None
    macro_driver: str | None = None
    macro_driver_component: float | None = None

    direction: str  # "up" | "down"

    regime_mkt: str | None = None
    regime_sector: str | None = None

    near_earnings: bool = False
    near_fomc: bool = False
    near_cpi: bool = False

    peer_comove: float | None = None
    # v1.5: same-day co-movement of the `competitor` edges and of the
    # `supplier`/`customer` edges, the two inputs to `sub_routing`.
    rival_comove: float | None = None
    chain_comove: float | None = None
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


class GeoEvent(SQLModel, table=True):
    """Dated geopolitical headline counts for one country, from one news source.

    v1.5 decision 5 — the GDELT Events feed is out of scope, so a "geo event" is
    how loudly a country was in the news that day: the hit count for the geo
    vocabulary query, with a few titles kept for citation. Built only for
    countries a company has a `country` edge to, and only when a move exists, so
    the table stays small. Re-querying the same `(date, country, news_source)`
    updates the row rather than inserting a second count.
    """

    __tablename__ = "geo_events"
    __table_args__ = (
        UniqueConstraint("date", "country", "news_source", name="uq_geo_event_date_country_source"),
    )

    id: int | None = Field(default=None, primary_key=True)
    date: DateT = Field(index=True)
    country: str = Field(index=True)  # ISO-3166 alpha-2

    headline_count: int
    sample_titles_json: str = "[]"
    news_source: str  # "google_rss" | "gdelt"
    fetched_at: datetime = Field(default_factory=utcnow)

    @property
    def sample_titles(self) -> list[str]:
        """Kept headlines decoded from `sample_titles_json`; `[]` if unset or malformed."""
        try:
            parsed = json.loads(self.sample_titles_json or "[]")
        except (TypeError, ValueError):
            return []
        if not isinstance(parsed, list):
            return []
        return [str(title) for title in parsed]


class MoveArticle(SQLModel, table=True):
    """Move-to-article link with the scored relevance and its five components.

    `relevance` is the weighted sum of the components (0.35 bucket_match,
    0.20 entity_match, 0.15 timing, 0.15 source_tier, 0.15 coverage), or the
    mean of that sum and `model_score` when a model key was present.

    v1.5 adds `geo_gate`, which is not one of the weighted components: it is a
    post-hoc cap recorded after the sum, so the stored components still explain
    the score they produced.
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
    # v1.5 decision 7: 1.0 the geo gate opened, 0.0 it closed and `relevance`
    # was capped, None the article never matched the geo vocabulary.
    geo_gate: float | None = None


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
