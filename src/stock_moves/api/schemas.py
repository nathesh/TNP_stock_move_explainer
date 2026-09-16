"""Response and request schemas — the documented shape of every payload.

Field names match the dicts produced by `stock_moves.queries` exactly, which
is the point: the routes build plain dicts from the read layer, the chat tools
hand the *same* dicts to the model, and these models only validate and
document them. Nothing here reshapes data, so the JSON a reviewer sees in
`/docs` is the JSON the model reasoned over.

Two rules make dict-built responses safe:

* every optional field defaults to `None`, so a serialiser that omits a key
  (an unexplained move, an article shown outside a move) still validates;
* numeric fields are `float | None`, because the trailing-window columns
  (`ret_z`, `vol_z`, the OLS components) are genuinely null until the window
  warms up, and `queries._num` turns NaN into null rather than emitting the
  non-JSON token `NaN`.

Dates arrive as ISO strings from the serialisers and as `date`/`datetime`
objects when a model is built straight off a SQLModel row; pydantic accepts
both and always serialises ISO. `from_attributes` is on for that second case.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ArticleOut",
    "ChatRequest",
    "ChatResponse",
    "CompanyOut",
    "ExplanationOut",
    "IngestResponse",
    "MoveOut",
    "PriceOut",
    "TickerResponse",
    "ToolCallOut",
    "WindowOut",
]

# `date` is a field name on `PriceOut` and `MoveOut`, which shadows the
# imported type inside those class bodies; annotate with these aliases.
DateT = date
DateTimeT = datetime


class _Schema(BaseModel):
    """Base for every payload: validates from ORM rows as well as from dicts."""

    model_config = ConfigDict(from_attributes=True)


class CompanyOut(_Schema):
    """The company ontology row (DESIGN section 2)."""

    ticker: str
    name: str
    sector: str | None = None
    industry: str | None = None
    sector_etf: str | None = None
    peers: list[str] = Field(default_factory=list)
    peers_source: str | None = None
    updated_at: DateTimeT | None = None


class PriceOut(_Schema):
    """One trading day: raw OHLCV plus every stored derived column."""

    date: DateT
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None

    ret: float | None = None
    ret_z: float | None = None
    gap_ret: float | None = None
    intraday_ret: float | None = None
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


class ArticleOut(_Schema):
    """A headline. `relevance` and `category` come from the move link, so they
    are null when the article is shown outside the context of a move."""

    id: int | None = None
    title: str
    source: str | None = None
    url: str
    published_at: DateTimeT | None = None
    relevance: float | None = None
    category: str | None = None


class ExplanationOut(_Schema):
    """The cached, cited explanation of one move (DESIGN section 4)."""

    summary: str
    primary_category: str
    confidence: float | None = None
    cited_article_ids: list[int] = Field(default_factory=list)
    unexplained: bool = False
    provider: str | None = None
    created_at: DateTimeT | None = None


class MoveOut(_Schema):
    """A day flagged as major, with its decomposition, news and explanation."""

    date: DateT
    ret: float | None = None
    ret_z: float | None = None
    gap_ret: float | None = None
    intraday_ret: float | None = None
    vol_z: float | None = None
    direction: str
    routing: str
    near_earnings: bool = False
    near_fomc: bool = False
    near_cpi: bool = False
    regime_mkt: str | None = None
    regime_sector: str | None = None
    mkt_component: float | None = None
    sector_component: float | None = None
    idio_component: float | None = None
    peer_comove: float | None = None
    explanation: ExplanationOut | None = None
    articles: list[ArticleOut] = Field(default_factory=list)


class TickerResponse(_Schema):
    """`GET /tickers/{ticker}`. `ingested` says whether this read triggered the
    lazy population of DESIGN section 5; `filters` echoes the query the server
    actually ran, so a surprising result set is self-explaining."""

    ticker: str
    company: CompanyOut | None = None
    ingested: bool = False
    filters: dict[str, Any] = Field(default_factory=dict)
    moves: list[MoveOut] = Field(default_factory=list)
    prices: list[PriceOut] | None = None


class IngestResponse(_Schema):
    """`POST /tickers/{ticker}/ingest` — what the run actually wrote."""

    ticker: str
    period: str
    top_n: int
    n_prices: int
    n_moves: int
    n_articles: int
    n_explanations: int
    provider: str
    news_source: str


class ChatRequest(_Schema):
    """`POST /chat`. `ticker` is the session default for the tools; omit
    `session_id` on the first turn and the server mints one."""

    message: str
    ticker: str | None = None
    session_id: str | None = None


class ToolCallOut(_Schema):
    """One tool the provider ran, echoed back so the UI can show its work."""

    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: Any = None


class WindowOut(_Schema):
    """The time window the server resolved from the question, or absent.

    `phrase` is the wording it was read from ("this year", "last week"), so a
    client can say which period it answered for. The model is never asked for
    a date and never sees this; `stock_moves.timeframe` says why.
    """

    phrase: str
    start: DateT
    end: DateT

    @classmethod
    def from_window(cls, window: Any) -> WindowOut:
        """Build from a `timeframe.Window` without importing it here."""
        return cls(phrase=window.phrase, start=window.start, end=window.end)


class ChatResponse(_Schema):
    """`POST /chat`. Plain JSON, no SSE (DESIGN section 6)."""

    reply: str
    session_id: str
    tool_calls: list[ToolCallOut] = Field(default_factory=list)
    #: `null` when the question named no period, which is the usual case.
    window: WindowOut | None = None
