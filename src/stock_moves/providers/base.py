"""The model layer's interface (DESIGN section 4).

Deliberately dependency-free: plain dataclasses, a `Protocol`, and the tool
specs. This module must not import `stock_moves.models` or
`stock_moves.settings`, so the provider layer can be imported, unit-tested and
reasoned about without a database or a config file. The `from_object` /
`from_objects` constructors read attributes by name, which is what lets the
ingest and API layers hand their SQLModel rows straight to a provider without
this file knowing the table classes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Protocol

__all__ = [
    "MACRO_TERMS",
    "TOOL_SPECS",
    "ArticleInput",
    "ArticleScore",
    "ChatReply",
    "ChatTurn",
    "ExplanationResult",
    "ModelProvider",
    "MoveContext",
    "ToolCallRecord",
    "ToolFn",
]


def _opt_float(obj: Any, name: str) -> float | None:
    """Read `name` off `obj` as a float, or None when absent/unset/unparsable."""
    value = getattr(obj, name, None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _req_float(obj: Any, name: str) -> float:
    value = _opt_float(obj, name)
    return 0.0 if value is None else value


def _opt_str(obj: Any, name: str) -> str | None:
    value = getattr(obj, name, None)
    return None if value is None else str(value)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArticleInput:
    """One headline handed to a provider. Headline only: neither news source
    in v1 gives a body."""

    id: int
    title: str
    source: str | None
    url: str
    published_at: datetime | None

    @classmethod
    def from_object(cls, a: Any) -> ArticleInput:
        """Build from anything with the `articles` columns (duck-typed)."""
        raw_id = getattr(a, "id", None)
        return cls(
            id=int(raw_id) if raw_id is not None else 0,
            title=str(getattr(a, "title", "") or ""),
            source=_opt_str(a, "source"),
            url=str(getattr(a, "url", "") or ""),
            published_at=getattr(a, "published_at", None),
        )


@dataclass(frozen=True)
class MoveContext:
    """Everything quantitative that is known about a move before any article
    is read: the decomposition, the regime, the event flags and the peer set.

    Field order puts the defaulted event flags last so the dataclass is valid;
    build it with keyword arguments.
    """

    ticker: str
    company_name: str
    date: date
    ret: float
    ret_z: float
    gap_ret: float | None
    intraday_ret: float | None
    vol_z: float | None
    mkt_component: float | None
    sector_component: float | None
    idio_component: float | None
    routing: str
    direction: str
    regime_mkt: str | None
    regime_sector: str | None
    near_earnings: bool
    sector: str | None
    industry: str | None
    peers: tuple[str, ...]
    peer_comove: float | None
    near_fomc: bool = False
    near_cpi: bool = False

    @classmethod
    def from_objects(cls, move: Any, company: Any) -> MoveContext:
        """Build from a `moves` row (or a prices row) plus a `companies` row.

        Everything is read with `getattr(obj, name, None)`, so a caller may
        pass a joined object, a SimpleNamespace, or a row that is missing the
        newer columns.
        """
        ticker = str(getattr(move, "ticker", None) or getattr(company, "ticker", None) or "")
        ret = _req_float(move, "ret")
        direction = _opt_str(move, "direction") or ("down" if ret < 0 else "up")
        peers_raw = getattr(company, "peers", None) or ()
        return cls(
            ticker=ticker,
            company_name=str(getattr(company, "name", None) or ticker),
            date=getattr(move, "date", None),
            ret=ret,
            ret_z=_req_float(move, "ret_z"),
            gap_ret=_opt_float(move, "gap_ret"),
            intraday_ret=_opt_float(move, "intraday_ret"),
            vol_z=_opt_float(move, "vol_z"),
            mkt_component=_opt_float(move, "mkt_component"),
            sector_component=_opt_float(move, "sector_component"),
            idio_component=_opt_float(move, "idio_component"),
            routing=_opt_str(move, "routing") or "company",
            direction=direction,
            regime_mkt=_opt_str(move, "regime_mkt"),
            regime_sector=_opt_str(move, "regime_sector"),
            near_earnings=bool(getattr(move, "near_earnings", False)),
            sector=_opt_str(company, "sector"),
            industry=_opt_str(company, "industry"),
            peers=tuple(str(p) for p in peers_raw),
            peer_comove=_opt_float(move, "peer_comove"),
            near_fomc=bool(getattr(move, "near_fomc", False)),
            near_cpi=bool(getattr(move, "near_cpi", False)),
        )


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArticleScore:
    """A provider's verdict on one headline. `category` is one of
    "company", "industry", "macro"."""

    article_id: int
    relevance: float
    category: str


@dataclass(frozen=True)
class ExplanationResult:
    """The cached per-move explanation. `primary_category` is one of
    "company", "industry", "macro", "unexplained"; `confidence` is in [0, 1].
    """

    summary: str
    primary_category: str
    confidence: float
    cited_article_ids: tuple[int, ...]
    unexplained: bool
    #: True when a keyed provider failed and this result came from the keyless
    #: fallback, so the stored row is labelled with the provider that actually
    #: wrote it rather than the one that was configured.
    degraded: bool = False


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #


@dataclass
class ChatTurn:
    """One stored chat message. `role` is "user" or "assistant"."""

    role: str
    content: str


@dataclass
class ToolCallRecord:
    """A tool the provider ran, echoed back to the client for transparency."""

    name: str
    input: dict[str, Any]
    output: Any


@dataclass
class ChatReply:
    reply: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


ToolFn = Callable[..., Any]
"""A read function from `queries.py`, called with keyword arguments taken from
the tool input and returning a JSON-serialisable result."""


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MACRO_TERMS: tuple[str, ...] = (
    "federal reserve",
    "fed ",
    "fomc",
    "interest rate",
    "rate cut",
    "rate hike",
    "inflation",
    "cpi",
    "ppi",
    "jobs report",
    "payrolls",
    "unemployment",
    "tariff",
    "treasury",
    "yields",
    "recession",
    "gdp",
    "oil price",
    "crude",
    "dollar",
    "geopolit",
    "stimulus",
)
"""The macro vocabulary: used both to build the `macro` news query and to
classify a headline that names no company."""


_MOVE_SHAPE = (
    "Each move is {date, ret, ret_z, gap_ret, intraday_ret, vol_z, direction, "
    "routing, near_earnings, regime_mkt, regime_sector, mkt_component, "
    "sector_component, idio_component, peer_comove, explanation: {summary, "
    "primary_category, confidence, cited_article_ids, unexplained} or null}."
)
_ARTICLE_SHAPE = "Each article is {id, title, source, url, published_at, relevance, category}."

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "list_moves",
        "description": (
            "The ticker's notable daily moves, ranked by `order`, with the "
            "cached explanation summary when one exists. " + _MOVE_SHAPE
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Defaults to the session ticker.",
                },
                "start": {
                    "type": "string",
                    "description": "Inclusive start date, YYYY-MM-DD.",
                },
                "end": {
                    "type": "string",
                    "description": "Inclusive end date, YYYY-MM-DD.",
                },
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "Keep only up days or only down days.",
                },
                "order": {
                    "type": "string",
                    "enum": ["z", "pct"],
                    "description": (
                        "How to rank the moves. 'z' (default) ranks by how "
                        "unusual the day was for that stock, which is not the "
                        "same as how large it was. 'pct' ranks by the size of "
                        "the percentage move -- use it for 'biggest', "
                        "'largest', 'most', 'worst' or 'best' questions."
                    ),
                    "default": "z",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum moves to return (default 10).",
                    "default": 10,
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_move",
        "description": (
            "One move with its decomposition, linked articles and explanation. "
            + _MOVE_SHAPE
            + " It also carries articles: a list of articles. "
            + _ARTICLE_SHAPE
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Defaults to the session ticker.",
                },
                "date": {
                    "type": "string",
                    "description": "The trading day, YYYY-MM-DD.",
                },
            },
            "required": ["date"],
        },
    },
    {
        "name": "search_news",
        "description": (
            "Headlines linked to a ticker's moves, filtered by a substring of "
            "the title. " + _ARTICLE_SHAPE
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Defaults to the session ticker.",
                },
                "query": {
                    "type": "string",
                    "description": "Case-insensitive substring of the headline.",
                },
                "start": {
                    "type": "string",
                    "description": "Inclusive start date, YYYY-MM-DD.",
                },
                "end": {
                    "type": "string",
                    "description": "Inclusive end date, YYYY-MM-DD.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum articles to return (default 20).",
                    "default": 20,
                },
            },
            "required": [],
        },
    },
]


class ModelProvider(Protocol):
    """The one interface the rest of the app talks to.

    `HeuristicProvider` implements it with keyword rules and the
    decomposition; `AnthropicProvider` implements it with `claude-sonnet-5`.
    """

    name: str

    def score_articles(
        self, move: MoveContext, articles: Sequence[ArticleInput]
    ) -> list[ArticleScore]:
        """Relevance 0-1 and a category for every article, in input order."""
        ...

    def explain(
        self, move: MoveContext, scored: Sequence[tuple[ArticleInput, ArticleScore]]
    ) -> ExplanationResult:
        """Write the move's explanation.

        `scored` is sorted by relevance descending and already cut to top-K by
        the caller.
        """
        ...

    def suggest_peers(
        self, ticker: str, name: str, sector: str | None, industry: str | None
    ) -> list[str]:
        """Peer tickers for the company ontology; may be empty."""
        ...

    def chat(
        self,
        history: Sequence[ChatTurn],
        tools: Mapping[str, ToolFn],
        ticker: str | None,
    ) -> ChatReply:
        """Answer a chat turn, using the tools.

        `tools` maps the `TOOL_SPECS` names to callables that take keyword
        arguments and return the documented dicts. `history` ends with the
        newest user turn.
        """
        ...
