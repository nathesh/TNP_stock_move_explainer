"""The read layer: every query the API routes and the chat tools share.

One module owns all reads, for two reasons. First, the filters in DESIGN
section 6 (`z_threshold`, `pct_threshold`, `direction`, `category`,
`min_relevance`, `start`, `end`, `limit`) are applied in SQL against stored
columns, so a threshold change is a `WHERE`, never a recomputation. Second,
the dicts the serialisers here produce are simultaneously the JSON the API
returns and the tool results promised by `providers.base.TOOL_SPECS`, so the
model and the HTTP client see exactly the same shapes.

Deliberately depends on `stock_moves.models` alone: no settings, no engine, no
network. Callers pass in the `Session`, which is what lets the chat tools, the
routes and the tests share these functions.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any

from sqlalchemy import func, nulls_last, or_
from sqlmodel import Session, col, select

from stock_moves.models import (
    Article,
    ChatMessage,
    Company,
    Explanation,
    Move,
    MoveArticle,
    Price,
)

__all__ = [
    "MoveFilters",
    "article_to_dict",
    "articles_for_move",
    "chat_history",
    "explanation_to_dict",
    "get_company",
    "get_explanation",
    "get_move",
    "has_prices",
    "list_moves",
    "list_prices",
    "move_to_dict",
    "price_to_dict",
    "save_chat_message",
    "search_news",
]

#: Floats are rounded before they leave this module so the JSON does not carry
#: sixteen digits of binary-float noise into a prompt or a response body.
_PLACES = 6

_DAY_START = time(0, 0, 0)
_DAY_END = time(23, 59, 59, 999999)


@dataclass
class MoveFilters:
    """The query-time filter bundle for `list_moves`.

    Defaults match DESIGN section 1: a move is a day with `abs(ret_z) >= 2.0`
    **or** `abs(ret) >= 0.02`. Because the two thresholds are OR-ed, raising
    only `z_threshold` does not tighten the result set on its own — raise
    `pct_threshold` with it to filter on z alone.

    `min_relevance` is not used by `list_moves`; it travels with the rest of
    the filters because the same bundle drives the article queries on the
    ticker route.
    """

    start: date | None = None
    end: date | None = None
    z_threshold: float = 2.0
    pct_threshold: float = 0.02
    direction: str | None = None
    category: str | None = None
    min_relevance: float = 0.0
    limit: int = 50


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _norm_ticker(ticker: str) -> str:
    """Tickers are stored upper-case; accept `nvda` from a URL path anyway."""
    return ticker.strip().upper()


def _num(value: float | None) -> float | None:
    """Round for JSON. `None` passes through, and NaN/inf become `None`.

    A NaN would serialise to the bare token `NaN`, which is not JSON, so the
    rolling windows that have not warmed up yet (the first 20 days of `ret_z`)
    have to come out as nulls.
    """
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, _PLACES)


def _iso(value: date | datetime | None) -> str | None:
    """ISO-8601 string for a date or a naive-UTC datetime; `None` passes through."""
    return None if value is None else value.isoformat()


def _window(start: date | None, end: date | None) -> tuple[datetime | None, datetime | None]:
    """Turn an inclusive date window into the naive-UTC datetime bounds that
    `articles.published_at` is stored in."""
    lower = None if start is None else datetime.combine(start, _DAY_START)
    upper = None if end is None else datetime.combine(end, _DAY_END)
    return lower, upper


# --------------------------------------------------------------------------- #
# Company and prices
# --------------------------------------------------------------------------- #


def get_company(session: Session, ticker: str) -> Company | None:
    """The `companies` row, or `None` if the ticker has never been ingested."""
    return session.get(Company, _norm_ticker(ticker))


def has_prices(session: Session, ticker: str) -> bool:
    """Whether any price row exists — the lazy-populate check on every read."""
    statement = select(Price.id).where(col(Price.ticker) == _norm_ticker(ticker)).limit(1)
    return session.exec(statement).first() is not None


def list_prices(
    session: Session,
    ticker: str,
    start: date | None = None,
    end: date | None = None,
) -> list[Price]:
    """Price rows in the window, oldest first (the order a chart wants)."""
    statement = select(Price).where(col(Price.ticker) == _norm_ticker(ticker))
    if start is not None:
        statement = statement.where(col(Price.date) >= start)
    if end is not None:
        statement = statement.where(col(Price.date) <= end)
    statement = statement.order_by(col(Price.date).asc())
    return list(session.exec(statement).all())


# --------------------------------------------------------------------------- #
# Moves
# --------------------------------------------------------------------------- #


def list_moves(session: Session, ticker: str, filters: MoveFilters) -> list[Move]:
    """Moves for a ticker matching `filters`, biggest absolute z-score first.

    `category` matches the explanation's `primary_category` when the move has
    been explained and falls back to the quantitative `routing` when it has
    not, so filtering by category does not silently drop every unexplained
    move. The outer join cannot fan rows out: `explanations.move_id` is unique.
    """
    statement = select(Move).where(col(Move.ticker) == _norm_ticker(ticker))
    if filters.start is not None:
        statement = statement.where(col(Move.date) >= filters.start)
    if filters.end is not None:
        statement = statement.where(col(Move.date) <= filters.end)
    statement = statement.where(
        or_(
            func.abs(col(Move.ret_z)) >= filters.z_threshold,
            func.abs(col(Move.ret)) >= filters.pct_threshold,
        )
    )
    if filters.direction:
        statement = statement.where(col(Move.direction) == filters.direction)
    if filters.category:
        statement = statement.outerjoin(
            Explanation, col(Explanation.move_id) == col(Move.id)
        ).where(
            func.coalesce(col(Explanation.primary_category), col(Move.routing)) == filters.category
        )
    statement = statement.order_by(
        nulls_last(func.abs(col(Move.ret_z)).desc()), col(Move.date).desc()
    )
    if filters.limit > 0:
        statement = statement.limit(filters.limit)
    return list(session.exec(statement).all())


def get_move(session: Session, ticker: str, on: date) -> Move | None:
    """The move on one trading day, or `None` if that day was not major."""
    statement = select(Move).where(col(Move.ticker) == _norm_ticker(ticker), col(Move.date) == on)
    return session.exec(statement).first()


def articles_for_move(
    session: Session,
    move_id: int,
    min_relevance: float = 0.0,
    limit: int | None = None,
) -> list[tuple[Article, MoveArticle]]:
    """Articles linked to one move, most relevant first, with their link rows.

    The link row carries the scored relevance and category, so it is returned
    alongside the article rather than folded in — `article_to_dict` needs both.
    """
    statement = (
        select(Article, MoveArticle)
        .join(MoveArticle, col(MoveArticle.article_id) == col(Article.id))
        .where(
            col(MoveArticle.move_id) == move_id,
            col(MoveArticle.relevance) >= min_relevance,
        )
        .order_by(col(MoveArticle.relevance).desc(), col(Article.id).asc())
    )
    if limit is not None and limit > 0:
        statement = statement.limit(limit)
    return list(session.exec(statement).all())


def get_explanation(session: Session, move_id: int) -> Explanation | None:
    """The cached explanation for a move, or `None` if it has not been written."""
    statement = select(Explanation).where(col(Explanation.move_id) == move_id)
    return session.exec(statement).first()


def search_news(
    session: Session,
    ticker: str,
    query: str | None = None,
    start: date | None = None,
    end: date | None = None,
    min_relevance: float = 0.0,
    limit: int = 20,
) -> list[tuple[Article, MoveArticle]]:
    """Headlines linked to any move of the ticker, most relevant first.

    One article is often linked to several moves of the same ticker, so rows
    are deduplicated on `articles.id` after ordering — the surviving row is the
    highest-relevance link, and `limit` is applied to distinct articles rather
    than to link rows.
    """
    lower, upper = _window(start, end)
    statement = (
        select(Article, MoveArticle)
        .join(MoveArticle, col(MoveArticle.article_id) == col(Article.id))
        .join(Move, col(Move.id) == col(MoveArticle.move_id))
        .where(
            col(Move.ticker) == _norm_ticker(ticker),
            col(MoveArticle.relevance) >= min_relevance,
        )
    )
    if query:
        statement = statement.where(col(Article.title).ilike(f"%{query}%"))
    if lower is not None:
        statement = statement.where(col(Article.published_at) >= lower)
    if upper is not None:
        statement = statement.where(col(Article.published_at) <= upper)
    statement = statement.order_by(
        col(MoveArticle.relevance).desc(),
        nulls_last(col(Article.published_at).desc()),
    )

    seen: set[int | None] = set()
    results: list[tuple[Article, MoveArticle]] = []
    for article, link in session.exec(statement):
        if article.id in seen:
            continue
        seen.add(article.id)
        results.append((article, link))
        if limit > 0 and len(results) >= limit:
            break
    return results


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #


def save_chat_message(
    session: Session,
    session_id: str,
    role: str,
    content: str,
    tool_calls: list[dict[str, Any]] | None = None,
) -> ChatMessage:
    """Append one turn to a chat session and return the committed row."""
    message = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        tool_calls_json=None if tool_calls is None else json.dumps(tool_calls),
    )
    session.add(message)
    session.commit()
    session.refresh(message)
    return message


def chat_history(session: Session, session_id: str, limit: int = 20) -> list[ChatMessage]:
    """The most recent `limit` turns of a session, oldest first.

    Newest-first in SQL then reversed, so a long session is truncated at the
    old end — the model needs the turns nearest the question, not the first
    twenty ever sent.
    """
    statement = (
        select(ChatMessage)
        .where(col(ChatMessage.session_id) == session_id)
        .order_by(col(ChatMessage.created_at).desc(), col(ChatMessage.id).desc())
    )
    if limit > 0:
        statement = statement.limit(limit)
    return list(reversed(session.exec(statement).all()))


# --------------------------------------------------------------------------- #
# Serialisers — the tool-result and API JSON shapes
# --------------------------------------------------------------------------- #


def price_to_dict(p: Price) -> dict[str, Any]:
    """One price row as JSON: raw OHLCV plus everything derived from it."""
    return {
        "date": _iso(p.date),
        "open": _num(p.open),
        "high": _num(p.high),
        "low": _num(p.low),
        "close": _num(p.close),
        "volume": _num(p.volume),
        "ret": _num(p.ret),
        "gap_ret": _num(p.gap_ret),
        "intraday_ret": _num(p.intraday_ret),
        "ret_z": _num(p.ret_z),
        "vol_z": _num(p.vol_z),
        "mkt_component": _num(p.mkt_component),
        "sector_component": _num(p.sector_component),
        "idio_component": _num(p.idio_component),
        "routing": p.routing,
        "regime_mkt": p.regime_mkt,
        "regime_sector": p.regime_sector,
        "near_earnings": bool(p.near_earnings),
        "near_fomc": bool(p.near_fomc),
        "near_cpi": bool(p.near_cpi),
    }


def article_to_dict(a: Article, link: MoveArticle | None = None) -> dict[str, Any]:
    """One article as JSON. `relevance` and `category` come from the link row
    and are `None` when the article is shown outside the context of a move."""
    return {
        "id": a.id,
        "title": a.title,
        "source": a.source,
        "url": a.url,
        "published_at": _iso(a.published_at),
        "relevance": None if link is None else _num(link.relevance),
        "category": None if link is None else link.category,
    }


def explanation_to_dict(e: Explanation | None) -> dict[str, Any] | None:
    """The cached explanation as JSON, or `None` for a move with none yet."""
    if e is None:
        return None
    return {
        "summary": e.summary,
        "primary_category": e.primary_category,
        "confidence": _num(e.confidence),
        "cited_article_ids": e.cited_article_ids,
        "unexplained": bool(e.unexplained),
        "provider": e.provider,
        "created_at": _iso(e.created_at),
    }


def move_to_dict(
    m: Move,
    explanation: Explanation | None = None,
    articles: Sequence[tuple[Article, MoveArticle]] | None = None,
) -> dict[str, Any]:
    """One move as JSON: the decomposition, the flags and the explanation.

    `articles` is a tri-state. Omitted (`None`) leaves the key out entirely —
    a list endpoint says nothing about news rather than claiming there is
    none — while an empty sequence emits `"articles": []`.
    """
    payload: dict[str, Any] = {
        "date": _iso(m.date),
        "ret": _num(m.ret),
        "ret_z": _num(m.ret_z),
        "gap_ret": _num(m.gap_ret),
        "intraday_ret": _num(m.intraday_ret),
        "vol_z": _num(m.vol_z),
        "direction": m.direction,
        "routing": m.routing,
        "near_earnings": bool(m.near_earnings),
        "near_fomc": bool(m.near_fomc),
        "near_cpi": bool(m.near_cpi),
        "regime_mkt": m.regime_mkt,
        "regime_sector": m.regime_sector,
        "mkt_component": _num(m.mkt_component),
        "sector_component": _num(m.sector_component),
        "idio_component": _num(m.idio_component),
        "peer_comove": _num(m.peer_comove),
        "explanation": explanation_to_dict(explanation),
    }
    if articles is not None:
        payload["articles"] = [article_to_dict(a, link) for a, link in articles]
    return payload
