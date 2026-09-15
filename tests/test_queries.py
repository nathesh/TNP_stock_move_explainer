"""Tests for stock_moves.queries. No network: rows are seeded by hand.

The fixture below is one ticker with six moves whose z-scores and returns are
chosen to separate the two thresholds from each other, and two explanations
whose `primary_category` deliberately disagrees with the move's quantitative
`routing`, so the category filter has to prove which of the two it used.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

import pytest

from stock_moves.db import Session
from stock_moves.models import Article, Company, Explanation, Move, MoveArticle, Price
from stock_moves.queries import (
    MoveFilters,
    article_to_dict,
    articles_for_move,
    chat_history,
    explanation_to_dict,
    get_company,
    get_explanation,
    get_move,
    has_prices,
    list_moves,
    list_prices,
    move_to_dict,
    price_to_dict,
    save_chat_message,
    search_news,
)

TICKER = "TEST"


def _at(month: int, day: int, hour: int, minute: int) -> datetime:
    """A naive-UTC timestamp in 2025, the way `models.utcnow()` stores one."""
    return datetime(2025, month, day, hour, minute, tzinfo=UTC).replace(tzinfo=None)


MOVE_KEYS = {
    "date",
    "ret",
    "ret_z",
    "gap_ret",
    "intraday_ret",
    "vol_z",
    "direction",
    "routing",
    "near_earnings",
    "near_fomc",
    "near_cpi",
    "regime_mkt",
    "regime_sector",
    "mkt_component",
    "sector_component",
    "idio_component",
    "peer_comove",
    "explanation",
}

# (label, date, ret, ret_z, direction, routing)
#
# `c` is the only row that clears neither threshold (|z| 1.2, |ret| 0.01) and
# `e` clears the percentage threshold alone (|z| 0.5, |ret| 0.025).
MOVE_ROWS: tuple[tuple[str, date, float, float, str, str], ...] = (
    ("a", date(2025, 2, 3), 0.05, 2.5, "up", "company"),
    ("b", date(2025, 3, 10), -0.06, -3.1, "down", "macro"),
    ("c", date(2025, 4, 15), 0.01, 1.2, "up", "industry"),
    ("d", date(2025, 5, 20), -0.04, -2.2, "down", "company"),
    ("e", date(2025, 6, 25), 0.025, 0.5, "up", "industry"),
    ("f", date(2025, 8, 1), 0.08, 4.0, "up", "company"),
)


@dataclass
class Seeded:
    """Handles on the seeded rows, so a test can name `moves["f"]`."""

    moves: dict[str, Move]
    articles: dict[str, Article]


@pytest.fixture
def seeded(session: Session) -> Seeded:
    """One company, one price row, six moves, two explanations, two articles."""
    session.add(Company(ticker=TICKER, name="Test Corp", sector="Technology", sector_etf="XLK"))
    session.add(
        Price(
            ticker=TICKER,
            date=date(2025, 2, 3),
            open=100.0,
            high=106.0,
            low=99.0,
            close=105.0,
            volume=1_000_000.0,
            ret=0.05,
            ret_z=2.5,
            routing="company",
        )
    )
    session.add(
        Price(
            ticker=TICKER,
            date=date(2025, 1, 2),
            open=99.0,
            high=100.0,
            low=98.0,
            close=100.0,
            volume=900_000.0,
        )
    )

    moves: dict[str, Move] = {}
    for label, day, ret, ret_z, direction, routing in MOVE_ROWS:
        move = Move(
            ticker=TICKER,
            date=day,
            ret=ret,
            ret_z=ret_z,
            direction=direction,
            routing=routing,
            vol_z=1.5,
            idio_component=ret,
            peer_comove=0.01,
        )
        session.add(move)
        moves[label] = move
    session.commit()
    for move in moves.values():
        session.refresh(move)

    # The explained moves contradict their routing on purpose: `a` routes
    # company but is explained macro, `b` routes macro but is explained company.
    session.add(
        Explanation(
            move_id=int(moves["a"].id or 0),
            summary="Rate expectations repriced the whole tape.",
            primary_category="macro",
            confidence=0.7,
            cited_article_ids_json="[1]",
            provider="heuristic",
        )
    )
    session.add(
        Explanation(
            move_id=int(moves["b"].id or 0),
            summary="Guidance cut, company specific.",
            primary_category="company",
            confidence=0.55,
            provider="heuristic",
        )
    )

    articles = {
        "hi": Article(
            url="https://example.com/guidance",
            title="Test Corp Raises Guidance After Blowout Quarter",
            source="Reuters",
            published_at=_at(2, 2, 13, 0),
            news_source="google_news_rss",
        ),
        "lo": Article(
            url="https://example.com/macro",
            title="Chip sector slips on macro jitters",
            source="Some Blog",
            published_at=_at(2, 3, 9, 30),
            news_source="google_news_rss",
        ),
    }
    for article in articles.values():
        session.add(article)
    session.commit()
    for article in articles.values():
        session.refresh(article)

    # The high-relevance article is linked to two moves of the same ticker, so
    # `search_news` has to deduplicate it.
    session.add(
        MoveArticle(
            move_id=int(moves["a"].id or 0),
            article_id=int(articles["hi"].id or 0),
            relevance=0.9,
            category="company",
            provider="heuristic",
        )
    )
    session.add(
        MoveArticle(
            move_id=int(moves["b"].id or 0),
            article_id=int(articles["hi"].id or 0),
            relevance=0.6,
            category="company",
            provider="heuristic",
        )
    )
    session.add(
        MoveArticle(
            move_id=int(moves["a"].id or 0),
            article_id=int(articles["lo"].id or 0),
            relevance=0.3,
            category="macro",
            provider="heuristic",
        )
    )
    session.commit()
    return Seeded(moves=moves, articles=articles)


def _labels(rows: list[Move]) -> list[str]:
    """Map moves back to their seed labels by date, for readable assertions."""
    by_date = {day: label for label, day, *_ in MOVE_ROWS}
    return [by_date[row.date] for row in rows]


# --------------------------------------------------------------------------- #
# Company and prices
# --------------------------------------------------------------------------- #


def test_get_company_and_has_prices(seeded: Seeded, session: Session) -> None:
    company = get_company(session, "test")
    assert company is not None
    assert company.name == "Test Corp"
    assert get_company(session, "NOPE") is None
    assert has_prices(session, TICKER) is True
    assert has_prices(session, "NOPE") is False


def test_list_prices_is_date_ascending_and_windowed(seeded: Seeded, session: Session) -> None:
    assert [p.date for p in list_prices(session, TICKER)] == [
        date(2025, 1, 2),
        date(2025, 2, 3),
    ]
    windowed = list_prices(session, TICKER, start=date(2025, 2, 1))
    assert [p.date for p in windowed] == [date(2025, 2, 3)]


# --------------------------------------------------------------------------- #
# list_moves
# --------------------------------------------------------------------------- #


def test_default_filters_are_z_or_pct_ordered_by_abs_z(seeded: Seeded, session: Session) -> None:
    rows = list_moves(session, TICKER, MoveFilters())
    # The four |z| >= 2 rows plus `e`, which is only a move by |ret| >= 0.02.
    # `c` clears neither threshold.
    assert _labels(rows) == ["f", "b", "a", "d", "e"]


def test_ticker_is_scoped(seeded: Seeded, session: Session) -> None:
    assert list_moves(session, "OTHER", MoveFilters()) == []


def test_direction_filter(seeded: Seeded, session: Session) -> None:
    rows = list_moves(session, TICKER, MoveFilters(direction="down"))
    assert _labels(rows) == ["b", "d"]


def test_z_threshold_needs_pct_raised_too(seeded: Seeded, session: Session) -> None:
    # The thresholds are OR-ed, so z alone still admits every 2%+ day.
    assert len(list_moves(session, TICKER, MoveFilters(z_threshold=3.0))) == 5
    rows = list_moves(session, TICKER, MoveFilters(z_threshold=3.0, pct_threshold=1.0))
    assert _labels(rows) == ["f", "b"]


def test_category_uses_the_explanation_when_there_is_one(seeded: Seeded, session: Session) -> None:
    # `a` routes "company" but is explained "macro"; nothing else is macro.
    rows = list_moves(session, TICKER, MoveFilters(category="macro"))
    assert _labels(rows) == ["a"]


def test_category_falls_back_to_routing_when_unexplained(seeded: Seeded, session: Session) -> None:
    # `b` is explained "company" despite routing "macro"; `d` and `f` are
    # unexplained with routing "company"; `a` is excluded by its explanation.
    rows = list_moves(session, TICKER, MoveFilters(category="company"))
    assert _labels(rows) == ["f", "b", "d"]


def test_start_end_window(seeded: Seeded, session: Session) -> None:
    rows = list_moves(session, TICKER, MoveFilters(start=date(2025, 4, 1), end=date(2025, 6, 30)))
    assert _labels(rows) == ["d", "e"]


def test_limit(seeded: Seeded, session: Session) -> None:
    assert _labels(list_moves(session, TICKER, MoveFilters(limit=2))) == ["f", "b"]


def test_get_move(seeded: Seeded, session: Session) -> None:
    move = get_move(session, "test", date(2025, 8, 1))
    assert move is not None
    assert move.ret_z == pytest.approx(4.0)
    assert get_move(session, TICKER, date(2025, 8, 2)) is None


# --------------------------------------------------------------------------- #
# Articles, explanations, news search
# --------------------------------------------------------------------------- #


def test_articles_for_move_orders_and_filters_by_relevance(
    seeded: Seeded, session: Session
) -> None:
    move_id = int(seeded.moves["a"].id or 0)
    rows = articles_for_move(session, move_id)
    assert [link.relevance for _, link in rows] == [0.9, 0.3]

    relevant = articles_for_move(session, move_id, min_relevance=0.5)
    assert len(relevant) == 1
    assert relevant[0][0].id == seeded.articles["hi"].id

    assert len(articles_for_move(session, move_id, limit=1)) == 1


def test_get_explanation(seeded: Seeded, session: Session) -> None:
    explained = get_explanation(session, int(seeded.moves["a"].id or 0))
    assert explained is not None
    assert explained.primary_category == "macro"
    assert get_explanation(session, int(seeded.moves["f"].id or 0)) is None


def test_search_news_is_case_insensitive_and_distinct(seeded: Seeded, session: Session) -> None:
    rows = search_news(session, TICKER, query="guidance")
    # The article is linked to two moves but comes back once, on its best link.
    assert len(rows) == 1
    article, link = rows[0]
    assert article.id == seeded.articles["hi"].id
    assert link.relevance == pytest.approx(0.9)

    assert len(search_news(session, TICKER)) == 2
    assert search_news(session, TICKER, query="no such words") == []


def test_search_news_window_relevance_and_limit(seeded: Seeded, session: Session) -> None:
    only_the_third = search_news(session, TICKER, start=date(2025, 2, 3))
    assert [a.id for a, _ in only_the_third] == [seeded.articles["lo"].id]

    assert len(search_news(session, TICKER, end=date(2025, 2, 2))) == 1
    assert len(search_news(session, TICKER, min_relevance=0.5)) == 1
    assert len(search_news(session, TICKER, limit=1)) == 1


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #


def test_chat_round_trip_is_oldest_first(session: Session) -> None:
    save_chat_message(session, "s1", "user", "biggest drop?")
    saved = save_chat_message(
        session,
        "s1",
        "assistant",
        "March 10.",
        tool_calls=[{"name": "list_moves", "input": {"limit": 1}}],
    )
    save_chat_message(session, "other", "user", "different session")

    assert saved.id is not None
    assert saved.tool_calls_json is not None
    assert "list_moves" in saved.tool_calls_json

    history = chat_history(session, "s1")
    assert [(m.role, m.content) for m in history] == [
        ("user", "biggest drop?"),
        ("assistant", "March 10."),
    ]
    assert chat_history(session, "s1", limit=1)[0].role == "assistant"
    assert chat_history(session, "missing") == []


# --------------------------------------------------------------------------- #
# Serialisers
# --------------------------------------------------------------------------- #


def test_move_to_dict_keys_and_optional_articles(seeded: Seeded, session: Session) -> None:
    move = seeded.moves["a"]
    bare = move_to_dict(move)
    assert set(bare) == MOVE_KEYS
    assert bare["explanation"] is None
    assert bare["date"] == "2025-02-03"
    assert bare["direction"] == "up"
    assert bare["near_fomc"] is False

    explanation = get_explanation(session, int(move.id or 0))
    articles = articles_for_move(session, int(move.id or 0))
    full = move_to_dict(move, explanation, articles)
    assert set(full) == MOVE_KEYS | {"articles"}
    assert full["explanation"] is not None
    assert full["explanation"]["primary_category"] == "macro"
    assert full["explanation"]["cited_article_ids"] == [1]
    assert [a["relevance"] for a in full["articles"]] == [0.9, 0.3]

    assert move_to_dict(move, None, [])["articles"] == []


def test_article_to_dict_without_a_link(seeded: Seeded, session: Session) -> None:
    payload = article_to_dict(seeded.articles["lo"])
    assert payload["relevance"] is None
    assert payload["category"] is None
    assert payload["published_at"] == "2025-02-03T09:30:00"
    assert payload["source"] == "Some Blog"


def test_explanation_to_dict_of_none() -> None:
    assert explanation_to_dict(None) is None


def test_serialisers_round_floats_and_drop_nan() -> None:
    price = Price(
        ticker=TICKER,
        date=date(2025, 2, 3),
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=1.0,
        ret=0.123456789,
        ret_z=float("nan"),
        vol_z=None,
    )
    payload = price_to_dict(price)
    assert payload["ret"] == 0.123457
    assert payload["ret_z"] is None
    assert payload["vol_z"] is None
    assert payload["routing"] is None
