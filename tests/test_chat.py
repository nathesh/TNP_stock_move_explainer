"""Tests for `POST /chat` (DESIGN section 6).

No network and no key: the `no_api_key` fixture strips `ANTHROPIC_API_KEY`
(which `.env` may have put in the environment) and the provider cache is
cleared on both sides, so the request is answered by `HeuristicProvider`
against an in-memory database. That makes the assertions here about the route
— session minting, history, persistence, tool binding — and not about a model.

One company, one move and one linked article is enough: the heuristic router
picks a different tool for each of the three questions asked below, which is
how the tool bindings get exercised through the real dependency graph.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime

import pytest
from fastapi.testclient import TestClient
from sqlmodel import col, func, select

from stock_moves.api import deps
from stock_moves.api.app import create_app
from stock_moves.db import Session, configure_engine, get_engine, init_db
from stock_moves.models import (
    Article,
    ChatMessage,
    Company,
    Explanation,
    Move,
    MoveArticle,
)

TICKER = "TEST"
MOVE_DATE = date(2025, 6, 3)
SUMMARY = "Guidance cut."


@pytest.fixture
def client(no_api_key: None) -> Iterator[TestClient]:
    """A client whose app is bound to a freshly seeded in-memory database.

    `configure_engine` runs before `create_app`, so the lifespan's `init_db()`
    and every `get_db` session share this engine's single pooled connection —
    the rows seeded here are the rows the route reads.
    """
    engine = configure_engine("sqlite://")
    init_db(engine)
    with Session(engine) as seed_session:
        _seed(seed_session)

    deps.reset_provider_cache()
    with TestClient(create_app()) as test_client:
        yield test_client
    deps.reset_provider_cache()
    engine.dispose()


def _seed(session: Session) -> None:
    """One company, one explained move, one linked headline."""
    session.add(
        Company(
            ticker=TICKER,
            name="Testco Industries",
            sector="Technology",
            industry="Semiconductors",
            sector_etf="XLK",
        )
    )
    move = Move(
        ticker=TICKER,
        date=MOVE_DATE,
        ret=-0.05,
        ret_z=-2.5,
        gap_ret=-0.03,
        intraday_ret=-0.02,
        vol_z=2.1,
        mkt_component=-0.001,
        sector_component=-0.002,
        idio_component=-0.047,
        routing="company",
        direction="down",
    )
    article = Article(
        url="https://example.test/testco-guidance",
        title="Testco cuts full-year guidance",
        source="Reuters",
        published_at=datetime(2025, 6, 3, 12, 30, tzinfo=UTC).replace(tzinfo=None),
        news_source="google_rss",
    )
    session.add(move)
    session.add(article)
    session.commit()
    session.refresh(move)
    session.refresh(article)
    assert move.id is not None
    assert article.id is not None

    session.add(
        Explanation(
            move_id=move.id,
            summary=SUMMARY,
            primary_category="company",
            confidence=0.8,
            provider="heuristic",
        )
    )
    session.add(
        MoveArticle(
            move_id=move.id,
            article_id=article.id,
            relevance=0.9,
            category="company",
            provider="heuristic",
        )
    )
    session.commit()


def _n_messages(session_id: str) -> int:
    """Stored turns for one chat session, read on a fresh session."""
    with Session(get_engine()) as session:
        statement = (
            select(func.count())
            .select_from(ChatMessage)
            .where(col(ChatMessage.session_id) == session_id)
        )
        return int(session.exec(statement).one())


def test_dated_question_calls_get_move_and_returns_the_explanation(
    client: TestClient,
) -> None:
    response = client.post("/chat", json={"message": f"what happened to {TICKER} on {MOVE_DATE}?"})
    assert response.status_code == 200

    body = response.json()
    assert SUMMARY in body["reply"]
    assert body["session_id"]
    assert body["tool_calls"][0]["name"] == "get_move"
    # The tool output is the same dict the ticker route serves, articles included.
    output = body["tool_calls"][0]["output"]
    assert output["date"] == MOVE_DATE.isoformat()
    assert output["explanation"]["summary"] == SUMMARY
    assert [a["title"] for a in output["articles"]] == ["Testco cuts full-year guidance"]


def test_second_turn_reuses_the_session_and_persists_both_turns(
    client: TestClient,
) -> None:
    first = client.post(
        "/chat", json={"message": f"what happened to {TICKER} on {MOVE_DATE}?"}
    ).json()
    session_id = first["session_id"]

    second = client.post(
        "/chat", json={"message": f"any news on {TICKER}", "session_id": session_id}
    )
    assert second.status_code == 200

    body = second.json()
    assert body["session_id"] == session_id
    assert body["tool_calls"][0]["name"] == "search_news"
    assert "Testco cuts full-year guidance" in body["reply"]
    # Two turns each, user and assistant, in the order they were sent.
    assert _n_messages(session_id) == 4


def test_direction_word_reaches_list_moves_as_a_filter(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "why did it drop", "ticker": TICKER})
    assert response.status_code == 200

    call = response.json()["tool_calls"][0]
    assert call["name"] == "list_moves"
    assert call["input"]["direction"] == "down"
    # A `direction` filter that matched: the seeded move is a down day.
    assert [move["date"] for move in call["output"]] == [MOVE_DATE.isoformat()]


def test_session_ticker_answers_a_question_that_names_no_ticker(
    client: TestClient,
) -> None:
    response = client.post("/chat", json={"message": "biggest moves please", "ticker": "test"})
    assert response.status_code == 200

    call = response.json()["tool_calls"][0]
    assert call["name"] == "list_moves"
    assert [move["date"] for move in call["output"]] == [MOVE_DATE.isoformat()]


def test_no_ticker_anywhere_is_a_message_not_a_crash(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "so what happened?"})
    assert response.status_code == 200

    body = response.json()
    assert body["tool_calls"] == []
    assert "ticker" in body["reply"].lower()
