"""Tests for the keyed provider.

No network and no API key: the SDK client is replaced by `FakeClient`, whose
`messages.parse` / `messages.create` pop pre-canned results off a queue and
record the kwargs they were called with. A queued `Exception` is raised
instead, which is how the heuristic fallback paths are exercised.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from stock_moves.providers.anthropic import (
    DEFAULT_MODEL,
    MAX_TOKENS_LONG,
    MAX_TOKENS_SHORT,
    AnthropicProvider,
)
from stock_moves.providers.base import (
    ArticleInput,
    ArticleScore,
    ChatTurn,
    MoveContext,
)

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeMessages:
    """The `client.messages` surface the provider uses, and nothing else."""

    def __init__(self) -> None:
        self.parse_queue: list[Any] = []
        self.create_queue: list[Any] = []
        self.parse_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> Any:
        self.parse_calls.append(kwargs)
        item = self.parse_queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(parsed_output=item)

    def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        item = self.create_queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self) -> None:
        self.messages = FakeMessages()


def provider(client: FakeClient) -> AnthropicProvider:
    return AnthropicProvider(api_key="test-key", client=client)


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_use_block(block_id: str, name: str, payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=payload)


def response(stop_reason: str, *blocks: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks))


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def ctx(**overrides: Any) -> MoveContext:
    base: dict[str, Any] = {
        "ticker": "AMD",
        "company_name": "Advanced Micro Devices Inc",
        "date": date(2025, 6, 3),
        "ret": -0.061,
        "ret_z": -2.8,
        "gap_ret": -0.04,
        "intraday_ret": -0.02,
        "vol_z": 2.4,
        "mkt_component": -0.004,
        "sector_component": -0.009,
        "idio_component": -0.048,
        "routing": "company",
        "direction": "down",
        "regime_mkt": "bull",
        "regime_sector": "bear",
        "near_earnings": True,
        "sector": "Technology",
        "industry": "Semiconductors",
        "peers": ("NVDA", "INTC"),
        "peer_comove": -0.012,
    }
    base.update(overrides)
    return MoveContext(**base)


def art(article_id: int, title: str, source: str | None = "Reuters") -> ArticleInput:
    return ArticleInput(
        id=article_id,
        title=title,
        source=source,
        url=f"https://example.test/{article_id}",
        published_at=datetime(2025, 6, 3, 13, 30, tzinfo=UTC),
    )


ARTICLES = [
    art(1, "Advanced Micro Devices cuts guidance after data-center miss"),
    art(2, "Semiconductors slide as chip sentiment sours"),
    art(3, "Local bakery wins award"),
]


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_defaults_to_sonnet_and_never_builds_a_real_client() -> None:
    client = FakeClient()
    prov = provider(client)
    assert prov.name == "anthropic"
    assert prov.model == DEFAULT_MODEL == "claude-sonnet-5"
    assert client.messages.parse_calls == []
    assert client.messages.create_calls == []


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def test_score_articles_maps_ids_and_fills_the_missing_ones() -> None:
    client = FakeClient()
    client.messages.parse_queue.append(
        SimpleNamespace(
            scores=[
                SimpleNamespace(article_id=2, relevance=0.6, category="industry"),
                # Out of order, over-range, and one id (3) never returned.
                SimpleNamespace(article_id=1, relevance=1.4, category="company"),
                SimpleNamespace(article_id=99, relevance=0.9, category="macro"),
            ]
        )
    )

    scores = provider(client).score_articles(ctx(), ARTICLES)

    assert scores == [
        ArticleScore(1, 1.0, "company"),  # clamped into [0, 1]
        ArticleScore(2, 0.6, "industry"),
        ArticleScore(3, 0.0, "company"),  # missing -> 0.0, routed by the move
    ]
    # Exactly one parse call, with the documented budget and the move's fields.
    assert len(client.messages.parse_calls) == 1
    call = client.messages.parse_calls[0]
    assert call["model"] == DEFAULT_MODEL
    assert call["max_tokens"] == MAX_TOKENS_SHORT
    content = call["messages"][0]["content"]
    assert '"ticker": "AMD"' in content
    assert "1 | 2025-06-03 | Reuters |" in content


def test_score_articles_needs_no_call_for_an_empty_list() -> None:
    client = FakeClient()
    assert provider(client).score_articles(ctx(), []) == []
    assert client.messages.parse_calls == []


def test_score_articles_falls_back_to_the_heuristic_when_the_client_raises() -> None:
    client = FakeClient()
    client.messages.parse_queue.append(RuntimeError("429 rate limited"))

    scores = provider(client).score_articles(ctx(), ARTICLES)

    # The heuristic's own verdicts: name hit, peer/industry hit, nothing.
    assert [s.relevance for s in scores] == [1.0, 0.7, 0.1]
    assert [s.category for s in scores] == ["company", "industry", "company"]


# --------------------------------------------------------------------------- #
# Explanation
# --------------------------------------------------------------------------- #


def scored() -> list[tuple[ArticleInput, ArticleScore]]:
    return [
        (ARTICLES[0], ArticleScore(1, 0.95, "company")),
        (ARTICLES[1], ArticleScore(2, 0.6, "industry")),
    ]


def test_explain_normalises_the_model_output() -> None:
    client = FakeClient()
    client.messages.parse_queue.append(
        SimpleNamespace(
            summary="  AMD fell 6.1% on 2025-06-03.  ",
            primary_category="company",
            confidence=1.4,
            cited_article_ids=[1, 1, 42],
            unexplained=False,
        )
    )

    result = provider(client).explain(ctx(), scored())

    assert result.summary == "AMD fell 6.1% on 2025-06-03."
    assert result.primary_category == "company"
    assert result.confidence == 1.0  # clamped
    assert result.cited_article_ids == (1,)  # deduped, id 42 was never offered
    assert result.unexplained is False
    assert client.messages.parse_calls[0]["max_tokens"] == MAX_TOKENS_LONG


def test_explain_keeps_the_unexplained_verdict_consistent() -> None:
    client = FakeClient()
    client.messages.parse_queue.append(
        SimpleNamespace(
            summary="No headline in the window explains the move.",
            primary_category="unexplained",
            confidence=0.2,
            cited_article_ids=[],
            unexplained=False,  # contradicts the category; the category wins
        )
    )

    result = provider(client).explain(ctx(), scored())

    assert result.primary_category == "unexplained"
    assert result.unexplained is True


def test_explain_falls_back_to_the_heuristic_when_the_client_raises() -> None:
    client = FakeClient()
    client.messages.parse_queue.append(TimeoutError("read timeout"))

    result = provider(client).explain(ctx(), scored())

    assert "AMD fell 6.1%" in result.summary
    assert result.primary_category == "company"
    assert result.cited_article_ids == (1, 2)


# --------------------------------------------------------------------------- #
# Peers
# --------------------------------------------------------------------------- #


def test_suggest_peers_upper_cases_dedupes_and_drops_self() -> None:
    client = FakeClient()
    client.messages.parse_queue.append(
        SimpleNamespace(tickers=["nvda", " intc ", "AMD", "NVDA", "", "mu"])
    )

    peers = provider(client).suggest_peers(
        "amd", "Advanced Micro Devices Inc", "Technology", "Semiconductors"
    )

    assert peers == ["NVDA", "INTC", "MU"]


def test_suggest_peers_returns_empty_when_the_client_raises() -> None:
    client = FakeClient()
    client.messages.parse_queue.append(ValueError("no parsed output"))

    assert provider(client).suggest_peers("AMD", "Advanced Micro Devices", None, None) == []


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #


def fake_tools() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The three read functions, recording the kwargs they were called with."""
    calls: list[dict[str, Any]] = []

    def get_move(**kwargs: Any) -> dict[str, Any]:
        calls.append({"name": "get_move", **kwargs})
        return {"date": kwargs.get("date"), "ret": -0.061, "articles": []}

    def list_moves(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append({"name": "list_moves", **kwargs})
        return [{"date": "2025-06-03", "ret": -0.061, "ret_z": -2.8}]

    def search_news(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append({"name": "search_news", **kwargs})
        return []

    tools: dict[str, Any] = {
        "get_move": get_move,
        "list_moves": list_moves,
        "search_news": search_news,
    }
    return tools, calls


def test_chat_runs_the_tool_with_the_default_ticker_injected() -> None:
    client = FakeClient()
    client.messages.create_queue += [
        response(
            "tool_use",
            tool_use_block("t1", "get_move", {"date": "2025-06-03"}),
        ),
        response("end_turn", text_block("AMD fell 6.1% on 2025-06-03.")),
    ]
    tools, calls = fake_tools()

    reply = provider(client).chat(
        [ChatTurn("user", "why did it drop on 2025-06-03?")], tools, "AMD"
    )

    assert reply.reply == "AMD fell 6.1% on 2025-06-03."
    assert len(reply.tool_calls) == 1
    record = reply.tool_calls[0]
    assert record.name == "get_move"
    assert record.input == {"date": "2025-06-03", "ticker": "AMD"}
    assert record.output == {"date": "2025-06-03", "ret": -0.061, "articles": []}
    # The tool really ran, with the ticker injected.
    assert calls == [{"name": "get_move", "date": "2025-06-03", "ticker": "AMD"}]

    # Two turns: the tool result went back as one user message, and the
    # default ticker reached the system prompt.
    assert len(client.messages.create_calls) == 2
    second = client.messages.create_calls[1]
    assert "Default ticker: AMD." in second["system"]
    tool_results = second["messages"][-1]["content"]
    assert tool_results[0]["type"] == "tool_result"
    assert tool_results[0]["tool_use_id"] == "t1"
    assert "is_error" not in tool_results[0]


def test_chat_reports_a_failing_tool_to_the_model_instead_of_raising() -> None:
    client = FakeClient()
    client.messages.create_queue += [
        response("tool_use", tool_use_block("t1", "get_move", {"date": "bad"})),
        response("end_turn", text_block("That date is not a trading day.")),
    ]

    def boom(**_: Any) -> dict[str, Any]:
        raise ValueError("no such move")

    reply = provider(client).chat(
        [ChatTurn("user", "why did AMD drop on bad?")], {"get_move": boom}, "AMD"
    )

    assert reply.reply == "That date is not a trading day."
    assert reply.tool_calls[0].output is None
    error_result = client.messages.create_calls[1]["messages"][-1]["content"][0]
    assert error_result["is_error"] is True
    assert "no such move" in error_result["content"]


def test_chat_falls_back_to_the_heuristic_when_the_client_raises() -> None:
    client = FakeClient()
    client.messages.create_queue.append(RuntimeError("401 invalid x-api-key"))
    tools, calls = fake_tools()

    reply = provider(client).chat(
        [ChatTurn("user", "why did AMD drop on 2025-06-03?")], tools, "AMD"
    )

    # The heuristic answered from the same tools rather than the error escaping.
    assert "2025-06-03" in reply.reply
    assert [record.name for record in reply.tool_calls] == ["get_move"]
    assert calls == [{"name": "get_move", "ticker": "AMD", "date": "2025-06-03"}]


def test_chat_stops_at_the_iteration_cap() -> None:
    client = FakeClient()
    # Always asking for a tool: the loop must stop, not spin.
    client.messages.create_queue += [
        response("tool_use", tool_use_block(f"t{i}", "list_moves", {})) for i in range(10)
    ]
    tools, _ = fake_tools()

    reply = provider(client).chat([ChatTurn("user", "tell me everything")], tools, "AMD")

    assert len(client.messages.create_calls) == 6
    assert len(reply.tool_calls) == 6
    assert "tool-call budget" in reply.reply


@pytest.mark.parametrize("history", [[], [ChatTurn("user", "")]])
def test_chat_with_no_usable_history_asks_for_a_question(
    history: list[ChatTurn],
) -> None:
    client = FakeClient()
    tools, _ = fake_tools()

    reply = provider(client).chat(history, tools, "AMD")

    assert reply.tool_calls == []
    assert client.messages.create_calls == []
    assert "Ask me" in reply.reply
