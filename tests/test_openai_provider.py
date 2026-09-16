"""Tests for the OpenAI provider. No network: the client is a fake that
queues responses and records the kwargs it was called with.

The Anthropic suite covers the same behaviours against the other SDK. What is
specific here is the transport that differs: a system message rather than a
top-level argument, tool arguments arriving as a JSON *string*, one
`role="tool"` message per result, and a refusal being a failure rather than an
answer.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from stock_moves.providers.base import (
    ArticleInput,
    ArticleScore,
    ChatTurn,
    MoveContext,
    ToolFn,
)
from stock_moves.providers.openai import (
    DEFAULT_MODEL,
    MAX_TOKENS_LONG,
    MAX_TOKENS_SHORT,
    OpenAIProvider,
)
from stock_moves.providers.prompts import Explanation, Peers, Scores

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeCompletions:
    """The `client.chat.completions` surface the provider uses, and no more."""

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
        if isinstance(item, str):  # a refusal
            return _completion(SimpleNamespace(parsed=None, refusal=item, content=None))
        return _completion(SimpleNamespace(parsed=item, refusal=None, content=None))

    def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        item = self.create_queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return _completion(item)


class FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions())


def _completion(message: Any) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def message(content: str | None, *tool_calls: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=list(tool_calls) or None, refusal=None)


def tool_call(call_id: str, name: str, payload: dict[str, Any]) -> SimpleNamespace:
    """A tool call as the SDK hands it over: arguments are a JSON string."""
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(payload)),
    )


def provider(client: FakeClient) -> OpenAIProvider:
    return OpenAIProvider(api_key="test-key", client=client)


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
]


def scored() -> list[tuple[ArticleInput, ArticleScore]]:
    return [
        (ARTICLES[0], ArticleScore(1, 0.95, "company")),
        (ARTICLES[1], ArticleScore(2, 0.6, "industry")),
    ]


def explanation(**overrides: Any) -> Explanation:
    base: dict[str, Any] = {
        "summary": "AMD was down 6.1%, about three times the size of a typical day.",
        "primary_category": "company",
        "confidence": 0.8,
        "cited_article_ids": [1, 2],
        "unexplained": False,
    }
    base.update(overrides)
    return Explanation(**base)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_the_default_model_is_the_workhorse_tier() -> None:
    assert DEFAULT_MODEL == "gpt-4.1"
    assert provider(FakeClient()).model == DEFAULT_MODEL


def test_the_model_is_overridable() -> None:
    assert OpenAIProvider(api_key="k", model="gpt-5.1", client=FakeClient()).model == "gpt-5.1"


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def test_score_sends_one_call_and_returns_input_order() -> None:
    client = FakeClient()
    client.chat.completions.parse_queue.append(
        Scores.model_validate(
            {
                "scores": [
                    {"article_id": 2, "relevance": 0.6, "category": "industry"},
                    {"article_id": 1, "relevance": 0.95, "category": "company"},
                ]
            }
        )
    )
    result = provider(client).score_articles(ctx(), ARTICLES)

    assert len(client.chat.completions.parse_calls) == 1
    assert [s.article_id for s in result] == [1, 2]
    assert result[0].relevance == pytest.approx(0.95)
    assert result[1].category == "industry"


def test_an_article_the_model_skipped_scores_zero_under_the_routing_bucket() -> None:
    client = FakeClient()
    client.chat.completions.parse_queue.append(
        Scores.model_validate(
            {"scores": [{"article_id": 1, "relevance": 0.9, "category": "company"}]}
        )
    )
    result = provider(client).score_articles(ctx(), ARTICLES)
    assert result[1].relevance == 0.0
    assert result[1].category == "company"  # the move's own routing


def test_relevance_is_clamped() -> None:
    client = FakeClient()
    client.chat.completions.parse_queue.append(
        Scores.model_validate(
            {"scores": [{"article_id": 1, "relevance": 7.0, "category": "company"}]}
        )
    )
    assert provider(client).score_articles(ctx(), [ARTICLES[0]])[0].relevance == 1.0


def test_no_articles_makes_no_call() -> None:
    client = FakeClient()
    assert provider(client).score_articles(ctx(), []) == []
    assert client.chat.completions.parse_calls == []


def test_score_falls_back_to_the_heuristic_when_the_client_raises() -> None:
    client = FakeClient()
    client.chat.completions.parse_queue.append(RuntimeError("429 rate limit"))
    result = provider(client).score_articles(ctx(), ARTICLES)
    # The keyless scorer's own verdict: the company is named in headline 1.
    assert result[0].relevance == 1.0
    assert result[0].category == "company"


# --------------------------------------------------------------------------- #
# Explanation
# --------------------------------------------------------------------------- #


def test_explain_sends_the_prompt_and_the_budget() -> None:
    client = FakeClient()
    client.chat.completions.parse_queue.append(explanation())
    result = provider(client).explain(ctx(), scored())

    call = client.chat.completions.parse_calls[0]
    assert call["max_completion_tokens"] == MAX_TOKENS_LONG
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1]["role"] == "user"
    assert result.confidence == pytest.approx(0.8)
    assert result.cited_article_ids == (1, 2)
    assert result.degraded is False


def test_the_narration_is_in_the_prompt() -> None:
    """The model is handed the numbers already said in English, so it is not
    the one converting a factor loading into prose."""
    client = FakeClient()
    client.chat.completions.parse_queue.append(explanation())
    provider(client).explain(ctx(), scored())

    sent = client.chat.completions.parse_calls[0]["messages"][1]["content"]
    assert "The same numbers in plain English:" in sent
    assert "typical day" in sent


def test_the_readability_contract_is_in_the_system_prompt() -> None:
    client = FakeClient()
    client.chat.completions.parse_queue.append(explanation())
    provider(client).explain(ctx(), scored())

    system = client.chat.completions.parse_calls[0]["messages"][0]["content"]
    for banned in ("sigma", "idiosyncratic", "z-score"):
        assert banned in system  # named in order to be forbidden


def test_a_cited_id_that_was_never_offered_is_dropped() -> None:
    client = FakeClient()
    client.chat.completions.parse_queue.append(explanation(cited_article_ids=[1, 99]))
    assert provider(client).explain(ctx(), scored()).cited_article_ids == (1,)


def test_unexplained_forces_the_category() -> None:
    client = FakeClient()
    client.chat.completions.parse_queue.append(explanation(unexplained=True))
    result = provider(client).explain(ctx(), scored())
    assert result.unexplained is True
    assert result.primary_category == "unexplained"


def test_explain_falls_back_to_the_heuristic_and_marks_it_degraded() -> None:
    client = FakeClient()
    client.chat.completions.parse_queue.append(TimeoutError("read timeout"))
    result = provider(client).explain(ctx(), scored())

    assert "was down 6.1%" in result.summary
    assert result.degraded is True


def test_a_refusal_degrades_rather_than_returning_half_an_answer() -> None:
    client = FakeClient()
    client.chat.completions.parse_queue.append("I cannot help with that")
    result = provider(client).explain(ctx(), scored())
    assert result.degraded is True


def test_no_credit_degrades_like_any_other_failure() -> None:
    """The failure this app actually met: a valid key on an unfunded account."""
    client = FakeClient()
    client.chat.completions.parse_queue.append(RuntimeError("You have no credits remaining."))
    result = provider(client).explain(ctx(), scored())
    assert result.degraded is True
    assert "was down 6.1%" in result.summary


# --------------------------------------------------------------------------- #
# Peers
# --------------------------------------------------------------------------- #


def test_suggest_peers_upper_cases_dedupes_and_drops_self() -> None:
    client = FakeClient()
    client.chat.completions.parse_queue.append(
        Peers.model_validate({"tickers": ["nvda", "NVDA", "amd", "intc"]})
    )
    assert provider(client).suggest_peers("AMD", "Advanced Micro Devices", "Tech", None) == [
        "NVDA",
        "INTC",
    ]


def test_peers_are_empty_on_failure_rather_than_failing_the_ingest() -> None:
    client = FakeClient()
    client.chat.completions.parse_queue.append(RuntimeError("bad key"))
    assert provider(client).suggest_peers("AMD", "Advanced Micro Devices", None, None) == []


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #


def fake_tools() -> tuple[dict[str, ToolFn], list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def get_move(**kwargs: Any) -> dict[str, Any]:
        calls.append({"name": "get_move", **kwargs})
        return {"date": "2025-06-03", "ret": -0.061, "articles": []}

    return {"get_move": get_move}, calls


def test_chat_runs_a_tool_and_answers() -> None:
    client = FakeClient()
    client.chat.completions.create_queue.append(
        message(None, tool_call("call_1", "get_move", {"date": "2025-06-03"}))
    )
    client.chat.completions.create_queue.append(message("AMD fell on a guidance cut."))
    tools, calls = fake_tools()

    reply = provider(client).chat([ChatTurn("user", "why did AMD drop?")], tools, "AMD")

    assert reply.reply == "AMD fell on a guidance cut."
    assert [record.name for record in reply.tool_calls] == ["get_move"]
    # The session ticker is filled in for a call that omitted it.
    assert calls == [{"name": "get_move", "date": "2025-06-03", "ticker": "AMD"}]


def test_the_tool_result_goes_back_as_its_own_tool_message() -> None:
    client = FakeClient()
    client.chat.completions.create_queue.append(
        message(None, tool_call("call_1", "get_move", {"date": "2025-06-03"}))
    )
    client.chat.completions.create_queue.append(message("Done."))
    tools, _ = fake_tools()

    provider(client).chat([ChatTurn("user", "why?")], tools, "AMD")

    sent = client.chat.completions.create_calls[1]["messages"]
    assert sent[-2]["role"] == "assistant" and sent[-2]["tool_calls"]
    assert sent[-1]["role"] == "tool"
    assert sent[-1]["tool_call_id"] == "call_1"
    assert json.loads(sent[-1]["content"])["date"] == "2025-06-03"


def test_the_system_prompt_leads_the_messages() -> None:
    client = FakeClient()
    client.chat.completions.create_queue.append(message("No tools needed."))
    provider(client).chat([ChatTurn("user", "hello")], {}, "AMD")

    sent = client.chat.completions.create_calls[0]["messages"]
    assert sent[0]["role"] == "system"
    assert "Default ticker: AMD." in sent[0]["content"]
    assert sent[1] == {"role": "user", "content": "hello"}


def test_the_tools_are_translated_into_openai_shape() -> None:
    client = FakeClient()
    client.chat.completions.create_queue.append(message("ok"))
    provider(client).chat([ChatTurn("user", "hi")], {}, None)

    tools = client.chat.completions.create_calls[0]["tools"]
    assert {tool["function"]["name"] for tool in tools} == {
        "list_moves",
        "get_move",
        "search_news",
    }
    assert all(tool["type"] == "function" for tool in tools)
    assert all("parameters" in tool["function"] for tool in tools)


def test_malformed_tool_arguments_do_not_kill_the_turn() -> None:
    client = FakeClient()
    broken = SimpleNamespace(
        id="call_1", function=SimpleNamespace(name="get_move", arguments="{not json")
    )
    client.chat.completions.create_queue.append(message(None, broken))
    client.chat.completions.create_queue.append(message("Recovered."))
    tools, calls = fake_tools()

    reply = provider(client).chat([ChatTurn("user", "why?")], tools, "AMD")
    assert reply.reply == "Recovered."
    assert calls == [{"name": "get_move", "ticker": "AMD"}]


def test_a_raising_tool_is_reported_back_to_the_model() -> None:
    client = FakeClient()
    client.chat.completions.create_queue.append(
        message(None, tool_call("call_1", "get_move", {"date": "bad"}))
    )
    client.chat.completions.create_queue.append(message("That date is not stored."))

    def boom(**_: Any) -> dict[str, Any]:
        raise ValueError("unparsable date")

    reply = provider(client).chat([ChatTurn("user", "why?")], {"get_move": boom}, "AMD")

    assert reply.reply == "That date is not stored."
    assert reply.tool_calls[0].output is None
    assert "unparsable date" in client.chat.completions.create_calls[1]["messages"][-1]["content"]


def test_an_unknown_tool_is_reported_rather_than_raised() -> None:
    client = FakeClient()
    client.chat.completions.create_queue.append(
        message(None, tool_call("call_1", "no_such_tool", {}))
    )
    client.chat.completions.create_queue.append(message("I cannot do that."))

    reply = provider(client).chat([ChatTurn("user", "why?")], {}, "AMD")
    assert reply.reply == "I cannot do that."
    assert (
        "unknown tool no_such_tool"
        in (client.chat.completions.create_calls[1]["messages"][-1]["content"])
    )


def test_chat_stops_at_the_iteration_cap() -> None:
    client = FakeClient()
    for index in range(10):
        client.chat.completions.create_queue.append(
            message(None, tool_call(f"call_{index}", "get_move", {"date": "2025-06-03"}))
        )
    tools, _ = fake_tools()

    reply = provider(client).chat([ChatTurn("user", "why?")], tools, "AMD")

    assert len(client.chat.completions.create_calls) == 6
    assert "could not reach an answer" in reply.reply


def test_chat_falls_back_to_the_heuristic_when_the_client_raises() -> None:
    client = FakeClient()
    client.chat.completions.create_queue.append(RuntimeError("401 invalid api key"))
    tools, calls = fake_tools()

    reply = provider(client).chat(
        [ChatTurn("user", "why did AMD drop on 2025-06-03?")], tools, "AMD"
    )

    assert "Tuesday, 3 June 2025" in reply.reply
    assert calls == [{"name": "get_move", "ticker": "AMD", "date": "2025-06-03"}]


def test_an_empty_history_asks_for_a_question() -> None:
    client = FakeClient()
    reply = provider(client).chat([], {}, "AMD")
    assert reply.reply == "Ask me about a ticker's moves."
    assert client.chat.completions.create_calls == []


def test_the_short_budget_is_used_for_scoring() -> None:
    client = FakeClient()
    client.chat.completions.parse_queue.append(Scores.model_validate({"scores": []}))
    provider(client).score_articles(ctx(), ARTICLES)
    assert client.chat.completions.parse_calls[0]["max_completion_tokens"] == MAX_TOKENS_SHORT
