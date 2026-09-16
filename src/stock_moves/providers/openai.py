"""The OpenAI keyed provider (DESIGN section 4).

`gpt-4.1` through the official `openai` SDK, behind the same `ModelProvider`
protocol as the Anthropic provider and sharing its prompts and schemas from
`prompts.py`. The differences from `anthropic.py` are entirely transport:

- structured output is `chat.completions.parse(response_format=Model)` rather
  than `messages.parse(output_format=Model)`;
- the system prompt is the first message rather than a top-level argument;
- tool calls come back on `message.tool_calls` with JSON-string arguments, and
  each result goes back as its own `role="tool"` message rather than as blocks
  inside one user turn.

That list is the whole point of the interface: adding a second vendor did not
touch move detection, scoring policy, storage or the API.

Every method degrades to `HeuristicProvider` on *any* exception (bad key, no
credit, rate limit, timeout, malformed output). The app is specified to run
with zero keys, so a broken key must behave like no key at all rather than
fail a request. The broad excepts are deliberate and each is marked.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, TypeVar

from pydantic import BaseModel

from stock_moves.providers.base import (
    TOOL_SPECS,
    ArticleInput,
    ArticleScore,
    ChatReply,
    ChatTurn,
    ExplanationResult,
    MoveContext,
    ToolCallRecord,
    ToolFn,
)
from stock_moves.providers.heuristic import HeuristicProvider
from stock_moves.providers.prompts import (
    EXPLAIN_SYSTEM,
    PEERS_SYSTEM,
    SCORE_SYSTEM,
    Explanation,
    Peers,
    Scores,
    chat_system,
    clamp,
    explain_prompt,
    openai_tools,
    peers_prompt,
    score_prompt,
)

__all__ = ["OpenAIProvider"]

DEFAULT_MODEL = "gpt-4.1"
"""The workhorse tier, chosen for the prose rather than for reasoning: this
app makes roughly one call per move, and a reasoning model's latency is paid
ten times over on a single ingest. Overridable with `OPENAI_MODEL`."""

MAX_TOKENS_SHORT = 2048
"""Scoring and peers: small, fixed-shape JSON."""

MAX_TOKENS_LONG = 4096
"""Explanations and chat: prose."""

MAX_TOOL_ITERATIONS = 6
"""Hard stop on the chat tool loop, so a model that keeps calling tools cannot
run a request forever."""

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class OpenAIProvider:
    """`gpt-4.1` behind the `ModelProvider` protocol."""

    name: str = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        client: Any | None = None,
    ) -> None:
        """Build a provider.

        `client` is injected by the tests; in production it defaults to
        `openai.OpenAI(api_key=api_key)`. The SDK import is inside the
        constructor so importing this module never requires the SDK.
        """
        self.model = model
        if client is None:
            import openai

            client = openai.OpenAI(api_key=api_key)
        self._client = client
        self._fallback = HeuristicProvider()

    # ------------------------------------------------------------------ #
    # Structured-output plumbing
    # ------------------------------------------------------------------ #

    def _parse(
        self,
        system: str,
        content: str,
        response_format: type[_ModelT],
        max_tokens: int,
    ) -> _ModelT:
        """One `chat.completions.parse` call, returning the validated model.

        A refusal (the model declining rather than answering) is raised as an
        error so it takes the same degrade path as a transport failure: the
        caller's contract is "an explanation or the heuristic's", never a
        half-filled object.
        """
        response = self._client.chat.completions.parse(
            model=self.model,
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            response_format=response_format,
        )
        message = response.choices[0].message
        if getattr(message, "refusal", None):
            raise ValueError(f"the model refused: {message.refusal}")
        parsed = getattr(message, "parsed", None)
        if parsed is None:
            raise ValueError("the model returned no parsed output")
        return parsed

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def score_articles(
        self, move: MoveContext, articles: Sequence[ArticleInput]
    ) -> list[ArticleScore]:
        """Re-score the heuristic's top-K headlines in one call.

        Returns one score per input article, in input order. An id the model
        omitted gets relevance 0.0 filed under the move's own routing bucket:
        "not mentioned" is a score, not a missing row.
        """
        if not articles:
            return []
        try:
            parsed = self._parse(
                SCORE_SYSTEM, score_prompt(move, articles), Scores, MAX_TOKENS_SHORT
            )
            by_id = {int(score.article_id): score for score in parsed.scores}
            return [
                ArticleScore(
                    article_id=article.id,
                    relevance=clamp(float(by_id[article.id].relevance)),
                    category=str(by_id[article.id].category),
                )
                if article.id in by_id
                else ArticleScore(article.id, 0.0, move.routing)
                for article in articles
            ]
        except Exception:  # noqa: BLE001 - any model failure degrades, never raises
            return self._fallback.score_articles(move, articles)

    # ------------------------------------------------------------------ #
    # Explanation
    # ------------------------------------------------------------------ #

    def explain(
        self, move: MoveContext, scored: Sequence[tuple[ArticleInput, ArticleScore]]
    ) -> ExplanationResult:
        """Write the move's explanation from the numbers and the top-K articles."""
        try:
            parsed = self._parse(
                EXPLAIN_SYSTEM, explain_prompt(move, scored), Explanation, MAX_TOKENS_LONG
            )
        except Exception:  # noqa: BLE001 - any model failure degrades, never raises
            return replace(self._fallback.explain(move, scored), degraded=True)

        allowed = {article.id for article, _ in scored}
        cited = tuple(
            dict.fromkeys(
                int(article_id)
                for article_id in parsed.cited_article_ids
                if int(article_id) in allowed
            )
        )
        unexplained = bool(parsed.unexplained) or parsed.primary_category == "unexplained"
        category = "unexplained" if unexplained else str(parsed.primary_category)
        return ExplanationResult(
            summary=parsed.summary.strip(),
            primary_category=category,
            confidence=clamp(float(parsed.confidence)),
            cited_article_ids=cited,
            unexplained=unexplained,
        )

    # ------------------------------------------------------------------ #
    # Ontology
    # ------------------------------------------------------------------ #

    def suggest_peers(
        self, ticker: str, name: str, sector: str | None, industry: str | None
    ) -> list[str]:
        """Up to 6 peer tickers. Empty on any failure; the caller then falls
        back to the sector ETF's top holdings."""
        try:
            parsed = self._parse(
                PEERS_SYSTEM,
                peers_prompt(ticker, name, sector, industry),
                Peers,
                MAX_TOKENS_SHORT,
            )
        except Exception:  # noqa: BLE001 - peers are optional; never fail ingest
            return []

        self_ticker = ticker.strip().upper()
        cleaned = [
            candidate
            for candidate in (str(t).strip().upper() for t in parsed.tickers)
            if candidate and candidate != self_ticker
        ]
        return list(dict.fromkeys(cleaned))[:6]

    # ------------------------------------------------------------------ #
    # Chat
    # ------------------------------------------------------------------ #

    def chat(
        self,
        history: Sequence[ChatTurn],
        tools: Mapping[str, ToolFn],
        ticker: str | None,
    ) -> ChatReply:
        """A manual tool-calling loop over the read functions in `TOOL_SPECS`."""
        try:
            return self._chat(history, tools, ticker)
        except Exception:  # noqa: BLE001 - a failed loop answers from the heuristic
            return self._fallback.chat(history, tools, ticker)

    def _chat(
        self,
        history: Sequence[ChatTurn],
        tools: Mapping[str, ToolFn],
        ticker: str | None,
    ) -> ChatReply:
        messages: list[dict[str, Any]] = [{"role": "system", "content": chat_system(ticker)}]
        messages += [
            {"role": turn.role, "content": turn.content}
            for turn in history
            if turn.role in {"user", "assistant"} and turn.content
        ]
        if len(messages) == 1:
            return ChatReply("Ask me about a ticker's moves.", [])

        records: list[ToolCallRecord] = []
        reply = ""
        for _ in range(MAX_TOOL_ITERATIONS):
            response = self._client.chat.completions.create(
                model=self.model,
                max_completion_tokens=MAX_TOKENS_LONG,
                tools=openai_tools(TOOL_SPECS),
                messages=messages,
            )
            message = response.choices[0].message
            reply = (message.content or "").strip() or reply

            calls = list(getattr(message, "tool_calls", None) or [])
            if not calls:
                break
            # The assistant turn carrying the calls has to be echoed back
            # before their results, or the results reference nothing.
            messages.append(_assistant_turn(message, calls))
            results, new_records = self._run_tools(calls, tools, ticker)
            records.extend(new_records)
            messages.extend(results)

        if not reply:
            reply = "I could not reach an answer within the tool-call budget."
        return ChatReply(reply, records)

    def _run_tools(
        self,
        calls: Sequence[Any],
        tools: Mapping[str, ToolFn],
        ticker: str | None,
    ) -> tuple[list[dict[str, Any]], list[ToolCallRecord]]:
        """Run every tool call in one assistant turn.

        Unlike the Anthropic path, each result is its own `role="tool"`
        message. A tool that raises comes back as an error string in that same
        shape so the model can recover instead of the request dying.
        """
        results: list[dict[str, Any]] = []
        records: list[ToolCallRecord] = []
        for call in calls:
            name = call.function.name
            payload = _tool_arguments(call)
            if not payload.get("ticker") and ticker:
                payload["ticker"] = ticker

            tool = tools.get(name)
            if tool is None:
                results.append(_tool_message(call.id, f"unknown tool {name}"))
                records.append(ToolCallRecord(name, payload, None))
                continue
            try:
                output = tool(**payload)
            except Exception as exc:  # noqa: BLE001 - reported to the model, not raised
                results.append(_tool_message(call.id, f"{type(exc).__name__}: {exc}"))
                records.append(ToolCallRecord(name, payload, None))
                continue
            results.append(_tool_message(call.id, json.dumps(output, default=str)))
            records.append(ToolCallRecord(name, payload, output))
        return results, records


def _tool_arguments(call: Any) -> dict[str, Any]:
    """The call's arguments, which arrive as a JSON *string*.

    A model that emits malformed JSON gets an empty payload rather than an
    exception: the tool then runs on its defaults, or reports its own error
    back into the loop.
    """
    raw = getattr(call.function, "arguments", None) or "{}"
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _assistant_turn(message: Any, calls: Sequence[Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": getattr(call.function, "arguments", None) or "{}",
                },
            }
            for call in calls
        ],
    }


def _tool_message(tool_call_id: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}
