"""The Anthropic keyed provider (DESIGN section 4).

`claude-sonnet-5` through the official `anthropic` SDK. The prompts, the
structured-output schemas and the rendering of a move into a prompt are shared
with the OpenAI provider in `prompts.py`; what is left here is transport.

Three of the four methods are a single structured-output call (`messages.parse` with a Pydantic
`output_format`), which is what keeps the cost story in DESIGN section 4 true:
roughly one call per move, not one per headline. `chat` is the only multi-turn
path -- a manual tool-calling loop over `TOOL_SPECS`.

Every method degrades to `HeuristicProvider` on *any* exception (bad key,
rate limit, timeout, malformed output). The app is specified to run with zero
keys, so a broken key must behave like no key at all rather than fail a
request. The broad excepts are deliberate and each is marked.
"""

from __future__ import annotations

import json
import logging
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
    Relations,
    ToolCallRecord,
    ToolFn,
)
from stock_moves.providers.heuristic import HeuristicProvider
from stock_moves.providers.prompts import (
    EXPLAIN_SYSTEM,
    MAX_COUNTRIES,
    MAX_PEERS,
    MAX_SUPPLY_CHAIN,
    PEERS_SYSTEM,
    RELATIONS_SYSTEM,
    SCORE_SYSTEM,
    Explanation,
    Peers,
    RelationsOut,
    Scores,
    chat_system,
    clamp,
    clean_tickers,
    explain_prompt,
    normalise_countries,
    peers_prompt,
    relations_prompt,
    score_prompt,
)

__all__ = ["AnthropicProvider"]

DEFAULT_MODEL = "claude-sonnet-5"
"""Sonnet is fixed as the default by DESIGN section 4 (never Opus). Settings
normally pass the model in; this only matters as a fallback."""

MAX_TOKENS_SHORT = 2048
"""Scoring and peers: small, fixed-shape JSON."""

MAX_TOKENS_LONG = 4096
"""Explanations and chat: prose."""

MAX_TOOL_ITERATIONS = 6
"""Hard stop on the chat tool loop, so a model that keeps calling tools cannot
run a request forever."""

logger = logging.getLogger(__name__)

_NO_RELATIONS = Relations(competitors=(), suppliers=(), customers=(), countries=())
"""What a failed `suggest_relations` returns: the same value the keyless
provider returns, so a lost call degrades to "no edges", never to bad ones."""

_ModelT = TypeVar("_ModelT", bound=BaseModel)

# --------------------------------------------------------------------------- #
# The provider
# --------------------------------------------------------------------------- #


class AnthropicProvider:
    """`claude-sonnet-5` behind the `ModelProvider` protocol."""

    name: str = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        client: Any | None = None,
    ) -> None:
        """Build a provider.

        `client` is injected by the tests; in production it defaults to
        `anthropic.Anthropic(api_key=api_key)`. The SDK import is inside the
        constructor so importing this module never requires the SDK.
        """
        self.model = model
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
        self._client = client
        self._fallback = HeuristicProvider()

    # ------------------------------------------------------------------ #
    # Structured-output plumbing
    # ------------------------------------------------------------------ #

    def _parse(
        self,
        system: str,
        content: str,
        output_format: type[_ModelT],
        max_tokens: int,
    ) -> _ModelT:
        """One `messages.parse` call, returning the validated Pydantic model."""
        response = self._client.messages.parse(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_format=output_format,
        )
        parsed = getattr(response, "parsed_output", None)
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
        unexplained = bool(parsed.unexplained) or parsed.primary_category == ("unexplained")
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

    def suggest_relations(
        self, ticker: str, name: str, sector: str | None, industry: str | None
    ) -> Relations:
        """The company's competitors, suppliers, customers and countries, in
        one call (v1.5 decision 3).

        Same transport and the same degrade contract as `suggest_peers`: one
        structured-output call, and any failure at all returns the empty
        `Relations` rather than raising. Empty is a meaningful answer here --
        it is exactly what the keyless provider returns -- so an ingest that
        loses this call stores no edges instead of failing.
        """
        try:
            parsed = self._parse(
                RELATIONS_SYSTEM,
                relations_prompt(ticker, name, sector, industry),
                RelationsOut,
                MAX_TOKENS_SHORT,
            )
        except Exception as exc:  # noqa: BLE001 - edges are optional; never fail ingest
            logger.warning("%s: suggest_relations failed for %s: %s", self.name, ticker, exc)
            return _NO_RELATIONS

        try:
            return Relations(
                competitors=clean_tickers(parsed.competitors, ticker, MAX_PEERS),
                suppliers=clean_tickers(parsed.suppliers, ticker, MAX_SUPPLY_CHAIN),
                customers=clean_tickers(parsed.customers, ticker, MAX_SUPPLY_CHAIN),
                countries=normalise_countries(parsed.countries, MAX_COUNTRIES),
            )
        except Exception as exc:  # noqa: BLE001 - a malformed field is a failed call
            logger.warning("%s: unusable relations for %s: %s", self.name, ticker, exc)
            return _NO_RELATIONS

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
        system = chat_system(ticker)
        messages: list[dict[str, Any]] = [
            {"role": turn.role, "content": turn.content}
            for turn in history
            if turn.role in {"user", "assistant"} and turn.content
        ]
        if not messages:
            return ChatReply("Ask me about a ticker's moves.", [])

        records: list[ToolCallRecord] = []
        reply = ""
        for _ in range(MAX_TOOL_ITERATIONS):
            response = self._client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS_LONG,
                system=system,
                tools=TOOL_SPECS,
                messages=messages,
            )
            reply = _reply_text(response) or reply
            if getattr(response, "stop_reason", None) != "tool_use":
                break
            messages.append({"role": "assistant", "content": response.content})
            results, new_records = self._run_tools(response, tools, ticker)
            records.extend(new_records)
            messages.append({"role": "user", "content": results})

        if not reply:
            reply = "I could not reach an answer within the tool-call budget."
        return ChatReply(reply, records)

    def _run_tools(
        self,
        response: Any,
        tools: Mapping[str, ToolFn],
        ticker: str | None,
    ) -> tuple[list[dict[str, Any]], list[ToolCallRecord]]:
        """Run every `tool_use` block in one assistant turn.

        All results go back in a single user message, which is what the API
        expects for parallel tool use. A tool that raises comes back as an
        error result so the model can recover instead of the request dying.
        """
        results: list[dict[str, Any]] = []
        records: list[ToolCallRecord] = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            payload: dict[str, Any] = dict(block.input or {})
            if not payload.get("ticker") and ticker:
                payload["ticker"] = ticker

            tool = tools.get(block.name)
            if tool is None:
                results.append(_tool_result(block.id, f"unknown tool {block.name}", True))
                records.append(ToolCallRecord(block.name, payload, None))
                continue
            try:
                output = tool(**payload)
            except Exception as exc:  # noqa: BLE001 - reported to the model, not raised
                results.append(_tool_result(block.id, f"{type(exc).__name__}: {exc}", True))
                records.append(ToolCallRecord(block.name, payload, None))
                continue
            results.append(_tool_result(block.id, output, False))
            records.append(ToolCallRecord(block.name, payload, output))
        return results, records


def _tool_result(tool_use_id: str, content: Any, is_error: bool) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content if is_error else json.dumps(content, default=str),
    }
    if is_error:
        block["is_error"] = True
    return block


def _reply_text(response: Any) -> str:
    """Every text block of one response, concatenated."""
    texts = [
        str(block.text).strip()
        for block in getattr(response, "content", None) or []
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    return "\n".join(text for text in texts if text)
