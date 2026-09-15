"""The keyed provider (DESIGN section 4).

`claude-sonnet-5` through the official `anthropic` SDK. Three of the four
methods are a single structured-output call (`messages.parse` with a Pydantic
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
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal, TypeVar

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

_CATEGORY = Literal["company", "industry", "macro"]
_PRIMARY_CATEGORY = Literal["company", "industry", "macro", "unexplained"]

_ModelT = TypeVar("_ModelT", bound=BaseModel)


# --------------------------------------------------------------------------- #
# Structured output schemas
# --------------------------------------------------------------------------- #


class _Score(BaseModel):
    article_id: int
    relevance: float
    category: _CATEGORY


class _Scores(BaseModel):
    scores: list[_Score]


class _Explanation(BaseModel):
    summary: str
    primary_category: _PRIMARY_CATEGORY
    confidence: float
    cited_article_ids: list[int]
    unexplained: bool


class _Peers(BaseModel):
    tickers: list[str]


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_SCORE_SYSTEM = (
    "You are scoring news headlines for whether they explain a given stock's "
    "move on a given day. For each article return a relevance in [0, 1]: 1 "
    "means the headline is directly about this company on this day, 0 means it "
    "is unrelated to the move. The category is what the headline itself is "
    "about: 'company' for this specific company, 'industry' for its sector or "
    "peers, 'macro' for the economy, rates, policy or the whole market. "
    "Headlines are all you get -- there is no article body. Return one score "
    "per given article id and no ids that were not given."
)

_EXPLAIN_SYSTEM = (
    "You explain why a stock moved on one day. This is attribution, not "
    "causation: say what the evidence supports, never that one event caused "
    "the move. Cite only the article ids you are given, by id. Set "
    "`unexplained` true and `primary_category` 'unexplained' when the evidence "
    "is weak -- that is a correct answer, not a failure. Write 2-4 sentences. "
    "Mention the market/sector/idiosyncratic decomposition, the market and "
    "sector regime, proximity to earnings, proximity to an FOMC decision or a "
    "CPI release, and peer co-movement when they are informative; skip them "
    "when they are not. Confidence is in [0, 1] and should reflect how much "
    "the headlines and the decomposition agree."
)

_PEERS_SYSTEM = (
    "You name public-company peers for a given company. Return up to 6 "
    "US-listed public companies, by ticker symbol only, in upper case. No "
    "ETFs, no indices, no private companies, and never the company's own "
    "ticker. Closest competitors first; return fewer, or none, rather than "
    "padding the list."
)

_CHAT_SYSTEM = (
    "You explain stock moves using only the tools' results; cite dates; if the "
    "tools return nothing say so."
)


# --------------------------------------------------------------------------- #
# Prompt rendering helpers
# --------------------------------------------------------------------------- #


def _round(value: float | None, digits: int = 5) -> float | None:
    return None if value is None else round(value, digits)


def _move_block(move: MoveContext) -> str:
    """The move's quantitative context as a compact JSON object.

    Nulls are dropped: an absent decomposition should read as absent, not as
    a field the model has to reason about.
    """
    payload: dict[str, Any] = {
        "ticker": move.ticker,
        "company": move.company_name,
        "date": str(move.date),
        "sector": move.sector,
        "industry": move.industry,
        "ret": _round(move.ret),
        "ret_z": _round(move.ret_z, 3),
        "gap_ret": _round(move.gap_ret),
        "intraday_ret": _round(move.intraday_ret),
        "vol_z": _round(move.vol_z, 3),
        "mkt_component": _round(move.mkt_component),
        "sector_component": _round(move.sector_component),
        "idio_component": _round(move.idio_component),
        "routing": move.routing,
        "direction": move.direction,
        "regime_mkt": move.regime_mkt,
        "regime_sector": move.regime_sector,
        "near_earnings": move.near_earnings,
        "near_fomc": move.near_fomc,
        "near_cpi": move.near_cpi,
        "peers": list(move.peers) or None,
        "peer_comove": _round(move.peer_comove),
    }
    return json.dumps(
        {key: value for key, value in payload.items() if value is not None},
        default=str,
    )


def _article_date(article: ArticleInput) -> str:
    published = article.published_at
    return published.date().isoformat() if published is not None else "unknown"


def _article_line(article: ArticleInput) -> str:
    """`id | date | source | title` -- the four fields v1's news sources give."""
    return (
        f"{article.id} | {_article_date(article)} | {article.source or 'unknown'} | {article.title}"
    )


def _scored_line(article: ArticleInput, score: ArticleScore) -> str:
    return f"{_article_line(article)} | relevance={score.relevance:.2f} | category={score.category}"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


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

        lines = "\n".join(_article_line(article) for article in articles)
        content = (
            f"Move:\n{_move_block(move)}\n\n"
            f"Articles (id | date | source | title):\n{lines}\n\n"
            f"Score these {len(articles)} article ids."
        )
        try:
            parsed = self._parse(_SCORE_SYSTEM, content, _Scores, MAX_TOKENS_SHORT)
            by_id = {int(score.article_id): score for score in parsed.scores}
            return [
                ArticleScore(
                    article_id=article.id,
                    relevance=_clamp(float(by_id[article.id].relevance)),
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
        if scored:
            lines = "\n".join(_scored_line(article, score) for article, score in scored)
        else:
            lines = "(no headlines in the window)"
        content = (
            f"Move:\n{_move_block(move)}\n\n"
            f"Candidate articles (id | date | source | title | relevance | "
            f"category), best first:\n{lines}\n\n"
            "Explain this move."
        )
        try:
            parsed = self._parse(_EXPLAIN_SYSTEM, content, _Explanation, MAX_TOKENS_LONG)
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
            confidence=_clamp(float(parsed.confidence)),
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
        content = json.dumps(
            {
                "ticker": ticker,
                "company": name,
                "sector": sector,
                "industry": industry,
            }
        )
        try:
            parsed = self._parse(_PEERS_SYSTEM, content, _Peers, MAX_TOKENS_SHORT)
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
        system = _CHAT_SYSTEM + (
            f" Default ticker: {ticker}."
            if ticker
            else " There is no default ticker; ask for one if a tool needs it."
        )
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
