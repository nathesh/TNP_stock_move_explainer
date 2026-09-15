"""The keyless provider (DESIGN section 4).

Not a stub: it produces a relevance, a category and a templated explanation
from the same inputs the Anthropic provider sees, so every endpoint and every
test works with zero API keys and zero network calls. It also runs first on
every headline when a key *is* present, cutting the article set to top-K
before any token is spent.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from stock_moves.providers.base import (
    MACRO_TERMS,
    ArticleInput,
    ArticleScore,
    ChatReply,
    ChatTurn,
    ExplanationResult,
    MoveContext,
    ToolCallRecord,
    ToolFn,
)

__all__ = ["HeuristicProvider"]

# Corporate suffixes carry no signal and stop a phrase match ("Nvidia Corp"
# never appears in a headline; "Nvidia" does).
_NAME_SUFFIXES: frozenset[str] = frozenset(
    {"inc", "corp", "corporation", "ltd", "plc", "co", "holdings"}
)

_PUNCT_RE = re.compile(r"[^a-z0-9]+")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")

# All-caps words that are not tickers.
_TICKER_STOPLIST: frozenset[str] = frozenset(
    {
        "I",
        "A",
        "THE",
        "WHY",
        "DID",
        "WHAT",
        "HOW",
        "IS",
        "IT",
        "ON",
        "IN",
        "OF",
        "TO",
        "AND",
        "OR",
        "FOR",
        "USD",
        "CEO",
        "AI",
        "US",
        "ETF",
    }
)

_DOWN_WORDS: tuple[str, ...] = ("drop", "fall", "down", "plunge", "sank")
_UP_WORDS: tuple[str, ...] = ("jump", "rise", "up", "rally", "soar")

_COMPONENT_LABELS: tuple[str, str, str] = ("market", "sector", "idiosyncratic")

#: Words that name a *specific event* rather than commentary about the stock.
#: A headline that matches the company but none of these ("Can TEST stock reach
#: $350?") is coverage, not a cause, so the explanation says so and the
#: confidence is cut.
EVENT_TERMS: tuple[str, ...] = (
    "fall",
    "falls",
    "fell",
    "drop",
    "drops",
    "slump",
    "slide",
    "plunge",
    "sink",
    "tumble",
    "jump",
    "surge",
    "soar",
    "rally",
    "rise",
    "beat",
    "beats",
    "miss",
    "misses",
    "cut",
    "cuts",
    "raise",
    "raises",
    "forecast",
    "guidance",
    "outlook",
    "earnings",
    "revenue",
    "downgrade",
    "upgrade",
    "recall",
    "lawsuit",
    "probe",
    "investigation",
    "acquire",
    "acquisition",
    "merger",
    "layoff",
    "ceo",
    "resign",
    "tariff",
    "ban",
    "approval",
    "fda",
    "settlement",
)

WEAK_ATTRIBUTION_NOTE = (
    "The matched headlines mention the company but do not name a specific "
    "event, so this attribution is weak."
)
"""Appended to the summary when no relevant headline names an event."""

WEAK_ATTRIBUTION_PENALTY = 0.25
WEAK_ATTRIBUTION_FLOOR = 0.2


def _normalise(text: str) -> str:
    """Lower-case, punctuation to single spaces, trimmed."""
    return _PUNCT_RE.sub(" ", text.lower()).strip()


def _has_word(haystack: str, needle: str) -> bool:
    """Whole-word (well, whole-token) containment in an already normalised
    string. Used so "co" does not match "cocoa" and "AMD" does not match
    "amdocs"."""
    if not needle:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])"
    return re.search(pattern, haystack) is not None


def _names_an_event(title: str) -> bool:
    """True when the title contains an `EVENT_TERMS` word as a whole word."""
    normalised = _normalise(title)
    return any(_has_word(normalised, term) for term in EVENT_TERMS)


def _company_name_forms(name: str) -> tuple[str, ...]:
    """The strings a headline might use for this company: the cleaned full
    name, plus its first word when that is a distinctive stand-alone token
    ("Advanced Micro Devices" -> also "advanced"; "Ford Motor" -> "ford")."""
    words = [w for w in _normalise(name).split() if w not in _NAME_SUFFIXES]
    if not words:
        return ()
    forms = [" ".join(words)]
    if len(words) >= 2 and len(words[0]) >= 4:
        forms.append(words[0])
    return tuple(forms)


def _last_user_text(history: Sequence[ChatTurn]) -> str:
    for turn in reversed(history):
        if turn.role == "user":
            return turn.content or ""
    return history[-1].content if history else ""


def _guess_ticker(text: str) -> str | None:
    for match in _TICKER_RE.finditer(text):
        token = match.group(0)
        if token not in _TICKER_STOPLIST:
            return token
    return None


def _guess_direction(lowered: str) -> str | None:
    if any(w in lowered for w in _DOWN_WORDS):
        return "down"
    if any(w in lowered for w in _UP_WORDS):
        return "up"
    return None


def _as_list(output: Any, key: str) -> list[dict[str, Any]]:
    """Tolerate a bare list or a {key: [...]} envelope from queries.py."""
    if isinstance(output, list):
        return [row for row in output if isinstance(row, dict)]
    if isinstance(output, dict):
        inner = output.get(key)
        if isinstance(inner, list):
            return [row for row in inner if isinstance(row, dict)]
    return []


def _fmt_pct(value: Any, digits: int = 1) -> str:
    return f"{value:+.{digits}%}" if isinstance(value, (int, float)) else ""


def _article_line(article: Mapping[str, Any]) -> str:
    bits = [
        str(article.get("published_at") or "")[:10],
        str(article.get("title") or ""),
    ]
    source = article.get("source")
    if source:
        bits.append(f"({source})")
    return " ".join(b for b in bits if b)


def _move_line(move: Mapping[str, Any]) -> str:
    explanation = move.get("explanation") or {}
    summary = explanation.get("summary") if isinstance(explanation, Mapping) else None
    z = move.get("ret_z")
    cells = [
        str(move.get("date") or ""),
        _fmt_pct(move.get("ret")),
        f"z={z:+.1f}" if isinstance(z, (int, float)) else "",
        str(move.get("routing") or ""),
        str(summary or ""),
    ]
    return "  ".join(cells).rstrip()


class HeuristicProvider:
    """Keyword rules plus the decomposition. No external calls."""

    name: str = "heuristic"

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def score_articles(
        self, move: MoveContext, articles: Sequence[ArticleInput]
    ) -> list[ArticleScore]:
        """Relevance from entity hits, category from what was hit.

        1.0/company for the company name or its ticker, 0.7/industry for a
        peer ticker or the industry term, 0.5/macro for the macro vocabulary,
        else 0.1 filed under the move's own routing bucket.
        """
        name_forms = _company_name_forms(move.company_name)
        ticker = move.ticker.lower()
        peers = tuple(p.lower() for p in move.peers if len(p) >= 2)
        industry = _normalise(move.industry or "")

        scores: list[ArticleScore] = []
        for article in articles:
            title = article.title.lower()
            norm = _normalise(article.title)
            if any(_has_word(norm, form) for form in name_forms) or (
                len(ticker) >= 3 and _has_word(norm, ticker)
            ):
                scores.append(ArticleScore(article.id, 1.0, "company"))
            elif any(_has_word(norm, peer) for peer in peers) or (
                industry != "" and industry in norm
            ):
                scores.append(ArticleScore(article.id, 0.7, "industry"))
            elif any(term in title for term in MACRO_TERMS):
                scores.append(ArticleScore(article.id, 0.5, "macro"))
            else:
                scores.append(ArticleScore(article.id, 0.1, move.routing))
        return scores

    # ------------------------------------------------------------------ #
    # Explanation
    # ------------------------------------------------------------------ #

    def explain(
        self, move: MoveContext, scored: Sequence[tuple[ArticleInput, ArticleScore]]
    ) -> ExplanationResult:
        """Template the numbers. Returns `unexplained` when nothing in the
        window matched and the move is not extreme enough to assert a cause
        on the decomposition alone."""
        relevant = [(a, s) for a, s in scored if s.relevance >= 0.5]

        if not relevant and abs(move.ret_z) < 3:
            summary = (
                f"{move.ticker} moved {move.ret:+.1%} on {move.date} "
                f"(z={move.ret_z:+.1f}); no headline in the ±1 day window "
                f"matched the company, its peers or macro terms."
            )
            return ExplanationResult(
                summary=summary,
                primary_category="unexplained",
                confidence=0.2,
                cited_article_ids=(),
                unexplained=True,
            )

        opener = (
            f"{move.ticker} {'fell' if move.ret < 0 else 'rose'} "
            f"{abs(move.ret):.1%} on {move.date}, a {abs(move.ret_z):.1f}-sigma "
            f"day for the stock."
        )
        sentences: list[str] = [opener]

        components = dict(
            zip(
                _COMPONENT_LABELS,
                (move.mkt_component, move.sector_component, move.idio_component),
                strict=True,
            )
        )
        present = {k: v for k, v in components.items() if v is not None}
        if present:
            dominant = max(present, key=lambda k: abs(present[k]))
            total = sum(abs(v) for v in present.values())
            share = abs(present[dominant]) / total if total > 0 else 0.0
            detail = ", ".join(f"{k} {v * 100:+.1f}pp" for k, v in present.items())
            sentences.append(f"The {dominant} component dominated ({detail}).")
        else:
            share = 0.0
            sentences.append("No factor decomposition was available for the day.")

        if move.near_earnings:
            sentences.append("The day was inside an earnings window.")
        if move.near_fomc:
            sentences.append("The day was within a day of an FOMC decision.")
        if move.near_cpi:
            sentences.append("The day was within a day of a CPI release.")
        if move.peer_comove is not None:
            sentences.append(f"Peers moved {move.peer_comove:+.1%} on average the same day.")

        cited = relevant[:3]
        if cited:
            titles = "; ".join(
                article.title + (f" ({article.source})" if article.source else "")
                for article, _ in cited
            )
            sentences.append(f"Related headlines: {titles}")

        confidence = min(
            0.9,
            0.3
            + 0.15 * min(len(relevant), 3)
            + 0.2 * float(move.near_earnings)
            + 0.1 * float(share >= 0.6),
        )
        # Matching headlines that name no event are coverage, not evidence:
        # they should not buy the same confidence as "cuts guidance". Only
        # checked when there *are* matched headlines to judge.
        if relevant and not any(_names_an_event(article.title) for article, _ in relevant):
            confidence = max(WEAK_ATTRIBUTION_FLOOR, confidence - WEAK_ATTRIBUTION_PENALTY)
            sentences.append(WEAK_ATTRIBUTION_NOTE)

        return ExplanationResult(
            summary=" ".join(sentences),
            primary_category=move.routing,
            confidence=round(confidence, 3),
            cited_article_ids=tuple(article.id for article, _ in cited),
            unexplained=False,
        )

    # ------------------------------------------------------------------ #
    # Ontology
    # ------------------------------------------------------------------ #

    def suggest_peers(
        self, ticker: str, name: str, sector: str | None, industry: str | None
    ) -> list[str]:
        """No keyless way to name peers from a company alone; the caller falls
        back to the sector ETF's top holdings."""
        return []

    # ------------------------------------------------------------------ #
    # Chat
    # ------------------------------------------------------------------ #

    def chat(
        self,
        history: Sequence[ChatTurn],
        tools: Mapping[str, ToolFn],
        ticker: str | None,
    ) -> ChatReply:
        """One rule-based tool call and a plain-text rendering of its result.

        Same endpoint, same response shape as the model path: keyword routing
        replaces the tool-calling loop.
        """
        text = _last_user_text(history)
        lowered = text.lower()
        resolved = ticker or _guess_ticker(text)
        if not resolved:
            return ChatReply("Tell me a ticker, e.g. 'why did NVDA drop on 2025-08-28?'", [])

        date_match = _DATE_RE.search(text)
        day = date_match.group(0) if date_match else None

        name: str
        payload: dict[str, Any]
        if "news" in lowered or "headline" in lowered:
            name = "search_news"
            payload = {"ticker": resolved, "limit": 10}
        elif day is not None:
            name = "get_move"
            payload = {"ticker": resolved, "date": day}
        else:
            name = "list_moves"
            payload = {
                "ticker": resolved,
                "direction": _guess_direction(lowered),
                "limit": 5,
            }

        tool = tools.get(name)
        if tool is None:
            return ChatReply(f"The {name} tool is not available.", [])

        try:
            output = tool(**payload)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            return ChatReply(f"{name} failed: {exc}", [ToolCallRecord(name, payload, None)])

        return ChatReply(
            self._render(name, resolved, payload, output),
            [ToolCallRecord(name, payload, output)],
        )

    def _render(self, name: str, ticker: str, payload: dict[str, Any], output: Any) -> str:
        if name == "get_move":
            if not isinstance(output, Mapping) or not output:
                return f"No move stored for {ticker} on {payload.get('date')}."
            error = output.get("error")
            if error:
                return str(error)
            explanation = output.get("explanation")
            summary = (
                explanation.get("summary") if isinstance(explanation, Mapping) else None
            ) or "No explanation yet"
            lines = [f"{ticker} on {output.get('date', payload.get('date'))}", summary]
            lines += [
                f"- {_article_line(a)}" for a in _as_list(output.get("articles"), "articles")[:3]
            ]
            return "\n".join(lines)

        if name == "list_moves":
            moves = _as_list(output, "moves")
            if not moves:
                return f"No stored moves for {ticker}."
            header = f"Largest moves for {ticker}:"
            return "\n".join([header, *(_move_line(m) for m in moves)])

        articles = _as_list(output, "articles")
        if not articles:
            return f"No stored headlines for {ticker}."
        return "\n".join([f"Headlines for {ticker}:", *(f"- {_article_line(a)}" for a in articles)])
