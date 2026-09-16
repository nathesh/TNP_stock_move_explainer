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
from datetime import date
from typing import Any

from stock_moves import narrate
from stock_moves.providers.base import (
    MACRO_TERMS,
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

#: Words that ask for an extreme rather than a survey. They decide the
#: *ranking*: a reader who says "biggest" means the largest percentage move,
#: not the most statistically unusual one, and the two are routinely different
#: days.
_SUPERLATIVE_WORDS: tuple[str, ...] = (
    "most",
    "biggest",
    "largest",
    "greatest",
    "worst",
    "best",
    "sharpest",
    "steepest",
    "deepest",
    "single",
    "record",
)

#: Plural nouns for a move. "The biggest falls" wants a list; "the biggest
#: fall" wants one day, and answering it with five is the dump this fixes.
_PLURAL_MOVE_WORDS: tuple[str, ...] = (
    "moves",
    "days",
    "falls",
    "drops",
    "gains",
    "rallies",
    "declines",
    "times",
    "losses",
    "swings",
)

#: Spelled-out counts that mean "more than one". "one" is deliberately absent:
#: it asks for a single move anyway, and it hides inside "one-day".
_COUNT_WORDS: tuple[str, ...] = (
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "couple",
    "few",
    "several",
)

#: Words that ask about the company's *edges* rather than about its days
#: (v1.5 decision 10). Matched whole-word on the normalised text, so "country"
#: does not fire on "countryside" and "rival" does not fire on "rivalry", and
#: both the singular and the plural of each noun are listed because whole-word
#: matching gives no stemming. "depend on" is two tokens with one space, which
#: is what `_normalise` leaves behind.
_RELATION_WORDS: tuple[str, ...] = (
    "compete",
    "competitor",
    "competitors",
    "rival",
    "rivals",
    "supplier",
    "suppliers",
    "customer",
    "customers",
    "exposure",
    "exposed",
    "countries",
    "country",
    "relation",
    "relations",
    "related",
    "depend on",
    "peers",
)

#: A digit count ("top 3"). One or two digits only, so a bare year does not
#: read as a row count.
_COUNT_DIGITS_RE = re.compile(r"(?<!\d)\d{1,2}(?!\d)")

#: `{"error": "no move for TICKER on YYYY-MM-DD"}` from `tool_get_move`.
_NO_MOVE_RE = re.compile(r"^no move for (\S+) on (\d{4}-\d{2}-\d{2})$")

#: Rows returned for a superlative question, and for a survey question.
_SINGLE_LIMIT = 1
_LIST_LIMIT = 5

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
"""Appended to the summary when no cited headline names an event."""

WEAK_ATTRIBUTION_PENALTY = 0.25
WEAK_ATTRIBUTION_FLOOR = 0.2

#: The ceiling on confidence when no cited headline names an event. Matching
#: the company is not evidence of a cause: "Can TEST stock reach $350?" is
#: commentary, and an explanation built on nothing else may not read as more
#: than a guess. Sits *under* the penalty above, not instead of it.
NO_EVENT_CONFIDENCE_CAP = 0.45

#: The ceiling when at least one cited headline does name an event. Lower than
#: the old 0.9: a keyword match on a title, with no body text read, is never
#: near-certain evidence.
EVENT_CONFIDENCE_CAP = 0.85

#: Added per cited headline that names an event. Headlines that match only the
#: entity add nothing here -- their contribution to the move is already the
#: relevance stored on `move_articles`, and counting them again as confidence
#: is what let generic bullish commentary reach 0.85.
EVENT_HEADLINE_CREDIT = 0.15


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


def _event_headline_sentence(article: ArticleInput) -> str:
    """One sentence naming the cited headline that carried the event.

    The confidence now rests on this headline and no other, so the reader is
    told which one it is rather than being handed a number to trust.
    """
    source = f" ({article.source})" if article.source else ""
    return f"The event comes from \u201c{article.title}\u201d{source}."


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


def _wants_superlative(lowered: str) -> bool:
    """True when the question asks for an extreme ("the biggest fall").

    Whole-word matching, so "mostly" is not "most" and "singled" is not
    "single".
    """
    norm = _normalise(lowered)
    return any(_has_word(norm, word) for word in _SUPERLATIVE_WORDS)


def _names_several(lowered: str) -> bool:
    """True when the question asks for more than one move.

    Either a plural noun ("the biggest falls") or a count ("top 3", "the three
    worst days"). Without one of those, a superlative question is about a
    single day and should be answered with a single day.
    """
    norm = _normalise(lowered)
    if any(_has_word(norm, word) for word in _PLURAL_MOVE_WORDS):
        return True
    if any(_has_word(norm, word) for word in _COUNT_WORDS):
        return True
    return _COUNT_DIGITS_RE.search(norm) is not None


def _asks_about_relations(text: str) -> bool:
    """True when the question is about who the company is connected to.

    Whole-word, against the normalised text, for the same reason
    `_wants_superlative` is: under a substring rule "the arrival of the new
    chip" contains "rival" and "unrelated to earnings" contains "related",
    and both would be answered with a table of edges.
    """
    norm = _normalise(text)
    return any(_has_word(norm, word) for word in _RELATION_WORDS)


def _no_move_sentence(error: str) -> str:
    """`no move for TEST on 2026-03-02` said as a sentence.

    The raw error is a developer string: lower-cased, undated in English and
    starting mid-thought. A reader who asked about a quiet day should be told
    the day was quiet, in the same voice as every other answer.
    """
    match = _NO_MOVE_RE.match(error.strip())
    if match is None:
        return error
    ticker, day = match.groups()
    try:
        parsed = date.fromisoformat(day)
    except ValueError:
        return error
    return f"{ticker} did not have a major move on {narrate.plain_day(parsed)}."


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
        """Say the numbers in English, via `narrate`. Returns `unexplained`
        when nothing in the window matched and the move is not extreme
        enough to assert a cause on the decomposition alone.

        The phrasing is `narrate`'s so that a keyless install, a degraded
        model call and a keyed explanation all describe a move the same way.
        """
        relevant = [(a, s) for a, s in scored if s.relevance >= 0.5]

        if not relevant and abs(move.ret_z) < 3:
            summary = (
                f"{move.ticker} was {narrate.plain_move(move.ret)} on "
                f"{narrate.plain_day(move.date)}, but no headline in the day "
                f"either side mentioned the company, its competitors or the "
                f"wider economy, so there is nothing here to attribute it to."
            )
            return ExplanationResult(
                summary=summary,
                primary_category="unexplained",
                confidence=0.2,
                cited_article_ids=(),
                unexplained=True,
            )

        facts = narrate.MoveFacts.from_context(move)
        cited = relevant[:3]
        # One headline, named as coverage rather than as a cause. The full set
        # is already on the move's `articles`; three of them pasted into a
        # sentence is what made the stored summaries unreadable.
        headlines = []
        if cited:
            article = cited[0][0]
            source = f" ({article.source})" if article.source else ""
            headlines.append(f"the day's coverage led with \u201c{article.title}\u201d{source}")
        sentences = narrate.summary_sentences(facts, headlines)

        components = (move.mkt_component, move.sector_component, move.idio_component)
        present = [value for value in components if value is not None]
        total = sum(abs(value) for value in present)
        share = (max(abs(value) for value in present) / total) if present and total > 0 else 0.0

        # Only headlines that name an event earn confidence. A headline that
        # merely matches the company is coverage, not evidence: its worth is
        # the relevance already stored against the move, and counting it here
        # as well is what let a day of generic bullish commentary score 0.85.
        event_articles = [article for article, _ in cited if _names_an_event(article.title)]
        confidence = (
            0.3
            + EVENT_HEADLINE_CREDIT * len(event_articles)
            + 0.2 * float(move.near_earnings)
            + 0.1 * float(share >= 0.6)
        )
        if event_articles:
            confidence = min(EVENT_CONFIDENCE_CAP, confidence)
            sentences.append(_event_headline_sentence(event_articles[0]))
        else:
            # No cited headline names an event, so the penalty applies and the
            # result is held under the cap however well the decomposition and
            # the earnings window score on their own.
            if relevant:
                confidence = max(WEAK_ATTRIBUTION_FLOOR, confidence - WEAK_ATTRIBUTION_PENALTY)
                sentences.append(WEAK_ATTRIBUTION_NOTE)
            confidence = min(NO_EVENT_CONFIDENCE_CAP, confidence)

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

    def suggest_relations(
        self, ticker: str, name: str, sector: str | None, industry: str | None
    ) -> Relations:
        """Nothing, in all four relations (v1.5 decision 3).

        The keyless ETF fallback for `competitor` edges is applied by the
        ontology layer, from the sector ETF's top holdings, exactly as it
        already is for peers -- not here, because this provider is handed a
        company's identity and no prices. Suppliers, customers and countries
        have no keyless source at all: no free feed states them, and inventing
        them from a sector name would put guesses in a table the gate rule
        then treats as fact.

        So without a key there are no `country` edges, and the geopolitical
        gate can never open. That is the honest behaviour, and the write-up
        says so rather than hiding it behind a plausible default.
        """
        return Relations(competitors=(), suppliers=(), customers=(), countries=())

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

        The branch order is the rule, and it is deliberate:

        1. `news`/`headline` first, so "news about competitors" searches the
           headlines. A question that names the word "news" is asking for
           coverage even when it also names a relation, and the relations tool
           has no headlines to give it.
        2. relations next, *before* the date, so "which countries was TEST
           exposed to in 2025?" answers with the edges rather than with one
           day's move. Edges are facts about the company, not about a day, so
           a date in the question narrows nothing.
        3. a date, then the survey list as the fallback.
        """
        text = _last_user_text(history)
        lowered = text.lower()
        resolved = ticker or _guess_ticker(text)
        if not resolved:
            # The example date has to be inside the data window, or the first
            # thing a new user copies is a question with no answer. It matches
            # the chat box's own placeholder.
            return ChatReply("Tell me a ticker, e.g. 'why did NVDA drop on 2026-01-20?'", [])

        date_match = _DATE_RE.search(text)
        day = date_match.group(0) if date_match else None

        name: str
        payload: dict[str, Any]
        if "news" in lowered or "headline" in lowered:
            name = "search_news"
            payload = {"ticker": resolved, "limit": 10}
        elif _asks_about_relations(text):
            name = "get_relations"
            payload = {"ticker": resolved}
        elif day is not None:
            name = "get_move"
            payload = {"ticker": resolved, "date": day}
        else:
            name = "list_moves"
            # "Biggest" is a question about size, so it has to change the
            # ranking as well as the row count: the z-ordered list's lead row
            # is routinely not the largest move in it.
            superlative = _wants_superlative(lowered)
            payload = {
                "ticker": resolved,
                "direction": _guess_direction(lowered),
                "order": "pct" if superlative else "z",
                "limit": (
                    _SINGLE_LIMIT if superlative and not _names_several(lowered) else _LIST_LIMIT
                ),
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
        """The tool's result as prose.

        This is the keyless answer to "synthesise before sending it out": the
        rows are grouped, the numbers are said in English and the set gets a
        sentence of its own. `narrate` does the writing so that the keyed
        providers, which are handed the same sentences in their prompt, cannot
        drift from it.
        """
        company = _company_of(output)
        if name == "get_move":
            if not isinstance(output, Mapping) or not output:
                return f"No move stored for {ticker} on {payload.get('date')}."
            error = output.get("error")
            if error:
                return _no_move_sentence(str(error))
            return narrate.narrate_one_move(
                ticker, output, company, model_summary=_stored_summary(output)
            )

        if name == "list_moves":
            moves = _as_list(output, "moves")
            if not moves:
                return f"No stored moves for {ticker}."
            return narrate.narrate_moves(
                ticker,
                moves,
                company,
                direction=payload.get("direction"),
                order=str(payload.get("order") or "z"),
            )

        if name == "get_relations":
            return narrate.narrate_relations(ticker, company, _as_list(output, "edges"))

        return narrate.narrate_news(ticker, _as_list(output, "articles"))


def _company_of(output: Any) -> str:
    """The company name the chat tools attach to every move, if present."""
    rows = output if isinstance(output, list) else [output]
    for row in rows:
        if isinstance(row, Mapping) and row.get("company"):
            return str(row["company"])
    return ""


def _stored_summary(move: Mapping[str, Any]) -> str | None:
    """The stored summary, when replaying it beats regenerating it.

    Two cases qualify. A model wrote it, so it says more than the numbers can.
    Or it is marked `unexplained`, where the finding *is* that no headline
    matched -- a fact the decomposition alone cannot state, and one a reader
    would otherwise have to infer from an absent list of headlines.

    Otherwise the heuristic's own summaries are regenerated from the move's
    numbers rather than replayed, so rows written by an older version of this
    file still read the way the current one speaks.
    """
    explanation = move.get("explanation")
    if not isinstance(explanation, Mapping):
        return None
    summary = explanation.get("summary")
    if not summary:
        return None
    if explanation.get("unexplained"):
        return str(summary)
    provider = str(explanation.get("provider") or "")
    return None if provider in {"", "heuristic"} else str(summary)
