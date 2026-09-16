"""What the keyed providers say to a model, and the shapes they expect back.

Split out of `anthropic.py` when a second keyed provider arrived. The point is
that a provider should be *transport*: an OpenAI and an Anthropic explanation
differ because the models differ, not because someone edited one prompt and
forgot the other. Everything that decides the content of a call -- the system
prompts, the structured-output schemas, the rendering of a move into a prompt
block -- lives here and is imported by both.

Dependency-free for the same reason as `base.py` and `narrate.py`: no database,
no settings.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel

from stock_moves.narrate import MoveFacts, summary_sentences
from stock_moves.providers.base import ArticleInput, ArticleScore, MoveContext

__all__ = [
    "CHAT_SYSTEM",
    "EXPLAIN_SYSTEM",
    "MAX_COUNTRIES",
    "MAX_PEERS",
    "MAX_SUPPLY_CHAIN",
    "PEERS_SYSTEM",
    "RELATIONS_SYSTEM",
    "SCORE_SYSTEM",
    "Explanation",
    "Peers",
    "RelationsOut",
    "Scores",
    "article_line",
    "chat_system",
    "clamp",
    "clean_tickers",
    "explain_prompt",
    "move_block",
    "narrative_block",
    "normalise_countries",
    "openai_tools",
    "peers_prompt",
    "relations_prompt",
    "score_prompt",
    "scored_line",
]

MAX_PEERS = 6
"""Competitors, and peers before them: the list the prompt asks for and the
cut the caller applies, stated once so the two cannot drift."""

MAX_SUPPLY_CHAIN = 4
"""Suppliers, and customers: fewer than competitors, because a company has
many rivals and only a handful of *public* counterparties worth naming."""

MAX_COUNTRIES = 5
"""Country edges. Weights across them sum to at most 1 (v1.5 decision 3)."""

_CATEGORY = Literal["company", "industry", "macro"]
_PRIMARY_CATEGORY = Literal["company", "industry", "macro", "unexplained"]


# --------------------------------------------------------------------------- #
# Structured output schemas
# --------------------------------------------------------------------------- #


class _Score(BaseModel):
    article_id: int
    relevance: float
    category: _CATEGORY


class Scores(BaseModel):
    scores: list[_Score]


class Explanation(BaseModel):
    summary: str
    primary_category: _PRIMARY_CATEGORY
    confidence: float
    cited_article_ids: list[int]
    unexplained: bool


class Peers(BaseModel):
    tickers: list[str]


class _CountryWeight(BaseModel):
    country: str
    weight: float


class RelationsOut(BaseModel):
    """The `suggest_relations` structured output, before any cleaning.

    Named `RelationsOut` rather than `Relations` because the cleaned, frozen
    dataclass in `base.py` owns that name: this is what the model said, that is
    what the app stores. Everything here is a plain list so a model that
    returns nothing for a field returns an empty list rather than failing
    validation.
    """

    competitors: list[str] = []
    suppliers: list[str] = []
    customers: list[str] = []
    countries: list[_CountryWeight] = []


# --------------------------------------------------------------------------- #
# System prompts
# --------------------------------------------------------------------------- #

SCORE_SYSTEM = (
    "You are scoring news headlines for whether they explain a given stock's "
    "move on a given day. For each article return a relevance in [0, 1]: 1 "
    "means the headline is directly about this company on this day, 0 means it "
    "is unrelated to the move. The category is what the headline itself is "
    "about: 'company' for this specific company, 'industry' for its sector or "
    "peers, 'macro' for the economy, rates, policy or the whole market. "
    "Headlines are all you get -- there is no article body. Return one score "
    "per given article id and no ids that were not given."
)

#: The readability contract, shared by `explain` and `chat` so a move reads the
#: same whether it was written into the database or spoken in the chat UI.
#: Every banned term here is one the templated output used to emit verbatim.
PLAIN_ENGLISH = (
    "Write for an intelligent reader who does not work in markets. Never use "
    "the words 'sigma', 'z-score', 'idiosyncratic', 'beta', 'basis points' or "
    "the abbreviation 'pp', and never print a raw z-score: say how large the "
    "move was as a multiple of an ordinary day for that stock ('about four "
    "times the size of a typical day'). Give the market/sector/company split "
    "as points of the move itself ('8.6 of the 14.5 points came from the "
    "company'), not as separate percentages. Name events in ordinary words: "
    "an earnings report, a Federal Reserve rate decision, an inflation "
    "release. Prefer short sentences. Do not list headlines verbatim in a "
    "run-on sentence -- refer to what they say."
)

EXPLAIN_SYSTEM = (
    "You explain why a stock moved on one day. This is attribution, not "
    "causation: say what the evidence supports, never that one event caused "
    "the move. Cite only the article ids you are given, by id. Set "
    "`unexplained` true and `primary_category` 'unexplained' when the evidence "
    "is weak -- that is a correct answer, not a failure. Write 2-4 sentences.\n\n"
    "The first sentence must name the company, the date and the size of the "
    "move, because this summary is stored and read on its own, with no "
    "surrounding table to supply them: 'Tesla was down 14.5% on Thursday, 23 "
    "July 2026.' Never write 'fell sharply' in place of the number.\n\n"
    "Mention proximity to earnings, to an FOMC decision or to a CPI release, "
    "the market and sector regime, and how peers moved, when they are "
    "informative; skip them when they are not. Confidence is in [0, 1] and "
    "should reflect how much the headlines and the decomposition agree.\n\n"
    "When `sub_routing` is 'share_shift', say so and name the competitor from "
    "`competitors` that moved the other way, with the direction `rival_comove` "
    "gives: the tape is pointing at a specific rival, which is the most useful "
    "thing the numbers say that day. When `sub_routing` is 'country:XX', call "
    "it geopolitical and say 'via <country> exposure' -- the country the code "
    "names -- because the company has a stored exposure to that country and "
    "that country's ETF moved; cite a geopolitical event only if one of the "
    "articles you were given supports it, since `geo_events` is a count of "
    "headlines rather than an article you may cite.\n\n" + PLAIN_ENGLISH
)

PEERS_SYSTEM = (
    "You name public-company peers for a given company. Return up to 6 "
    "US-listed public companies, by ticker symbol only, in upper case. No "
    "ETFs, no indices, no private companies, and never the company's own "
    "ticker. Closest competitors first; return fewer, or none, rather than "
    "padding the list."
)

RELATIONS_SYSTEM = (
    "You name a company's business relationships, for a system that uses them "
    "to decide which other tickers and which countries to look at when that "
    "company's stock moves.\n\n"
    f"Return up to {MAX_PEERS} direct competitors as US-listed ticker symbols, "
    "closest first. A competitor sells a substitute to the same buyers; a "
    "company merely in the same sector is not one.\n\n"
    f"Return up to {MAX_SUPPLY_CHAIN} suppliers and up to {MAX_SUPPLY_CHAIN} "
    "customers, also as ticker symbols. A supplier sells this company an input "
    "it depends on; a customer buys enough from it to matter to its revenue. "
    "Public companies only: if the important counterparty is private, "
    "state-owned or a subsidiary with no listing of its own, omit it rather "
    "than naming the parent or writing the name out in words.\n\n"
    f"Return up to {MAX_COUNTRIES} countries as ISO-3166 alpha-2 codes ('TW', "
    "'CN', 'DE'), each with a weight in [0, 1] for the share of revenue or of "
    "critical supply that depends on that country. The weights must sum to at "
    "most 1; they are shares of the whole company, not shares of each other, "
    "so a domestic company's weights are small and need not add up to 1. "
    "Include a country only where a disruption there would move this stock.\n\n"
    "All tickers in upper case, no ETFs, no indices, and never the company's "
    "own ticker in any of the three lists. Return fewer, or an empty list, "
    "rather than padding: an empty list is a correct answer and a guessed one "
    "is not, because every entry here becomes a stored edge that later credits "
    "or discredits a headline."
)

CHAT_SYSTEM = (
    "You explain stock moves using only the tools' results. Cite the date of "
    "every move you discuss. If the tools return nothing, say so.\n\n"
    "You do not know today's date and must never guess or compute one. If the "
    "question names no exact YYYY-MM-DD date, call list_moves -- the server "
    "applies any time window the question implies. Only call get_move with a "
    "date the user wrote out.\n\n"
    "Answer the question that was asked before adding detail: if someone asks "
    "which day was worst, the first sentence names that day and what happened, "
    "and the supporting numbers come after. When you describe several moves, "
    "say what they have in common before listing them one by one -- a reader "
    "who wanted a list of rows would have read the table.\n\n"
    "The tool output includes a `narrative` field for each move: plain-English "
    "sentences already generated from that move's numbers. Prefer its phrasing "
    "and its framing; you are synthesising across moves and adding what the "
    "headlines say, not re-deriving arithmetic.\n\n"
    "Format for a plain-text chat window, not for markdown: no asterisks, no "
    "pound signs, no tables. Use blank lines between paragraphs and two-space "
    "indentation for any list.\n\n" + PLAIN_ENGLISH
)


def chat_system(ticker: str | None) -> str:
    """`CHAT_SYSTEM` plus this session's default ticker."""
    return CHAT_SYSTEM + (
        f"\n\nDefault ticker: {ticker}."
        if ticker
        else "\n\nThere is no default ticker; ask for one if a tool needs it."
    )


# --------------------------------------------------------------------------- #
# Rendering a move into a prompt
# --------------------------------------------------------------------------- #


def _round(value: float | None, digits: int = 5) -> float | None:
    return None if value is None else round(value, digits)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def move_block(move: MoveContext) -> str:
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
        # v1.5: the relationship layer. Same rule as the rest of the block --
        # an edge the app does not have is absent from the JSON rather than
        # present and empty, so a keyless install (which stores no model edges
        # at all) hands the model a v1-shaped block instead of a row of nulls
        # inviting it to explain their absence.
        "sub_routing": move.sub_routing,
        "macro_driver": move.macro_driver,
        "macro_driver_component": _round(move.macro_driver_component),
        "rival_comove": _round(move.rival_comove),
        "chain_comove": _round(move.chain_comove),
        "competitors": list(move.competitors) or None,
        "suppliers": list(move.suppliers) or None,
        "customers": list(move.customers) or None,
        "countries": [
            {"country": code, "weight": _round(weight, 3)} for code, weight in move.countries
        ]
        or None,
        "geo_events": list(move.geo_events) or None,
    }
    return json.dumps(
        {key: value for key, value in payload.items() if value is not None},
        default=str,
    )


def narrative_block(move: MoveContext) -> str:
    """The same numbers already said in English by `narrate`.

    Handing the model both the JSON and the narration is cheap and removes the
    step most likely to go wrong: the model is not asked to turn a factor
    loading into a sentence, only to weigh it against the headlines.
    """
    return " ".join(summary_sentences(MoveFacts.from_context(move)))


def _article_date(article: ArticleInput) -> str:
    published = article.published_at
    return published.date().isoformat() if published is not None else "unknown"


def article_line(article: ArticleInput) -> str:
    """`id | date | source | title` -- the four fields v1's news sources give."""
    return (
        f"{article.id} | {_article_date(article)} | {article.source or 'unknown'} | {article.title}"
    )


def scored_line(article: ArticleInput, score: ArticleScore) -> str:
    return f"{article_line(article)} | relevance={score.relevance:.2f} | category={score.category}"


def score_prompt(move: MoveContext, articles: Sequence[ArticleInput]) -> str:
    lines = "\n".join(article_line(article) for article in articles)
    return (
        f"Move:\n{move_block(move)}\n\n"
        f"Articles (id | date | source | title):\n{lines}\n\n"
        f"Score these {len(articles)} article ids."
    )


def explain_prompt(move: MoveContext, scored: Sequence[tuple[ArticleInput, ArticleScore]]) -> str:
    if scored:
        lines = "\n".join(scored_line(article, score) for article, score in scored)
    else:
        lines = "(no headlines in the window)"
    return (
        f"Move:\n{move_block(move)}\n\n"
        f"The same numbers in plain English:\n{narrative_block(move)}\n\n"
        f"Candidate articles (id | date | source | title | relevance | "
        f"category), best first:\n{lines}\n\n"
        "Explain this move."
    )


def peers_prompt(ticker: str, name: str, sector: str | None, industry: str | None) -> str:
    return json.dumps({"ticker": ticker, "company": name, "sector": sector, "industry": industry})


def relations_prompt(ticker: str, name: str, sector: str | None, industry: str | None) -> str:
    """The same identity block `peers_prompt` sends; `RELATIONS_SYSTEM` asks
    for four lists instead of one."""
    return json.dumps({"ticker": ticker, "company": name, "sector": sector, "industry": industry})


# --------------------------------------------------------------------------- #
# Cleaning a model's relationship answer
# --------------------------------------------------------------------------- #


def clean_tickers(raw: Sequence[Any], self_ticker: str, limit: int) -> tuple[str, ...]:
    """Upper-case, strip, drop blanks and the company itself, dedupe, cut.

    Shared by both keyed providers so "AMD asked about AMD" and "nvda twice"
    cannot be handled one way by one vendor and another way by the other.
    """
    own = self_ticker.strip().upper()
    cleaned = (str(item).strip().upper() for item in raw)
    kept = [candidate for candidate in cleaned if candidate and candidate != own]
    return tuple(dict.fromkeys(kept))[:limit]


def normalise_countries(
    raw: Sequence[Any], limit: int = MAX_COUNTRIES
) -> tuple[tuple[str, float], ...]:
    """`(ISO alpha-2, weight)` pairs obeying the contract in `base.Relations`.

    Each weight is clamped into [0, 1] first; if the clamped weights still sum
    above 1 they are scaled down *proportionally*, which keeps their ranking
    and their ratios -- a model that says 60% China and 30% Taiwan means twice
    as much China either way. Truncating the list instead would silently drop
    the exposure that the gate rule exists to catch. Codes are upper-cased and
    deduped on first mention; a non-two-letter code is dropped, since the
    country-ETF lookup downstream is keyed on alpha-2.
    """
    pairs: list[tuple[str, float]] = []
    seen: set[str] = set()
    for item in raw:
        code = str(getattr(item, "country", "") or "").strip().upper()
        if len(code) != 2 or not code.isalpha() or code in seen:
            continue
        try:
            weight = float(getattr(item, "weight", 0.0))
        except (TypeError, ValueError):
            continue
        seen.add(code)
        pairs.append((code, clamp(weight)))
        if len(pairs) == limit:
            break

    total = sum(weight for _, weight in pairs)
    if total > 1.0:
        pairs = [(code, weight / total) for code, weight in pairs]
    return tuple(pairs)


def openai_tools(tool_specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """`TOOL_SPECS` in OpenAI's function-calling shape.

    The specs in `base.py` are written once, in Anthropic's `input_schema`
    form, and translated at the edge. Keeping one canonical list is what makes
    the two providers answer the same questions with the same tools.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["input_schema"],
            },
        }
        for spec in tool_specs
    ]
