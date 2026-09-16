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
    "PEERS_SYSTEM",
    "SCORE_SYSTEM",
    "Explanation",
    "Peers",
    "Scores",
    "article_line",
    "chat_system",
    "clamp",
    "explain_prompt",
    "move_block",
    "narrative_block",
    "openai_tools",
    "peers_prompt",
    "score_prompt",
    "scored_line",
]

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
    "should reflect how much the headlines and the decomposition agree.\n\n" + PLAIN_ENGLISH
)

PEERS_SYSTEM = (
    "You name public-company peers for a given company. Return up to 6 "
    "US-listed public companies, by ticker symbol only, in upper case. No "
    "ETFs, no indices, no private companies, and never the company's own "
    "ticker. Closest competitors first; return fewer, or none, rather than "
    "padding the list."
)

CHAT_SYSTEM = (
    "You explain stock moves using only the tools' results. Cite the date of "
    "every move you discuss. If the tools return nothing, say so.\n\n"
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
