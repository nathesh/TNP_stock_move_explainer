"""The explanation layer (DESIGN section 4).

One move, one model call, one cached row. The division of labour is:

* `top_scored` re-reads the *already scored* `move_articles` links and hands the
  provider only the top-K headlines. Cheap headline scoring has happened
  before this module runs, so the prompt never sees the long tail — that is the
  cost control in DESIGN section 4 ("model usage is about one call per move,
  roughly ten per ticker, not one per headline").
* `get_or_create_explanation` is the cache: a stored row for the move is
  returned untouched unless `refresh=True`, in which case the row is updated in
  place so a move never accumulates a second explanation (the table has a
  unique index on `move_id`, and this keeps that promise instead of relying on
  the database to refuse).
* Whatever the provider returns is **validated before it is stored**. A model
  can invent a category, a confidence outside [0, 1] or a citation to an
  article it was never shown; the row that reaches the database cannot contain
  any of the three. That is why the free-text provider output is a dataclass
  and the stored row is written only through this function.

The caller supplies the `Session`, and nothing here touches the network or the
settings, so the whole module is testable against an in-memory database.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

from sqlmodel import Session, col, select

from stock_moves.models import (
    Article,
    Company,
    Explanation,
    Move,
    MoveArticle,
    utcnow,
)
from stock_moves.providers.base import (
    ArticleInput,
    ArticleScore,
    ExplanationResult,
    ModelProvider,
    MoveContext,
)
from stock_moves.providers.heuristic import HeuristicProvider

#: Stored in `explanations.provider` when a keyed provider degraded to the
#: keyless one, so a reader can tell a templated row from a model-written one.
FALLBACK_PROVIDER_NAME: str = HeuristicProvider.name

__all__ = [
    "CATEGORIES",
    "DEFAULT_TOP_K",
    "UNEXPLAINED",
    "build_context",
    "explanation_to_result",
    "get_or_create_explanation",
    "top_scored",
]

UNEXPLAINED = "unexplained"

CATEGORIES: frozenset[str] = frozenset({"company", "industry", "macro", UNEXPLAINED})
"""The only values `explanations.primary_category` may hold; anything else a
provider returns is stored as `unexplained` rather than trusted."""

DEFAULT_TOP_K = 8
"""Mirrors `Settings.top_k_articles`; passed explicitly by the API layer so
this module needs no settings import."""


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def top_scored(
    session: Session, move_id: int, top_k: int = DEFAULT_TOP_K
) -> list[tuple[ArticleInput, ArticleScore]]:
    """The move's best-scored headlines, most relevant first, cut to `top_k`.

    Reads the scores that were written when the articles were linked — this
    function never re-scores. The tie-break on `articles.id` makes the cut
    deterministic when several links share a relevance, so two runs on the same
    database prompt the model with the same headlines.
    """
    if top_k <= 0:
        return []
    statement = (
        select(Article, MoveArticle)
        .join(MoveArticle, col(MoveArticle.article_id) == col(Article.id))
        .where(col(MoveArticle.move_id) == move_id)
        .order_by(col(MoveArticle.relevance).desc(), col(Article.id).asc())
        .limit(top_k)
    )
    return [
        (
            ArticleInput.from_object(article),
            ArticleScore(
                article_id=int(article.id) if article.id is not None else 0,
                relevance=float(link.relevance),
                category=str(link.category),
            ),
        )
        for article, link in session.exec(statement).all()
    ]


def build_context(move: Move, company: Company) -> MoveContext:
    """The quantitative half of the prompt: decomposition, regime, event flags, peers."""
    return MoveContext.from_objects(move, company)


# --------------------------------------------------------------------------- #
# Validation of provider output
# --------------------------------------------------------------------------- #


def _category(value: str) -> str:
    """A known category, or `unexplained` for anything else."""
    return value if value in CATEGORIES else UNEXPLAINED


def _confidence(value: float) -> float:
    """`value` clamped into [0, 1]; a non-numeric or NaN confidence becomes 0.0."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(confidence):
        return 0.0
    return min(1.0, max(0.0, confidence))


def _citations(cited: Sequence[int], shown: set[int]) -> list[int]:
    """Citations restricted to the articles the provider was actually shown.

    Order is preserved and duplicates are dropped, so the stored list reads as
    the provider's own ranking of its evidence.
    """
    kept: list[int] = []
    for raw in cited:
        try:
            article_id = int(raw)
        except (TypeError, ValueError):
            continue
        if article_id in shown and article_id not in kept:
            kept.append(article_id)
    return kept


# --------------------------------------------------------------------------- #
# The cache
# --------------------------------------------------------------------------- #


def get_or_create_explanation(
    session: Session,
    move: Move,
    company: Company,
    provider: ModelProvider,
    *,
    top_k: int = DEFAULT_TOP_K,
    refresh: bool = False,
) -> Explanation:
    """Return the move's explanation, computing and caching it if needed.

    A stored row short-circuits the provider entirely unless `refresh` is set —
    that is what makes `GET /tickers/{ticker}/moves/{date}` free on the second
    read. With `refresh`, the provider runs again and the existing row is
    updated in place, so there is still exactly one explanation per move.
    """
    if move.id is None:
        raise ValueError("move must be persisted (move.id is None) before explaining")

    existing = session.exec(select(Explanation).where(col(Explanation.move_id) == move.id)).first()
    if existing is not None and not refresh:
        return existing

    scored = top_scored(session, move.id, top_k=top_k)
    result = provider.explain(build_context(move, company), scored)

    category = _category(result.primary_category)
    values: dict[str, Any] = {
        "summary": str(result.summary).strip(),
        "primary_category": category,
        "confidence": _confidence(result.confidence),
        "cited_article_ids_json": json.dumps(
            _citations(result.cited_article_ids, {score.article_id for _, score in scored})
        ),
        # Keep the flag and the category consistent: a category we had to
        # replace is not an explanation, whatever the provider claimed.
        "unexplained": bool(result.unexplained) or category == UNEXPLAINED,
        "provider": FALLBACK_PROVIDER_NAME if result.degraded else provider.name,
        "created_at": utcnow(),
    }

    if existing is None:
        row = Explanation(move_id=move.id, **values)
    else:
        row = existing
        for name, value in values.items():
            setattr(row, name, value)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def explanation_to_result(e: Explanation) -> ExplanationResult:
    """The stored row back as the provider-layer dataclass (what the API serialises)."""
    return ExplanationResult(
        summary=e.summary,
        primary_category=e.primary_category,
        confidence=e.confidence,
        cited_article_ids=tuple(e.cited_article_ids),
        unexplained=e.unexplained,
    )
