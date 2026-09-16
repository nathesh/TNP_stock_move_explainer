"""Storage and scoring of news (DESIGN sections 3 and 4).

Two jobs, kept in one module because they share the `articles` /
`move_articles` boundary:

* **Storage (section 3).** Articles are deduped on url and stored *once*;
  the link to a move lives in `move_articles`. The same headline can
  therefore be evidence for several moves without being copied.
* **Scoring (section 4).** The two providers are combined, not either/or.
  The free `HeuristicProvider` scores *every* headline and its five
  components are stored so a relevance is inspectable after the fact. Only
  the top-K survivors are shown to a model provider; the stored relevance is
  then the mean of the heuristic and the model score. Articles below the cut
  are still linked — the cut limits model calls, it does not throw evidence
  away.

The five components and their weights come from `docs/architecture-v1.md`
(`move_articles`): bucket_match 0.35, entity_match 0.20, timing 0.15,
source_tier 0.15, coverage 0.15.

v1.5 adds a sixth stored number that is *not* one of the five and carries no
weight: `geo_gate` (plan decision 7). A geopolitical headline is cheap to find
and almost always irrelevant, so one is capped at `GEO_CAP` unless the company
actually has the country edge *and* that country's factor is what moved the
stock that day. The cap is applied after the weighted sum, so the five stored
components still explain the number they produced.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlmodel import Session, col, select

from stock_moves.models import Article, Company, Move, MoveArticle, utcnow
from stock_moves.news import geo
from stock_moves.news.base import NewsItem
from stock_moves.providers.base import (
    ArticleInput,
    ArticleScore,
    ModelProvider,
    MoveContext,
)
from stock_moves.providers.heuristic import HeuristicProvider

__all__ = [
    "GEO_CAP",
    "WEIGHTS",
    "geo_gate",
    "heuristic_components",
    "heuristic_relevance",
    "linked_articles",
    "score_and_link",
    "source_tier",
    "story_key",
    "to_inputs",
    "upsert_articles",
]

WEIGHTS: dict[str, float] = {
    "bucket_match": 0.35,
    "entity_match": 0.20,
    "timing": 0.15,
    "source_tier": 0.15,
    "coverage": 0.15,
}

_CATEGORIES: frozenset[str] = frozenset({"company", "industry", "macro"})

#: A component map: the five weighted numbers, `timing_kind`, and `geo_gate`,
#: which is the one entry that may be `None`.
Components = dict[str, float | str | None]

GEO_CAP: float = 0.30
"""Relevance ceiling for a geopolitical headline whose gate is shut."""

# The two drivers that open the gate for any country the company touches: an
# oil or dollar move is the channel through which a foreign event reaches a
# domestic share price, so it needs no country-specific factor.
_GLOBAL_DRIVERS: frozenset[str] = frozenset({"oil", "dollar"})

_COUNTRY_PREFIX = "country:"

# Static outlet tiers. Matched case-insensitively as a substring, because the
# source string is whatever the feed printed ("Reuters", "reuters.com",
# "Yahoo Finance UK"). Deliberately short: a long list is a maintenance
# liability for 0.15 of one score, and v2 replaces it with a learned prior.
_TIER_1: tuple[str, ...] = (
    "reuters",
    "bloomberg",
    "wsj",
    "wall street journal",
    "financial times",
    "cnbc",
    "associated press",
    "ap news",
    "marketwatch",
    "barron's",
)
_TIER_2: tuple[str, ...] = (
    "yahoo finance",
    "seeking alpha",
    "motley fool",
    "investopedia",
    "business insider",
    "forbes",
    "the street",
    "benzinga",
    "investor's business daily",
)

# Distinct sources on one story, above which more coverage says nothing new.
_COVERAGE_CAP = 10

# Words of the normalised title that identify "the same story".
_STORY_KEY_WORDS = 8

_PUNCT_RE = re.compile(r"[^a-z0-9]+")


def _clamp(value: float) -> float:
    """Squeeze a provider's number into the [0, 1] the column promises."""
    return max(0.0, min(1.0, value))


def _naive_utc(value: datetime | None) -> datetime | None:
    """Convert an aware datetime to UTC and drop the offset.

    SQLite stores no offset, so an aware value read back would compare
    wrongly against the naive timestamps `models.utcnow()` writes. A value
    that is already naive is assumed to be UTC and returned untouched.
    """
    if value is None or value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def story_key(title: str) -> str:
    """The grouping key for "the same story": lower-cased, punctuation
    stripped, first `_STORY_KEY_WORDS` words."""
    words = _PUNCT_RE.sub(" ", title.lower()).split()
    return " ".join(words[:_STORY_KEY_WORDS])


def source_tier(source: str | None) -> float:
    """1.0 tier-1, 0.7 tier-2, 0.5 anything else, 0.4 when unattributed."""
    if source is None:
        return 0.4
    lowered = source.lower()
    if any(outlet in lowered for outlet in _TIER_1):
        return 1.0
    if any(outlet in lowered for outlet in _TIER_2):
        return 0.7
    return 0.5


# --------------------------------------------------------------------------- #
# Storage (DESIGN section 3)
# --------------------------------------------------------------------------- #


def upsert_articles(session: Session, items: Sequence[NewsItem]) -> list[Article]:
    """Store `items` once each, deduped on url, and return the stored rows.

    Duplicates inside `items` keep the first occurrence; urls already in the
    table are reused untouched, so an article linked to an earlier move keeps
    its original `fetched_at`. The return order is the order of the deduped
    input, which lets the caller keep its own per-item state (the routing
    bucket, say) aligned with the rows.
    """
    deduped: dict[str, NewsItem] = {}
    for item in items:
        if item.url and item.url not in deduped:
            deduped[item.url] = item
    if not deduped:
        return []

    urls = list(deduped)
    existing = {
        article.url: article
        for article in session.exec(select(Article).where(col(Article.url).in_(urls))).all()
    }

    fetched_at = utcnow()
    rows: list[Article] = []
    for url, item in deduped.items():
        article = existing.get(url)
        if article is None:
            article = Article(
                url=url,
                title=item.title,
                source=item.source,
                language=item.language,
                published_at=_naive_utc(item.published_at),
                news_source=item.news_source,
                fetched_at=fetched_at,
            )
            session.add(article)
        rows.append(article)

    session.commit()
    return rows


def to_inputs(articles: Sequence[Article]) -> list[ArticleInput]:
    """Stored rows as the provider-layer record, in order."""
    return [ArticleInput.from_object(article) for article in articles]


# --------------------------------------------------------------------------- #
# Scoring (DESIGN section 4)
# --------------------------------------------------------------------------- #


def geo_gate(title: str, move: MoveContext) -> float | None:
    """Whether a geopolitical headline may be credited to `move` (decision 7).

    `None` means the gate does not apply: the title names none of the
    geopolitical vocabulary, so it is scored like any other headline. When it
    does apply the gate opens (1.0) only if *both* halves hold:

    * the title names a country the company has a `country` edge to, and
    * the day's `macro_driver` is that country's factor (`country:XX`) or one
      of the two global channels a foreign event reaches a share price
      through, `oil` and `dollar`.

    Either half alone is a coincidence — a tariff headline on a day no macro
    factor moved, or a macro day whose headline names a country the company
    does not touch — so the gate shuts (0.0) and `heuristic_relevance` caps the
    score at `GEO_CAP`. Both or nothing is the point of the rule: without a
    model key a company has no country edges at all, so the gate can never
    open, which is the honest behaviour rather than a guess.
    """
    if not geo.is_geo_headline(title):
        return None
    named = geo.countries_in(title, [code for code, _ in move.countries])
    if not named:
        return 0.0
    driver = (move.macro_driver or "").strip()
    if driver in _GLOBAL_DRIVERS:
        return 1.0
    if driver.startswith(_COUNTRY_PREFIX) and driver[len(_COUNTRY_PREFIX) :].upper() in named:
        return 1.0
    return 0.0


def heuristic_components(
    ctx: MoveContext,
    article: ArticleInput,
    heuristic_score: ArticleScore,
    n_sources_same_story: int = 1,
) -> Components:
    """The five stored components in [0, 1], plus `timing_kind` and `geo_gate`.

    `heuristic_score` supplies the keyword category, which is what makes
    `bucket_match` and `entity_match` more than a restatement of each other:
    the first asks whether the headline is about the *kind* of thing the
    decomposition points at, the second how specific the headline is at all.

    `timing_kind` is `"cause"` for a headline published on or before the move
    day and `"report"` for one published after it — an after-the-fact write-up
    is still evidence, just weaker. It is `""` when the source gave no date,
    since neither label would be true; `score_and_link` stores that as NULL.

    `geo_gate` is not weighted with the five: it is a cap applied afterwards
    by `heuristic_relevance`, and `None` for the headlines it does not touch.
    """
    category = heuristic_score.category

    if category == ctx.routing:
        bucket_match = 1.0
    elif {category, ctx.routing} == {"industry", "company"}:
        # Adjacent buckets: a peer's headline on a company-routed move (and
        # the reverse) is partial evidence, not noise.
        bucket_match = 0.5
    else:
        bucket_match = 0.0

    entity_match = {"company": 1.0, "industry": 0.5, "macro": 0.25}.get(category, 0.0)

    published_at = article.published_at
    if published_at is None:
        timing, timing_kind = 0.5, ""
    elif ctx.date is not None and published_at.date() > ctx.date:
        timing, timing_kind = 0.7, "report"
    else:
        timing, timing_kind = 1.0, "cause"

    coverage = min(max(n_sources_same_story, 0), _COVERAGE_CAP) / _COVERAGE_CAP

    return {
        "bucket_match": bucket_match,
        "entity_match": entity_match,
        "timing": timing,
        "source_tier": source_tier(article.source),
        "coverage": coverage,
        "timing_kind": timing_kind,
        "geo_gate": geo_gate(article.title, ctx),
    }


def _component(components: Components, name: str) -> float:
    """One numeric component, 0.0 when absent or not a number."""
    value = components.get(name, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _gate(components: Components) -> float | None:
    """The stored `geo_gate`, or None when the gate did not apply."""
    value = components.get("geo_gate")
    return float(value) if isinstance(value, (int, float)) else None


def heuristic_relevance(components: Components) -> float:
    """The weighted sum of the five components, in [0, 1], after the geo cap.

    A shut gate (`geo_gate == 0.0`) caps the sum at `GEO_CAP`; an open gate and
    a gate that never applied leave the sum alone.
    """
    relevance = _clamp(
        sum(weight * _component(components, name) for name, weight in WEIGHTS.items())
    )
    if _gate(components) == 0.0:
        return min(relevance, GEO_CAP)
    return relevance


def _coverage_counts(articles: Sequence[ArticleInput]) -> dict[int, int]:
    """Article id -> distinct sources covering the same story (at least 1)."""
    groups: dict[str, set[str | None]] = {}
    for article in articles:
        groups.setdefault(story_key(article.title), set()).add(article.source)
    return {article.id: max(len(groups[story_key(article.title)]), 1) for article in articles}


def score_and_link(
    session: Session,
    move: Move,
    company: Company,
    articles: Sequence[Article],
    provider: ModelProvider,
    top_k: int = 15,
    *,
    context: MoveContext | None = None,
) -> list[MoveArticle]:
    """Score `articles` for `move` and rewrite its `move_articles` rows.

    The heuristic always runs, on every article. Only the top-`top_k` by
    heuristic relevance are shown to a non-heuristic provider; for those the
    stored relevance is the mean of the two scores and the category is the
    model's. The rest are linked with the heuristic relevance and a NULL
    `model_score`, so nothing is lost and model spend stays at one call per
    move. A headline whose `geo_gate` shut is capped at `GEO_CAP` on both
    paths, the model's mean included.

    `context` is the v1.5 escape hatch for the gate rule. `MoveContext` knows
    nothing of `company_edges` or `geo_events` — they are separate tables and
    this module does not read them — so `from_objects` builds a context with no
    countries, and :func:`geo_gate` then shuts on every geopolitical headline.
    A caller that *has* read the edges (`ingest.enrich_move`) passes the fuller
    context here and it is used verbatim. Nothing else about the scoring
    changes: the same context feeds the heuristic, the model and the gate.
    """
    if move.id is None:
        raise ValueError("score_and_link needs a persisted move: move.id is None")

    ctx = MoveContext.from_objects(move, company) if context is None else context
    # One row per article even if the caller passed a url twice: the link's
    # primary key is (move_id, article_id).
    unique: dict[int, Article] = {}
    for article in articles:
        if article.id is not None and article.id not in unique:
            unique[article.id] = article
    inputs = to_inputs(list(unique.values()))

    heuristic_scores = HeuristicProvider().score_articles(ctx, inputs)
    coverage = _coverage_counts(inputs)

    scored: list[tuple[ArticleInput, Components, float]] = []
    for article, score in zip(inputs, heuristic_scores, strict=False):
        components = heuristic_components(ctx, article, score, coverage.get(article.id, 1))
        scored.append((article, components, heuristic_relevance(components)))

    ranked = sorted(scored, key=lambda row: row[2], reverse=True)
    top = ranked[:top_k]

    model_run = provider.name != HeuristicProvider.name
    model_scores: dict[int, ArticleScore] = {}
    if model_run and top:
        model_scores = {
            result.article_id: result
            for result in provider.score_articles(ctx, [row[0] for row in top])
        }

    heuristic_category = {score.article_id: score.category for score in heuristic_scores}
    top_ids = {row[0].id for row in top}
    routing = move.routing

    # Rescoring replaces the old verdict rather than accumulating rows. Flush
    # the deletes before the inserts, or the new rows collide with the old
    # ones on the (move_id, article_id) primary key.
    for stale in session.exec(select(MoveArticle).where(col(MoveArticle.move_id) == move.id)).all():
        session.delete(stale)
    session.flush()

    links: list[MoveArticle] = []
    for article, components, base_relevance in ranked:
        category = heuristic_category.get(article.id, routing)
        relevance = base_relevance
        model_score: float | None = None
        gate = _gate(components)

        if model_run and article.id in top_ids:
            result = model_scores.get(article.id)
            # A provider that returned nothing for an article has no opinion
            # on it, which is not the same as agreeing with the heuristic:
            # score it 0 and file it under the move's own bucket.
            model_score = _clamp(result.relevance) if result is not None else 0.0
            model_category = result.category if result is not None else routing
            category = model_category if model_category in _CATEGORIES else routing
            relevance = _clamp((base_relevance + model_score) / 2)
            # The cap is on the stored relevance, not only on the heuristic
            # half of it: a model that loves a gated headline must not be able
            # to lift it back over `GEO_CAP`.
            if gate == 0.0:
                relevance = min(relevance, GEO_CAP)

        links.append(
            MoveArticle(
                move_id=move.id,
                article_id=article.id,
                relevance=relevance,
                category=category,
                provider=provider.name,
                bucket_match=_component(components, "bucket_match"),
                entity_match=_component(components, "entity_match"),
                timing=_component(components, "timing"),
                source_tier=_component(components, "source_tier"),
                coverage=_component(components, "coverage"),
                # "" means the source gave no date, so neither label is true.
                timing_kind=str(components.get("timing_kind") or "") or None,
                model_score=model_score,
                geo_gate=gate,
            )
        )

    session.add_all(links)
    session.commit()
    return sorted(links, key=lambda link: link.relevance, reverse=True)


def linked_articles(
    session: Session, move_id: int, min_relevance: float = 0.0
) -> list[tuple[Article, MoveArticle]]:
    """The articles linked to `move_id` with their links, most relevant first.

    `queries.py` has a similar read shaped for the API; this one exists so
    `explain.py` and `ingest.py` need not depend on it.
    """
    statement = (
        select(Article, MoveArticle)
        .join(MoveArticle, col(MoveArticle.article_id) == col(Article.id))
        .where(col(MoveArticle.move_id) == move_id)
        .where(col(MoveArticle.relevance) >= min_relevance)
        .order_by(col(MoveArticle.relevance).desc())
    )
    return [(article, link) for article, link in session.exec(statement).all()]
