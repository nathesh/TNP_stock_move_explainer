"""Tests for the explanation layer (DESIGN section 4).

Three things are worth testing and nothing else is: that the top-K cut handed
to the provider is the right articles in the right order, that the cache calls
the provider exactly once per move (and once more per `refresh`), and that a
provider returning nonsense cannot get that nonsense into the database. The
last one is the reason this layer exists, so it is tested with a provider that
returns a bad category, an out-of-range confidence and a citation to an article
it was never shown.

No network and no API key: `HeuristicProvider` is the real provider under test.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import func
from sqlmodel import Session, select

from stock_moves.explain import (
    build_context,
    explanation_to_result,
    get_or_create_explanation,
    top_scored,
)
from stock_moves.models import (
    Article,
    Company,
    CompanyEdge,
    Explanation,
    GeoEvent,
    Move,
    MoveArticle,
)
from stock_moves.providers.base import (
    ArticleInput,
    ArticleScore,
    ChatReply,
    ChatTurn,
    ExplanationResult,
    MoveContext,
    ToolFn,
)
from stock_moves.providers.heuristic import HeuristicProvider
from stock_moves.providers.prompts import EXPLAIN_SYSTEM, move_block, narrative_block

MOVE_DATE = date(2025, 6, 3)


def stamp(hour: int, minute: int) -> datetime:
    """A naive-UTC timestamp for `MOVE_DATE`, matching `models.utcnow()`."""
    return datetime(
        MOVE_DATE.year, MOVE_DATE.month, MOVE_DATE.day, hour, minute, tzinfo=UTC
    ).replace(tzinfo=None)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def seed(session: Session) -> tuple[Move, Company, Article, Article]:
    """A company, one company-routed down move, and two linked headlines.

    Relevances are 1.0 and 0.2 so the top-K cut and the heuristic provider's
    0.5 relevance floor both have something to bite on. The component columns
    on `move_articles` are left at their defaults — this module does not care
    how a relevance was produced, only that it is stored.
    """
    company = Company(
        ticker="TEST",
        name="Testco Industries Inc",
        sector="Technology",
        industry="Semiconductors",
        sector_etf="XLK",
        peers_json='["PEER"]',
    )
    move = Move(
        ticker="TEST",
        date=MOVE_DATE,
        ret=-0.061,
        ret_z=-2.6,
        gap_ret=-0.04,
        intraday_ret=-0.021,
        vol_z=2.2,
        mkt_component=-0.002,
        sector_component=-0.004,
        idio_component=-0.055,
        routing="company",
        direction="down",
        regime_mkt="bull",
        regime_sector="bull",
        near_earnings=True,
    )
    hit = Article(
        url="https://example.test/hit",
        title="Testco Industries cuts guidance after weak quarter",
        source="Reuters",
        published_at=stamp(13, 30),
        news_source="google_rss",
    )
    miss = Article(
        url="https://example.test/miss",
        title="Unrelated market wrap",
        source="Blogspot",
        published_at=stamp(9, 0),
        news_source="google_rss",
    )
    session.add_all([company, move, hit, miss])
    session.commit()
    for article, relevance, category in ((hit, 1.0, "company"), (miss, 0.2, "company")):
        assert move.id is not None
        assert article.id is not None
        session.add(
            MoveArticle(
                move_id=move.id,
                article_id=article.id,
                relevance=relevance,
                category=category,
                provider="heuristic",
            )
        )
    session.commit()
    return move, company, hit, miss


def explanation_count(session: Session) -> int:
    """How many rows the `explanations` table holds (must never exceed one here)."""
    return int(session.exec(select(func.count()).select_from(Explanation)).one())


class CountingProvider:
    """Wraps a real provider and counts `explain` calls.

    Everything else is delegated, so this stays a `ModelProvider` without
    restating the interface.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.explain_calls = 0
        self.name: str = str(getattr(inner, "name", "counting"))

    def explain(
        self, move: MoveContext, scored: Sequence[tuple[ArticleInput, ArticleScore]]
    ) -> ExplanationResult:
        self.explain_calls += 1
        return self._inner.explain(move, scored)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


class BadProvider:
    """A provider that violates every promise the schema makes."""

    name: str = "bad"

    def __init__(self) -> None:
        self.seen_titles: list[str] = []

    def score_articles(
        self, move: MoveContext, articles: Sequence[ArticleInput]
    ) -> list[ArticleScore]:
        return [ArticleScore(a.id, 1.0, "company") for a in articles]

    def explain(
        self, move: MoveContext, scored: Sequence[tuple[ArticleInput, ArticleScore]]
    ) -> ExplanationResult:
        self.seen_titles = [article.title for article, _ in scored]
        return ExplanationResult(
            summary="  it was the weather  ",
            primary_category="nonsense",
            confidence=7.0,
            cited_article_ids=(9_999,),
            unexplained=False,
        )

    def suggest_peers(
        self, ticker: str, name: str, sector: str | None, industry: str | None
    ) -> list[str]:
        return []

    def chat(
        self,
        history: Sequence[ChatTurn],
        tools: Mapping[str, ToolFn],
        ticker: str | None,
    ) -> ChatReply:
        return ChatReply("no", [])


# --------------------------------------------------------------------------- #
# top_scored / build_context
# --------------------------------------------------------------------------- #


def test_top_scored_orders_by_relevance_and_maps_to_base_types(
    session: Session,
) -> None:
    move, _, hit, miss = seed(session)
    assert move.id is not None

    scored = top_scored(session, move.id)

    assert [score.article_id for _, score in scored] == [hit.id, miss.id]
    assert [round(score.relevance, 3) for _, score in scored] == [1.0, 0.2]
    article, score = scored[0]
    assert isinstance(article, ArticleInput)
    assert isinstance(score, ArticleScore)
    assert article.title == hit.title
    assert article.source == "Reuters"
    assert score.category == "company"


def test_top_scored_cuts_to_top_k(session: Session) -> None:
    move, _, hit, _ = seed(session)
    assert move.id is not None

    assert [s.article_id for _, s in top_scored(session, move.id, top_k=1)] == [hit.id]
    assert top_scored(session, move.id, top_k=0) == []
    assert top_scored(session, 12_345) == []


def test_build_context_carries_the_decomposition_and_peers(session: Session) -> None:
    move, company, _, _ = seed(session)

    ctx = build_context(move, company)

    assert ctx.ticker == "TEST"
    assert ctx.company_name == "Testco Industries Inc"
    assert ctx.date == MOVE_DATE
    assert ctx.routing == "company"
    assert ctx.direction == "down"
    assert ctx.idio_component == pytest.approx(-0.055)
    assert ctx.near_earnings is True
    assert ctx.peers == ("PEER",)


# --------------------------------------------------------------------------- #
# The cache
# --------------------------------------------------------------------------- #


def test_first_call_writes_a_cited_row(session: Session, no_api_key: None) -> None:
    move, company, hit, miss = seed(session)

    explanation = get_or_create_explanation(session, move, company, HeuristicProvider())

    assert explanation.id is not None
    assert explanation.move_id == move.id
    assert explanation.primary_category == "company"
    assert explanation.unexplained is False
    assert explanation.provider == "heuristic"
    assert 0.0 <= explanation.confidence <= 1.0
    assert explanation.summary != ""
    # Only the 1.0 article clears the provider's relevance floor.
    assert explanation.cited_article_ids == [hit.id]
    assert miss.id not in explanation.cited_article_ids
    assert explanation_count(session) == 1


def test_second_call_is_cached_and_does_not_call_the_provider(
    session: Session, no_api_key: None
) -> None:
    move, company, _, _ = seed(session)
    provider = CountingProvider(HeuristicProvider())

    first = get_or_create_explanation(session, move, company, provider)
    second = get_or_create_explanation(session, move, company, provider)

    assert provider.explain_calls == 1
    assert second.id == first.id
    assert second.summary == first.summary
    assert explanation_count(session) == 1


def test_refresh_recomputes_in_place(session: Session, no_api_key: None) -> None:
    move, company, _, _ = seed(session)
    provider = CountingProvider(HeuristicProvider())

    first = get_or_create_explanation(session, move, company, provider)
    first_id = first.id
    refreshed = get_or_create_explanation(session, move, company, provider, refresh=True)

    assert provider.explain_calls == 2
    assert refreshed.id == first_id
    assert explanation_count(session) == 1


def test_refresh_replaces_the_stored_fields(session: Session, no_api_key: None) -> None:
    move, company, _, _ = seed(session)
    get_or_create_explanation(session, move, company, HeuristicProvider())

    refreshed = get_or_create_explanation(session, move, company, BadProvider(), refresh=True)

    assert refreshed.provider == "bad"
    assert refreshed.primary_category == "unexplained"
    assert explanation_count(session) == 1


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_bad_provider_output_is_normalised(session: Session, no_api_key: None) -> None:
    move, company, _, _ = seed(session)
    provider = BadProvider()

    explanation = get_or_create_explanation(session, move, company, provider)

    assert explanation.primary_category == "unexplained"
    assert explanation.confidence == 1.0
    assert explanation.cited_article_ids == []
    # A category we had to replace means the row is not an explanation.
    assert explanation.unexplained is True
    assert explanation.summary == "it was the weather"
    assert explanation_count(session) == 1


def test_confidence_floor_and_unknown_citations_only(session: Session, no_api_key: None) -> None:
    move, company, hit, miss = seed(session)

    class Negative(BadProvider):
        def explain(
            self, move: MoveContext, scored: Sequence[tuple[ArticleInput, ArticleScore]]
        ) -> ExplanationResult:
            return ExplanationResult(
                summary="macro day",
                primary_category="macro",
                confidence=-3.0,
                # One real id, one it was never shown, and a duplicate.
                cited_article_ids=(miss.id or 0, 4_242, miss.id or 0),
                unexplained=False,
            )

    explanation = get_or_create_explanation(session, move, company, Negative())

    assert explanation.primary_category == "macro"
    assert explanation.confidence == 0.0
    assert explanation.cited_article_ids == [miss.id]
    assert hit.id not in explanation.cited_article_ids
    assert explanation.unexplained is False


def test_provider_only_sees_the_top_k_articles(session: Session, no_api_key: None) -> None:
    move, company, hit, _ = seed(session)
    provider = BadProvider()

    get_or_create_explanation(session, move, company, provider, top_k=1)

    assert provider.seen_titles == [hit.title]


def test_explanation_to_result_round_trips(session: Session, no_api_key: None) -> None:
    move, company, hit, _ = seed(session)
    explanation = get_or_create_explanation(session, move, company, HeuristicProvider())

    result = explanation_to_result(explanation)

    assert isinstance(result, ExplanationResult)
    assert result.summary == explanation.summary
    assert result.primary_category == "company"
    assert result.confidence == explanation.confidence
    assert result.cited_article_ids == (hit.id,)
    assert result.unexplained is False


def test_unpersisted_move_is_rejected(session: Session, no_api_key: None) -> None:
    _, company, _, _ = seed(session)
    orphan = Move(
        ticker="TEST",
        date=date(2025, 7, 1),
        ret=-0.03,
        ret_z=-2.1,
        routing="macro",
        direction="down",
    )

    with pytest.raises(ValueError, match="move.id"):
        get_or_create_explanation(session, orphan, company, HeuristicProvider())


# --------------------------------------------------------------------------- #
# The relationship layer (v1.5 decisions 1 and 5)
# --------------------------------------------------------------------------- #


def seed_edges(session: Session) -> None:
    """Edges for TEST, edges for someone else, and three country-days.

    The rows that must *not* reach the context are seeded alongside the ones
    that must: another company's competitor, a `factor` edge (which is not a
    ticker), a country with no edge, and a geo event outside the move's window.
    """
    session.add_all(
        [
            CompanyEdge(src="TEST", dst="RIVAL", relation="competitor", weight=0.9, source="model"),
            CompanyEdge(src="TEST", dst="SUPP", relation="supplier", weight=0.7, source="model"),
            CompanyEdge(src="TEST", dst="CUST", relation="customer", weight=0.5, source="model"),
            CompanyEdge(src="TEST", dst="TW", relation="country", weight=0.4, source="model"),
            CompanyEdge(src="TEST", dst="CN", relation="country", weight=0.6, source="model"),
            CompanyEdge(src="TEST", dst="oil", relation="factor", weight=0.8, source="prices"),
            CompanyEdge(
                src="OTHER", dst="ALIEN", relation="competitor", weight=1.0, source="model"
            ),
        ]
    )
    session.add_all(
        [
            GeoEvent(
                date=MOVE_DATE,
                country="TW",
                headline_count=14,
                sample_titles_json='["Taiwan hit by new export controls"]',
                news_source="google_rss",
            ),
            GeoEvent(
                date=MOVE_DATE - timedelta(days=9),
                country="TW",
                headline_count=3,
                sample_titles_json="[]",
                news_source="google_rss",
            ),
            GeoEvent(
                date=MOVE_DATE,
                country="JP",
                headline_count=8,
                sample_titles_json="[]",
                news_source="google_rss",
            ),
        ]
    )
    session.commit()


def test_build_context_without_a_session_is_the_v1_context(session: Session) -> None:
    move, company, _, _ = seed(session)
    seed_edges(session)

    ctx = build_context(move, company)

    assert ctx.peers == ("PEER",)
    assert ctx.competitors == ()
    assert ctx.suppliers == () and ctx.customers == ()
    assert ctx.countries == () and ctx.geo_events == ()


def test_build_context_with_a_session_fills_the_edges_and_events(session: Session) -> None:
    move, company, _, _ = seed(session)
    seed_edges(session)

    ctx = build_context(move, company, session)

    assert ctx.competitors == ("RIVAL",)
    assert ctx.suppliers == ("SUPP",)
    assert ctx.customers == ("CUST",)
    # Largest weight first, which is the order the gate and the geo queries
    # walk; the `factor` edge is not a ticker and belongs to none of the three.
    assert ctx.countries == (("CN", 0.6), ("TW", 0.4))
    assert "oil" not in ctx.competitors + ctx.suppliers + ctx.customers
    assert "ALIEN" not in ctx.competitors
    # One line, for the country this company has an edge to, inside the move's
    # +/-1 day window.
    assert ctx.geo_events == (
        f"{MOVE_DATE:%Y-%m-%d} TW 14 headlines: Taiwan hit by new export controls",
    )
    # The v1 half is untouched.
    assert ctx.peers == ("PEER",)
    assert ctx.routing == "company"


def test_a_session_with_no_edges_leaves_the_context_as_v1(session: Session) -> None:
    """The keyless install: `suggest_relations` stores nothing, so the context
    is v1-shaped even though a session was handed in."""
    move, company, _, _ = seed(session)

    ctx = build_context(move, company, session)

    assert ctx.competitors == () and ctx.countries == () and ctx.geo_events == ()


def test_get_or_create_explanation_hands_the_provider_the_edges(
    session: Session, no_api_key: None
) -> None:
    move, company, _, _ = seed(session)
    seed_edges(session)

    class Capturing(BadProvider):
        context: MoveContext | None = None

        def explain(
            self, move: MoveContext, scored: Sequence[tuple[ArticleInput, ArticleScore]]
        ) -> ExplanationResult:
            Capturing.context = move
            return ExplanationResult("said", "company", 0.5, (), False)

    get_or_create_explanation(session, move, company, Capturing())

    assert Capturing.context is not None
    assert Capturing.context.competitors == ("RIVAL",)
    assert Capturing.context.geo_events != ()


# --------------------------------------------------------------------------- #
# The prompt block (v1.5 decision 9)
# --------------------------------------------------------------------------- #


def context_with(**overrides: Any) -> MoveContext:
    base: dict[str, Any] = {
        "ticker": "TEST",
        "company_name": "Testco Industries Inc",
        "date": MOVE_DATE,
        "ret": -0.061,
        "ret_z": -2.6,
        "gap_ret": None,
        "intraday_ret": None,
        "vol_z": None,
        "mkt_component": -0.002,
        "sector_component": -0.004,
        "idio_component": -0.055,
        "routing": "company",
        "direction": "down",
        "regime_mkt": None,
        "regime_sector": None,
        "near_earnings": False,
        "sector": "Technology",
        "industry": "Semiconductors",
        "peers": ("PEER",),
        "peer_comove": 0.01,
    }
    base.update(overrides)
    return MoveContext(**base)


def test_move_block_drops_the_new_keys_when_there_is_nothing_to_say() -> None:
    payload = json.loads(move_block(context_with()))

    for key in (
        "sub_routing",
        "macro_driver",
        "macro_driver_component",
        "rival_comove",
        "chain_comove",
        "competitors",
        "suppliers",
        "customers",
        "countries",
        "geo_events",
    ):
        assert key not in payload
    # The v1 block is unchanged by their absence.
    assert payload["ticker"] == "TEST"
    assert payload["peers"] == ["PEER"]


def test_move_block_carries_the_new_keys_when_they_are_set() -> None:
    payload = json.loads(
        move_block(
            context_with(
                sub_routing="country:TW",
                macro_driver="country:TW",
                macro_driver_component=-0.0234567,
                rival_comove=0.031,
                chain_comove=-0.036,
                competitors=("RIVAL",),
                suppliers=("SUPP",),
                customers=("CUST",),
                countries=(("TW", 0.4),),
                geo_events=("2025-06-03 TW 14 headlines: Taiwan hit",),
            )
        )
    )

    assert payload["sub_routing"] == "country:TW"
    assert payload["macro_driver"] == "country:TW"
    assert payload["macro_driver_component"] == pytest.approx(-0.02346, abs=1e-5)
    assert payload["rival_comove"] == pytest.approx(0.031)
    assert payload["chain_comove"] == pytest.approx(-0.036)
    assert payload["competitors"] == ["RIVAL"]
    assert payload["suppliers"] == ["SUPP"] and payload["customers"] == ["CUST"]
    assert payload["countries"] == [{"country": "TW", "weight": 0.4}]
    assert payload["geo_events"] == ["2025-06-03 TW 14 headlines: Taiwan hit"]


def test_the_narrative_block_says_the_new_signals_too() -> None:
    """The keyed prompt and the keyless summary are the same sentences."""
    said = narrative_block(
        context_with(sub_routing="share_shift", rival_comove=0.031, competitors=("RIVAL",))
    )
    assert "Share shift: RIVAL rose 3.1% the same day." in said


def test_the_explain_prompt_tells_the_model_what_the_sub_buckets_mean() -> None:
    assert "share_shift" in EXPLAIN_SYSTEM
    assert "country:XX" in EXPLAIN_SYSTEM
    assert "via <country> exposure" in EXPLAIN_SYSTEM
    # The geo count is context, not an article: it may never be cited on its own.
    assert "geo_events" in EXPLAIN_SYSTEM
