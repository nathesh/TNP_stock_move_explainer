"""Storage and scoring: dedupe on url, the five components, the top-K cut.

Everything runs against the in-memory `session` fixture with the keyless
provider, so there is no network and no API key anywhere in this module. The
one model path that is exercised is a stub that returns nothing, which is the
case that has to degrade gracefully.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest
from sqlmodel import select

from stock_moves.db import Session
from stock_moves.models import Article, Company, Move, MoveArticle
from stock_moves.news.base import NewsItem
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
from stock_moves.scoring import (
    GEO_CAP,
    geo_gate,
    heuristic_components,
    heuristic_relevance,
    linked_articles,
    score_and_link,
    source_tier,
    to_inputs,
    upsert_articles,
)

MOVE_DATE = date(2025, 6, 3)

COMPANY_URL = "https://example.test/testcorp-guidance"
MACRO_URL = "https://example.test/fed-holds"
PEER_URL = "https://example.test/peer-tumbles"
GEO_URL = "https://example.test/taiwan-tariffs"

COMPANY_TITLE = "Test Corp shares slide after guidance cut"
GEO_TITLE = "Taiwan tariff threat rattles chip supply chains"

# On or before the move day, so `timing` is 1.0 and the weighted sum of a
# well-sourced macro headline is 0.35 + 0.05 + 0.15 + 0.15 + 0.015 = 0.715 —
# comfortably above GEO_CAP, which is what makes the cap visible.
GEO_PUBLISHED = datetime(2025, 6, 2, 12, 0, tzinfo=UTC)
UNCAPPED = 0.715


def _item(
    url: str,
    title: str,
    source: str | None,
    published_at: datetime | None,
    bucket: str,
) -> NewsItem:
    return NewsItem(
        url=url,
        title=title,
        source=source,
        published_at=published_at,
        language=None,
        news_source="google_rss",
        bucket=bucket,
    )


def _items() -> list[NewsItem]:
    """Four items for three urls: the company headline arrives twice."""
    company = _item(
        COMPANY_URL,
        "Test Corp shares slide after guidance cut",
        "Reuters",
        datetime(2025, 6, 2, 21, 30, tzinfo=UTC),
        "company",
    )
    return [
        company,
        company,
        _item(MACRO_URL, "Fed holds rates", None, None, "macro"),
        _item(
            PEER_URL,
            "PEER stock tumbles",
            "Some Blog",
            datetime(2025, 6, 3, 14, 0, tzinfo=UTC),
            "industry",
        ),
    ]


@pytest.fixture
def company(session: Session) -> Company:
    row = Company(
        ticker="TEST",
        name="Test Corp",
        sector="Technology",
        industry="Semiconductors",
        sector_etf="XLK",
        peers_json=json.dumps(["PEER"]),
    )
    session.add(row)
    session.commit()
    return row


@pytest.fixture
def move(session: Session) -> Move:
    row = Move(
        ticker="TEST",
        date=MOVE_DATE,
        ret=-0.05,
        ret_z=-2.5,
        routing="company",
        direction="down",
    )
    session.add(row)
    session.commit()
    return row


class StubProvider:
    """A "model" with no opinions: the degenerate case of a short response."""

    name: str = "stub"

    def score_articles(
        self, move: MoveContext, articles: Sequence[ArticleInput]
    ) -> list[ArticleScore]:
        return []

    def explain(
        self, move: MoveContext, scored: Sequence[tuple[ArticleInput, ArticleScore]]
    ) -> ExplanationResult:
        raise NotImplementedError

    def suggest_peers(
        self, ticker: str, name: str, sector: str | None, industry: str | None
    ) -> list[str]:
        raise NotImplementedError

    def chat(
        self,
        history: Sequence[ChatTurn],
        tools: Mapping[str, ToolFn],
        ticker: str | None,
    ) -> ChatReply:
        raise NotImplementedError


def _by_url(session: Session, links: Sequence[MoveArticle]) -> dict[str, MoveArticle]:
    urls = {article.id: article.url for article in session.exec(select(Article)).all()}
    return {urls[link.article_id]: link for link in links}


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


def test_upsert_articles_dedupes_on_url(session: Session) -> None:
    rows = upsert_articles(session, _items())

    assert len(rows) == 3
    assert [row.url for row in rows] == [COMPANY_URL, MACRO_URL, PEER_URL]
    assert all(row.id is not None for row in rows)
    assert session.exec(select(Article)).all() == rows


def test_upsert_articles_is_idempotent(session: Session) -> None:
    first = upsert_articles(session, _items())
    again = upsert_articles(session, _items())

    assert [row.id for row in again] == [row.id for row in first]
    assert len(session.exec(select(Article)).all()) == 3


def test_upsert_articles_stores_naive_utc_and_source_name(session: Session) -> None:
    rows = upsert_articles(session, _items())
    session.expire_all()
    stored = session.exec(select(Article).where(Article.url == COMPANY_URL)).one()

    assert stored.published_at is not None
    assert stored.published_at.tzinfo is None
    assert stored.published_at == datetime(2025, 6, 2, 21, 30)  # noqa: DTZ001
    assert stored.news_source == "google_rss"
    assert rows[1].published_at is None  # the macro item carried no date


def test_to_inputs_preserves_order(session: Session) -> None:
    rows = upsert_articles(session, _items())
    inputs = to_inputs(rows)

    assert [i.id for i in inputs] == [row.id for row in rows]
    assert inputs[0].title == "Test Corp shares slide after guidance cut"


# --------------------------------------------------------------------------- #
# Components, unit-tested without a database
# --------------------------------------------------------------------------- #


def _ctx() -> MoveContext:
    return MoveContext(
        ticker="TEST",
        company_name="Test Corp",
        date=MOVE_DATE,
        ret=-0.05,
        ret_z=-2.5,
        gap_ret=None,
        intraday_ret=None,
        vol_z=None,
        mkt_component=None,
        sector_component=None,
        idio_component=None,
        routing="company",
        direction="down",
        regime_mkt=None,
        regime_sector=None,
        near_earnings=False,
        sector="Technology",
        industry="Semiconductors",
        peers=("PEER",),
        peer_comove=None,
    )


def _input(
    source: str | None = "Reuters",
    published_at: datetime | None = None,
    title: str = COMPANY_TITLE,
) -> ArticleInput:
    return ArticleInput(
        id=1,
        title=title,
        source=source,
        url=COMPANY_URL,
        published_at=published_at,
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Reuters", 1.0),
        ("The Wall Street Journal", 1.0),
        ("Barron's", 1.0),
        ("Yahoo Finance", 0.7),
        ("Seeking Alpha", 0.7),
        ("Some Blog", 0.5),
        (None, 0.4),
    ],
)
def test_source_tier_list(source: str | None, expected: float) -> None:
    assert source_tier(source) == expected

    components = heuristic_components(
        _ctx(), _input(source=source), ArticleScore(1, 1.0, "company")
    )
    assert components["source_tier"] == expected


@pytest.mark.parametrize(
    ("published_at", "timing", "kind"),
    [
        (datetime(2025, 6, 2, 12, 0, tzinfo=UTC), 1.0, "cause"),
        (datetime(2025, 6, 3, 12, 0, tzinfo=UTC), 1.0, "cause"),
        (datetime(2025, 6, 4, 12, 0, tzinfo=UTC), 0.7, "report"),
        (None, 0.5, ""),
    ],
)
def test_timing_and_timing_kind(published_at: datetime | None, timing: float, kind: str) -> None:
    components = heuristic_components(
        _ctx(), _input(published_at=published_at), ArticleScore(1, 1.0, "company")
    )

    assert components["timing"] == timing
    assert components["timing_kind"] == kind


@pytest.mark.parametrize(
    ("category", "bucket", "entity"),
    [
        ("company", 1.0, 1.0),
        ("industry", 0.5, 0.5),
        ("macro", 0.0, 0.25),
    ],
)
def test_bucket_and_entity_match(category: str, bucket: float, entity: float) -> None:
    components = heuristic_components(_ctx(), _input(), ArticleScore(1, 1.0, category))

    assert components["bucket_match"] == bucket
    assert components["entity_match"] == entity


def test_coverage_is_capped_at_ten_sources() -> None:
    one = heuristic_components(_ctx(), _input(), ArticleScore(1, 1.0, "company"), 1)
    four = heuristic_components(_ctx(), _input(), ArticleScore(1, 1.0, "company"), 4)
    lots = heuristic_components(_ctx(), _input(), ArticleScore(1, 1.0, "company"), 25)

    assert (one["coverage"], four["coverage"], lots["coverage"]) == (0.1, 0.4, 1.0)


def test_heuristic_relevance_is_the_weighted_sum() -> None:
    components = heuristic_components(
        _ctx(),
        _input(published_at=datetime(2025, 6, 2, 12, 0, tzinfo=UTC)),
        ArticleScore(1, 1.0, "company"),
    )

    # 0.35*1 + 0.20*1 + 0.15*1 + 0.15*1 + 0.15*0.1
    assert heuristic_relevance(components) == pytest.approx(0.865)


# --------------------------------------------------------------------------- #
# Linking
# --------------------------------------------------------------------------- #


def test_score_and_link_with_the_heuristic_provider(
    session: Session, company: Company, move: Move
) -> None:
    articles = upsert_articles(session, _items())

    links = score_and_link(session, move, company, articles, HeuristicProvider())

    assert len(links) == 3
    by_url = _by_url(session, links)

    # Highest relevance is the company headline, on a company-routed move.
    assert links == sorted(links, key=lambda link: link.relevance, reverse=True)
    assert by_url[COMPANY_URL] is links[0]
    assert by_url[COMPANY_URL].category == "company"
    assert by_url[MACRO_URL].category == "macro"
    assert by_url[PEER_URL].category == "industry"

    assert by_url[COMPANY_URL].timing_kind == "cause"
    assert by_url[MACRO_URL].timing_kind is None  # no date from the source

    for link in links:
        assert link.provider == "heuristic"
        assert link.model_score is None
        assert 0.0 <= link.relevance <= 1.0
        for name in (
            "bucket_match",
            "entity_match",
            "timing",
            "source_tier",
            "coverage",
        ):
            assert 0.0 <= getattr(link, name) <= 1.0


def test_score_and_link_replaces_existing_links(
    session: Session, company: Company, move: Move
) -> None:
    articles = upsert_articles(session, _items())
    provider = HeuristicProvider()

    first = score_and_link(session, move, company, articles, provider)
    again = score_and_link(session, move, company, articles, provider)

    assert len(session.exec(select(MoveArticle)).all()) == 3
    assert [link.relevance for link in again] == [link.relevance for link in first]


def test_linked_articles_filters_on_relevance(
    session: Session, company: Company, move: Move
) -> None:
    articles = upsert_articles(session, _items())
    links = score_and_link(session, move, company, articles, HeuristicProvider())
    assert links[0].relevance > 0.7 > links[1].relevance

    everything = linked_articles(session, int(move.id or 0))
    assert [article.url for article, _ in everything] == [
        COMPANY_URL,
        PEER_URL,
        MACRO_URL,
    ]

    top = linked_articles(session, int(move.id or 0), min_relevance=0.7)
    assert [article.url for article, _ in top] == [COMPANY_URL]
    assert top[0][1].relevance == pytest.approx(links[0].relevance)


def test_below_the_cut_articles_are_still_linked(
    session: Session, company: Company, move: Move
) -> None:
    articles = upsert_articles(session, _items())

    links = score_and_link(session, move, company, articles, StubProvider(), top_k=1)
    by_url = _by_url(session, links)

    # Only the top-1 was shown to the provider; the other two keep the
    # heuristic verdict and a NULL model_score, so nothing is lost.
    assert by_url[COMPANY_URL].model_score == pytest.approx(0.0)
    assert by_url[MACRO_URL].model_score is None
    assert by_url[PEER_URL].model_score is None
    assert by_url[MACRO_URL].category == "macro"
    assert all(link.provider == "stub" for link in links)


def test_a_model_provider_that_returns_nothing(
    session: Session, company: Company, move: Move
) -> None:
    articles = upsert_articles(session, _items())
    baseline = {
        link.article_id: link.relevance
        for link in score_and_link(session, move, company, articles, HeuristicProvider())
    }

    links = score_and_link(session, move, company, articles, StubProvider())

    assert len(links) == 3
    for link in links:
        # Missing score -> 0.0 and the move's own routing bucket; the stored
        # relevance is the mean of the heuristic and that zero.
        assert link.model_score == pytest.approx(0.0)
        assert link.category == "company"
        assert link.relevance == pytest.approx(baseline[link.article_id] / 2)


# --------------------------------------------------------------------------- #
# The geopolitical gate (v1.5 plan, decision 7)
# --------------------------------------------------------------------------- #


def _geo_ctx(
    countries: tuple[tuple[str, float], ...] = (("TW", 1.0),),
    macro_driver: str | None = "country:TW",
) -> MoveContext:
    """A macro-routed move with country edges: the gate's precondition."""
    return replace(_ctx(), routing="macro", countries=countries, macro_driver=macro_driver)


def _geo_components(ctx: MoveContext, title: str = GEO_TITLE) -> dict[str, float | str | None]:
    """The components of one well-sourced macro headline on `ctx`."""
    return heuristic_components(
        ctx,
        _input(published_at=GEO_PUBLISHED, title=title),
        ArticleScore(1, 1.0, "macro"),
    )


@pytest.mark.parametrize(
    ("macro_driver", "expected"),
    [
        ("country:TW", 1.0),  # the country's own factor moved
        ("oil", 1.0),  # a global channel is enough
        ("dollar", 1.0),
        ("country:CN", 0.0),  # a factor, but not this headline's country
        ("rates", 0.0),  # neither oil nor dollar
        (None, 0.0),  # no macro factor moved at all
    ],
)
def test_geo_gate_needs_the_edge_and_the_driver(macro_driver: str | None, expected: float) -> None:
    assert geo_gate(GEO_TITLE, _geo_ctx(macro_driver=macro_driver)) == expected


def test_an_open_gate_leaves_the_relevance_alone() -> None:
    components = _geo_components(_geo_ctx())

    assert components["geo_gate"] == 1.0
    assert heuristic_relevance(components) == pytest.approx(UNCAPPED)
    assert heuristic_relevance(components) > GEO_CAP


def test_a_shut_gate_caps_the_relevance() -> None:
    # The company has the Taiwan edge, but nothing macro moved that day, so
    # the headline is a coincidence rather than the cause.
    components = _geo_components(_geo_ctx(macro_driver=None))

    assert components["geo_gate"] == 0.0
    assert heuristic_relevance(components) <= GEO_CAP
    assert heuristic_relevance(components) == pytest.approx(GEO_CAP)


def test_a_country_with_no_edge_shuts_the_gate() -> None:
    # China moved the factor, the headline is about Taiwan, and Taiwan is not
    # a country this company touches.
    components = _geo_components(_geo_ctx(countries=(("CN", 1.0),), macro_driver="country:CN"))

    assert geo_gate(GEO_TITLE, _geo_ctx(countries=(("CN", 1.0),), macro_driver="country:CN")) == 0.0
    assert components["geo_gate"] == 0.0
    assert heuristic_relevance(components) == pytest.approx(GEO_CAP)


def test_a_non_geo_headline_has_no_gate() -> None:
    components = _geo_components(_geo_ctx(), title=COMPANY_TITLE)

    assert geo_gate(COMPANY_TITLE, _geo_ctx()) is None
    assert components["geo_gate"] is None
    # `None` is "not applicable", not "shut": nothing is capped.
    assert heuristic_relevance(components) == pytest.approx(UNCAPPED)


class EagerProvider(StubProvider):
    """A "model" that loves every headline — what the cap has to survive."""

    name: str = "eager"

    def score_articles(
        self, move: MoveContext, articles: Sequence[ArticleInput]
    ) -> list[ArticleScore]:
        return [ArticleScore(article.id, 1.0, "macro") for article in articles]


def _geo_article(session: Session) -> list[Article]:
    return upsert_articles(
        session,
        [_item(GEO_URL, GEO_TITLE, "Reuters", GEO_PUBLISHED, "macro")],
    )


def _with_countries(company: Company, countries: tuple[tuple[str, float], ...]) -> SimpleNamespace:
    """The `companies` row plus its `country` edges.

    `company_edges` is its own table, so the edges are attached by the caller
    rather than read off the row; `MoveContext.from_objects` reads every edge
    field with `getattr`, which is what lets a plain record stand in here.
    """
    return SimpleNamespace(
        ticker=company.ticker,
        name=company.name,
        sector=company.sector,
        industry=company.industry,
        peers=company.peers,
        countries=countries,
    )


def test_score_and_link_stores_a_shut_gate_and_caps_the_model(
    session: Session, company: Company, move: Move
) -> None:
    articles = _geo_article(session)

    links = score_and_link(session, move, company, articles, EagerProvider())

    assert len(links) == 1
    link = links[0]
    # The fixture company has no country edges — the keyless case — so the
    # gate shuts, and the model's 1.0 cannot lift the mean back over the cap.
    assert link.geo_gate == pytest.approx(0.0)
    assert link.model_score == pytest.approx(1.0)
    assert link.relevance == pytest.approx(GEO_CAP)


def test_score_and_link_stores_an_open_gate(session: Session, company: Company, move: Move) -> None:
    move.routing = "macro"
    move.macro_driver = "country:TW"
    session.add(move)
    session.commit()
    articles = _geo_article(session)

    links = score_and_link(
        session, move, _with_countries(company, (("TW", 1.0),)), articles, EagerProvider()
    )

    assert links[0].geo_gate == pytest.approx(1.0)
    assert links[0].relevance > GEO_CAP


def test_a_gate_that_never_applied_is_stored_as_null(
    session: Session, company: Company, move: Move
) -> None:
    links = score_and_link(
        session, move, company, upsert_articles(session, _items()), HeuristicProvider()
    )

    assert all(link.geo_gate is None for link in links)
