"""Tables, constraints, json properties and settings resolution."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from stock_moves.db import Session, session_scope
from stock_moves.models import (
    EDGE_RELATIONS,
    EDGE_SOURCES,
    Article,
    ChatMessage,
    Company,
    CompanyEdge,
    Explanation,
    GeoEvent,
    Move,
    MoveArticle,
    Price,
)
from stock_moves.settings import get_settings

MOVE_DATE = date(2026, 3, 12)
# SQLite stores no offset, so the models keep naive UTC timestamps.
PUBLISHED_AT = datetime(2026, 3, 11, 21, 30, tzinfo=UTC).replace(tzinfo=None)


def _company() -> Company:
    return Company(
        ticker="NVDA",
        name="NVIDIA Corporation",
        sector="Technology",
        industry="Semiconductors",
        sector_etf="XLK",
        peers_json=json.dumps(["AMD", "AVGO", "INTC"]),
        peers_source="sector_etf_holdings",
    )


def _price(move_date: date = MOVE_DATE) -> Price:
    return Price(
        ticker="NVDA",
        date=move_date,
        open=100.0,
        high=112.0,
        low=99.0,
        close=110.0,
        volume=5_000_000.0,
        ret=0.10,
        gap_ret=0.01,
        intraday_ret=0.09,
        ret_z=3.4,
        vol_z=2.1,
        mkt_component=0.01,
        sector_component=0.02,
        idio_component=0.07,
        routing="company",
        regime_mkt="bull",
        regime_sector="bull",
        near_earnings=True,
        near_fomc=False,
        near_cpi=True,
        macro_driver="oil",
        macro_driver_component=-0.004,
    )


def _move() -> Move:
    return Move(
        ticker="NVDA",
        date=MOVE_DATE,
        ret=0.10,
        ret_z=3.4,
        gap_ret=0.01,
        intraday_ret=0.09,
        vol_z=2.1,
        mkt_component=0.01,
        sector_component=0.02,
        idio_component=0.07,
        routing="company",
        direction="up",
        regime_mkt="bull",
        regime_sector="bull",
        near_earnings=True,
        near_fomc=False,
        near_cpi=True,
        peer_comove=0.012,
        sub_routing="share_shift",
        macro_driver="oil",
        macro_driver_component=-0.004,
        rival_comove=-0.031,
        chain_comove=0.002,
    )


def test_company_row_trips_and_peers_parse(session: Session) -> None:
    session.add(_company())
    session.commit()

    stored = session.exec(select(Company).where(Company.ticker == "NVDA")).one()
    assert stored.name == "NVIDIA Corporation"
    assert stored.sector_etf == "XLK"
    assert stored.peers_source == "sector_etf_holdings"
    assert stored.peers == ["AMD", "AVGO", "INTC"]
    assert isinstance(stored.updated_at, datetime)


def test_company_peers_default_and_malformed_json() -> None:
    assert Company(ticker="KO", name="Coca-Cola").peers == []
    assert Company(ticker="KO", name="Coca-Cola", peers_json="not json").peers == []
    assert Company(ticker="KO", name="Coca-Cola", peers_json='{"a": 1}').peers == []


def test_full_chain_round_trip(session: Session) -> None:
    """A price day, the move on it, an article linked to it, its explanation, a chat turn."""
    session.add(_company())
    price = _price()
    move = _move()
    article = Article(
        url="https://news.google.com/rss/articles/abc123",
        title="Nvidia beats on datacenter revenue",
        source="Reuters",
        language=None,
        published_at=PUBLISHED_AT,
        news_source="google_rss",
    )
    session.add(price)
    session.add(move)
    session.add(article)
    session.commit()

    assert move.id is not None
    assert article.id is not None

    link = MoveArticle(
        move_id=move.id,
        article_id=article.id,
        relevance=0.79,
        category="company",
        provider="anthropic",
        bucket_match=1.0,
        entity_match=1.0,
        timing=1.0,
        source_tier=0.9,
        coverage=0.6,
        timing_kind="cause",
        model_score=0.9,
    )
    explanation = Explanation(
        move_id=move.id,
        summary="Nvidia rose 10% on an earnings beat; the move is idiosyncratic.",
        primary_category="company",
        confidence=0.72,
        cited_article_ids_json=json.dumps([article.id]),
        unexplained=False,
        provider="anthropic",
    )
    chat = ChatMessage(
        session_id="s-1",
        role="assistant",
        content="The biggest up day was 2026-03-12.",
        tool_calls_json=json.dumps([{"name": "list_moves", "args": {"ticker": "NVDA"}}]),
    )
    session.add(link)
    session.add(explanation)
    session.add(chat)
    session.commit()

    stored_price = session.exec(select(Price).where(Price.ticker == "NVDA")).one()
    assert stored_price.date == MOVE_DATE
    assert stored_price.close == 110.0
    assert stored_price.routing == "company"
    assert stored_price.near_earnings is True
    assert stored_price.near_fomc is False
    assert stored_price.near_cpi is True
    assert stored_price.macro_driver == "oil"
    assert stored_price.macro_driver_component == pytest.approx(-0.004)

    stored_move = session.exec(select(Move).where(Move.date == MOVE_DATE)).one()
    assert stored_move.direction == "up"
    assert stored_move.ticker == stored_price.ticker
    assert stored_move.date == stored_price.date
    assert stored_move.peer_comove == pytest.approx(0.012)
    assert stored_move.near_cpi is True
    assert stored_move.sub_routing == "share_shift"
    assert stored_move.macro_driver == "oil"
    assert stored_move.macro_driver_component == pytest.approx(-0.004)
    assert stored_move.rival_comove == pytest.approx(-0.031)
    assert stored_move.chain_comove == pytest.approx(0.002)

    stored_link = session.exec(
        select(MoveArticle).where(MoveArticle.move_id == stored_move.id)
    ).one()
    assert stored_link.article_id == article.id
    assert stored_link.relevance == pytest.approx(0.79)
    assert stored_link.category == "company"
    assert stored_link.provider == "anthropic"
    assert stored_link.timing_kind == "cause"
    assert stored_link.model_score == pytest.approx(0.9)
    components = (
        stored_link.bucket_match,
        stored_link.entity_match,
        stored_link.timing,
        stored_link.source_tier,
        stored_link.coverage,
    )
    assert components == (1.0, 1.0, 1.0, 0.9, 0.6)

    stored_article = session.exec(select(Article)).one()
    assert stored_article.news_source == "google_rss"
    assert stored_article.source == "Reuters"
    assert stored_article.language is None
    assert isinstance(stored_article.fetched_at, datetime)

    stored_explanation = session.exec(
        select(Explanation).where(Explanation.move_id == stored_move.id)
    ).one()
    assert stored_explanation.primary_category == "company"
    assert stored_explanation.unexplained is False
    assert stored_explanation.cited_article_ids == [article.id]

    stored_chat = session.exec(select(ChatMessage).where(ChatMessage.session_id == "s-1")).one()
    assert stored_chat.role == "assistant"
    assert stored_chat.tool_calls_json is not None


def test_move_article_defaults_are_zero(session: Session) -> None:
    move = _move()
    article = Article(
        url="https://news.google.com/rss/articles/no-components",
        title="Chip stocks slip",
        news_source="gdelt",
    )
    session.add_all([_price(), move, article])
    session.commit()
    assert move.id is not None and article.id is not None

    session.add(
        MoveArticle(
            move_id=move.id,
            article_id=article.id,
            relevance=0.0,
            category="industry",
            provider="heuristic",
        )
    )
    session.commit()

    stored = session.exec(select(MoveArticle)).one()
    assert stored.bucket_match == 0.0
    assert stored.coverage == 0.0
    assert stored.timing_kind is None
    assert stored.model_score is None
    assert stored.geo_gate is None


def test_explanation_cited_ids_default(session: Session) -> None:
    move = _move()
    session.add_all([_price(), move])
    session.commit()
    assert move.id is not None

    session.add(
        Explanation(
            move_id=move.id,
            summary="No article in the window explains this move.",
            primary_category="unexplained",
            confidence=0.1,
            unexplained=True,
            provider="heuristic",
        )
    )
    session.commit()

    stored = session.exec(select(Explanation)).one()
    assert stored.unexplained is True
    assert stored.cited_article_ids == []


def test_duplicate_price_day_violates_unique_constraint(session: Session) -> None:
    session.add(_price())
    session.commit()

    session.add(_price())
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_duplicate_move_day_violates_unique_constraint(session: Session) -> None:
    session.add(_move())
    session.commit()

    session.add(_move())
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_duplicate_article_url_violates_unique_constraint(session: Session) -> None:
    url = "https://news.google.com/rss/articles/dupe"
    session.add(Article(url=url, title="First", news_source="google_rss"))
    session.commit()

    session.add(Article(url=url, title="Same url, different title", news_source="gdelt"))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_v15_tables_are_created(session: Session) -> None:
    """`create_all` builds the v1.5 tables alongside the v1 ones."""
    bind = session.get_bind()
    names = set(inspect(bind).get_table_names())
    assert {"company_edges", "geo_events"} <= names
    assert {
        "companies",
        "prices",
        "moves",
        "articles",
        "move_articles",
        "explanations",
        "chat_messages",
    } <= names


def test_edge_relation_and_source_vocabularies() -> None:
    assert EDGE_RELATIONS == ("competitor", "supplier", "customer", "country", "factor")
    assert EDGE_SOURCES == ("model", "etf_holdings", "prices")


def test_company_edge_round_trips(session: Session) -> None:
    session.add(_company())
    session.add(
        CompanyEdge(src="NVDA", dst="AMD", relation="competitor", weight=0.8, source="model")
    )
    session.add(CompanyEdge(src="NVDA", dst="TW", relation="country", weight=0.9, source="model"))
    session.add(
        CompanyEdge(src="NVDA", dst="oil", relation="factor", weight=-0.12, source="prices")
    )
    session.commit()

    stored = session.exec(
        select(CompanyEdge).where(CompanyEdge.src == "NVDA", CompanyEdge.relation == "competitor")
    ).one()
    assert stored.dst == "AMD"
    assert stored.weight == pytest.approx(0.8)
    assert stored.source == "model"
    assert stored.relation in EDGE_RELATIONS
    assert stored.source in EDGE_SOURCES
    assert isinstance(stored.updated_at, datetime)

    factor = session.exec(select(CompanyEdge).where(CompanyEdge.relation == "factor")).one()
    assert (factor.dst, factor.weight) == ("oil", pytest.approx(-0.12))

    assert len(session.exec(select(CompanyEdge).where(CompanyEdge.src == "NVDA")).all()) == 3


def test_company_edge_weight_defaults_to_one(session: Session) -> None:
    session.add(CompanyEdge(src="NVDA", dst="AVGO", relation="competitor", source="etf_holdings"))
    session.commit()

    assert session.exec(select(CompanyEdge)).one().weight == pytest.approx(1.0)


def test_company_edge_key_is_src_dst_relation(session: Session) -> None:
    """The same pair may carry two relations; the same triple may not repeat."""
    session.add(CompanyEdge(src="AAPL", dst="TSM", relation="supplier", source="model"))
    session.add(CompanyEdge(src="AAPL", dst="TSM", relation="customer", source="model"))
    session.commit()
    assert len(session.exec(select(CompanyEdge)).all()) == 2

    session.add(CompanyEdge(src="AAPL", dst="TSM", relation="supplier", source="etf_holdings"))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_geo_event_round_trips_with_titles(session: Session) -> None:
    session.add(
        GeoEvent(
            date=MOVE_DATE,
            country="TW",
            headline_count=7,
            sample_titles_json=json.dumps(["Taiwan export controls widen", "Chip curbs tighten"]),
            news_source="google_rss",
        )
    )
    session.commit()

    stored = session.exec(select(GeoEvent).where(GeoEvent.country == "TW")).one()
    assert stored.id is not None
    assert stored.date == MOVE_DATE
    assert stored.headline_count == 7
    assert stored.sample_titles == ["Taiwan export controls widen", "Chip curbs tighten"]
    assert stored.news_source == "google_rss"
    assert isinstance(stored.fetched_at, datetime)


def test_geo_event_titles_default_and_malformed_json() -> None:
    def event(titles_json: str | None = None) -> GeoEvent:
        kwargs = {} if titles_json is None else {"sample_titles_json": titles_json}
        return GeoEvent(
            date=MOVE_DATE, country="CN", headline_count=0, news_source="gdelt", **kwargs
        )

    assert event().sample_titles == []
    assert event("not json").sample_titles == []
    assert event('{"a": 1}').sample_titles == []


def test_duplicate_geo_event_violates_unique_constraint(session: Session) -> None:
    def event(news_source: str = "google_rss", headline_count: int = 4) -> GeoEvent:
        return GeoEvent(
            date=MOVE_DATE,
            country="CN",
            headline_count=headline_count,
            news_source=news_source,
        )

    session.add(event())
    session.commit()

    # A second count for the same day, country and source is the duplicate.
    session.add(event(headline_count=9))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    # The other source is a different row, not a duplicate.
    session.add(event(news_source="gdelt"))
    session.commit()
    assert len(session.exec(select(GeoEvent)).all()) == 2


def test_move_v15_fields_default_to_none(session: Session) -> None:
    """A v1-shaped `Move` still constructs; every v1.5 column defaults to None."""
    move = Move(
        ticker="AMD",
        date=MOVE_DATE,
        ret=-0.06,
        ret_z=-2.4,
        routing="industry",
        direction="down",
    )
    session.add(move)
    session.commit()

    stored = session.exec(select(Move).where(Move.ticker == "AMD")).one()
    assert stored.sub_routing is None
    assert stored.macro_driver is None
    assert stored.macro_driver_component is None
    assert stored.rival_comove is None
    assert stored.chain_comove is None
    assert stored.peer_comove is None


def test_price_macro_driver_defaults_to_none(session: Session) -> None:
    session.add(
        Price(
            ticker="AMD",
            date=MOVE_DATE,
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=1.0,
        )
    )
    session.commit()

    stored = session.exec(select(Price).where(Price.ticker == "AMD")).one()
    assert stored.macro_driver is None
    assert stored.macro_driver_component is None


def test_move_article_geo_gate_round_trips(session: Session) -> None:
    move = _move()
    article = Article(
        url="https://news.google.com/rss/articles/geo",
        title="China widens export controls",
        news_source="google_rss",
    )
    session.add_all([move, article])
    session.commit()
    assert move.id is not None and article.id is not None

    session.add(
        MoveArticle(
            move_id=move.id,
            article_id=article.id,
            relevance=0.30,
            category="macro",
            provider="heuristic",
            geo_gate=0.0,
        )
    )
    session.commit()

    assert session.exec(select(MoveArticle)).one().geo_gate == pytest.approx(0.0)


def test_session_scope_commits_and_rolls_back(session: Session) -> None:
    """`session_scope` uses the engine the fixture configured, so it sees the same rows."""
    with session_scope() as scoped:
        scoped.add(Company(ticker="AMD", name="Advanced Micro Devices"))

    with pytest.raises(RuntimeError), session_scope() as scoped:
        scoped.add(Company(ticker="INTC", name="Intel"))
        raise RuntimeError("boom")

    tickers = sorted(session.exec(select(Company.ticker)).all())
    assert tickers == ["AMD"]


def test_settings_defaults(no_api_key: None, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ANTHROPIC_MODEL", "NEWS_SOURCE", "DB_PATH"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.anthropic_api_key is None
    assert settings.anthropic_model == "claude-sonnet-5"
    assert settings.news_source == "google_rss"
    assert settings.db_path == Path("data/app.db")
    assert settings.default_period == "2y"
    assert settings.default_top_n == 10
    assert settings.default_z_threshold == pytest.approx(2.0)
    assert settings.default_pct_threshold == pytest.approx(0.02)
    assert settings.top_k_articles == 8
    assert settings.news_limit == 30
    assert settings.gdelt_throttle_s == pytest.approx(5.0)
    assert settings.http_timeout_s == pytest.approx(20.0)


def test_settings_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_PATH", "/tmp/other.db")
    monkeypatch.setenv("NEWS_SOURCE", "gdelt")
    monkeypatch.setenv("DEFAULT_TOP_N", "3")
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.db_path == Path("/tmp/other.db")
    assert settings.news_source == "gdelt"
    assert settings.default_top_n == 3

    get_settings.cache_clear()


def test_settings_empty_string_counts_as_unset(
    no_api_key: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_MODEL", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.anthropic_model == "claude-sonnet-5"
    assert settings.anthropic_api_key is None


def test_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    first = get_settings()
    monkeypatch.setenv("NEWS_SOURCE", "gdelt")
    assert get_settings() is first

    get_settings.cache_clear()
    assert get_settings().news_source == "gdelt"
    get_settings.cache_clear()
