"""Ingest tests — the whole pipeline, with no network anywhere.

`prices` is monkeypatched at the module level (which is why `ingest.py` calls
`prices.fetch_ohlcv(...)` rather than binding the function), the news source is
a fake that records its queries, and the provider is the keyless heuristic. So
these tests exercise the real feature, detection, scoring and explanation code
against a synthetic price frame with a shock planted at a known position.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from sqlmodel import Session, col, select
from synth import synthetic_market, synthetic_ohlcv

from stock_moves import ingest, prices
from stock_moves.models import (
    Article,
    CompanyEdge,
    Explanation,
    GeoEvent,
    Move,
    MoveArticle,
    Price,
)
from stock_moves.moves import FACTOR_COLUMNS, FEATURE_COLUMNS
from stock_moves.news import NewsItem
from stock_moves.ontology import get_or_build_company
from stock_moves.prices import TickerInfo
from stock_moves.providers import HeuristicProvider
from stock_moves.providers.base import ExplanationResult, MoveContext, Relations
from stock_moves.scoring import score_and_link as real_score_and_link

TICKER = "TEST"
COMPANY_NAME = "Test Corp"
PERIOD = "1y"
TOP_N = 3

#: A fixed seed per ticker rather than `hash(ticker)`, which is salted per
#: process and would make the frames differ between runs.
SEEDS: dict[str, int] = {"TEST": 1, "SPY": 2, "XLK": 3}

#: Index position of the planted shock, well past the 60-day OLS warm-up.
SHOCK_POSITION = 250
SHOCK_RETURN = -0.08

#: The typed facts the stub provider states (v1.5 decision 3). Without a key
#: the heuristic provider states none of this, so every edge test needs a stub.
RIVALS: tuple[str, ...] = ("AMD", "INTC")
SUPPLIERS: tuple[str, ...] = ("TSM",)
CUSTOMERS: tuple[str, ...] = ("DELL",)
COUNTRY = "TW"
COUNTRY_WEIGHT = 0.6
COUNTRY_FACTOR = "country:TW"

#: What the related names did on the shock day in the share-shift fixture: up,
#: while the stock fell 8%, which is `sub_route`'s share_shift rule.
RIVAL_SHOCK_RETURN = 0.03

#: Seeds for the four macro factor proxies, so their returns are deterministic
#: and independent of the stock — the country proxy is the one that matters.
FACTOR_SEEDS: dict[str, int] = {"oil": 11, "dollar": 12, "rates": 13, "gold": 14}

COMPANY_HEADLINE = "Test Corp shares plunge after a guidance cut"
MACRO_HEADLINE = "Broad market selloff as Treasury yields jump"

_EASTERN = ZoneInfo("America/New_York")


def frame_for(ticker: str) -> pd.DataFrame:
    """The synthetic OHLCV frame this test uses for one symbol."""
    return synthetic_ohlcv(
        seed=SEEDS.get(ticker.upper(), 0),
        shocks={SHOCK_POSITION: SHOCK_RETURN} if ticker.upper() == TICKER else None,
    )


@dataclass
class FakeNewsSource:
    """A `NewsSource` that records its queries and never touches the network.

    Two items per query, deduped on url across the buckets of one move, so a
    move ends up with two articles: one naming the company (which the heuristic
    scores as `company` at high relevance) and one macro headline that must not
    be mistaken for company news.
    """

    name: str = "fake"
    queries: list[str] = field(default_factory=list)
    calls: int = 0

    def search(self, query: str, start: date, end: date, limit: int = 30) -> list[NewsItem]:
        self.queries.append(query)
        self.calls += 1
        # `window_for` is ±1 day, so start+1 is the move date itself: the
        # article is then timed as a "cause" rather than a follow-up report.
        published = datetime.combine(start + timedelta(days=1), time(13, 0))
        day = published.date().isoformat()
        return [
            NewsItem(
                url=f"https://news.example/{day}/test-corp",
                title=COMPANY_HEADLINE,
                source="Reuters",
                published_at=published,
                language="en",
                news_source=self.name,
                bucket="",
            ),
            NewsItem(
                url=f"https://news.example/{day}/macro",
                title=MACRO_HEADLINE,
                source="Reuters",
                published_at=published,
                language="en",
                news_source=self.name,
                bucket="",
            ),
        ]


def empty_returns() -> pd.DataFrame:
    """The "nothing resolved" frame both `prices` fetchers return on failure."""
    return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))


def stock_returns() -> pd.Series:
    """Close-to-close returns of the ticker's synthetic frame, NaN filled."""
    return frame_for(TICKER)["close"].pct_change().fillna(0.0)


def correlated_market() -> pd.DataFrame:
    """An SPY frame that all but *is* the stock, so the decomposition routes macro.

    The market leg of the OLS then explains the move and `route` answers
    `macro`, which is the only routing with a macro sub-bucket — the one that
    can carry a `country:XX` and therefore reach the geo layer at all.
    """
    return synthetic_market(
        len(frame_for(TICKER).index), seed=SEEDS["SPY"], correlate_with=stock_returns(), beta=0.95
    )


def factor_frame(names: dict[str, str]) -> pd.DataFrame:
    """The stub factor-return frame: four independent proxies and one that isn't.

    The country proxy is handed the stock's own returns, so its trailing beta
    fits at about 1 and its contribution on the shock day is the whole move —
    which makes `country:TW` the `macro_driver` there without the test having
    to assert anything about the other three. The macro proxies are independent
    random walks, so their betas are noise and their contributions are small.
    """
    index = frame_for(TICKER).index
    columns: dict[str, pd.Series] = {}
    for key in names:
        column = prices.factor_column_for(key)
        if column == COUNTRY_FACTOR:
            columns[column] = stock_returns()
        else:
            seed = FACTOR_SEEDS.get(column, 0)
            columns[column] = synthetic_ohlcv(seed=seed)["close"].pct_change().fillna(0.0)
    return pd.DataFrame(columns, index=index)


def related_frame(tickers: Sequence[str]) -> pd.DataFrame:
    """Flat returns for each related name, except `RIVAL_SHOCK_RETURN` on the shock.

    Flat everywhere else so exactly one move has a co-movement story and the
    rest keep `sub_routing` None.
    """
    index = frame_for(TICKER).index
    frame = pd.DataFrame(0.0, index=index, columns=list(tickers))
    if len(frame.columns):
        frame.loc[pd.Timestamp(shock_date())] = RIVAL_SHOCK_RETURN
    return frame


class RelationsProvider(HeuristicProvider):
    """The keyless provider with one model-shaped answer bolted on.

    `HeuristicProvider.suggest_relations` returns all four relations empty by
    design (decision 3), so without this stub there are no competitor edges, no
    country edges, and the geo half of v1.5 is unreachable in a test.
    """

    def suggest_relations(
        self, ticker: str, name: str, sector: str | None, industry: str | None
    ) -> Relations:
        return Relations(
            competitors=RIVALS,
            suppliers=SUPPLIERS,
            customers=CUSTOMERS,
            countries=((COUNTRY, COUNTRY_WEIGHT),),
        )


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every `prices` entry point replaced by a deterministic local stand-in.

    `fetch_factor_returns` returns nothing here, which is the v1.5 path where
    no proxy resolved: the factor pass is skipped, no `factor` edge is written
    and `build_features` is called exactly as v1 called it. The fixtures below
    are the ones that hand it a frame.
    """

    def fetch_ohlcv(ticker: str, period: str = "1y") -> pd.DataFrame:
        return frame_for(ticker)

    monkeypatch.setattr(prices, "fetch_ohlcv", fetch_ohlcv)
    monkeypatch.setattr(
        prices,
        "fetch_info",
        lambda ticker: TickerInfo(TICKER, COMPANY_NAME, "Technology", "Semiconductors", "XLK"),
    )
    monkeypatch.setattr(prices, "fetch_earnings_dates", lambda ticker, limit=12: [])
    monkeypatch.setattr(
        prices,
        "fetch_peer_returns",
        lambda peers, period="1y": empty_returns(),
    )
    monkeypatch.setattr(prices, "fetch_etf_holdings", lambda etf, top_n=10: [])
    monkeypatch.setattr(
        prices,
        "fetch_factor_returns",
        lambda names, period="1y": empty_returns(),
    )
    yield


@pytest.fixture
def relations_provider() -> RelationsProvider:
    """The keyless provider plus one stubbed `suggest_relations` answer."""
    return RelationsProvider()


@pytest.fixture
def share_shift_network(no_network: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """`no_network`, with the related names up on the day the stock fell 8%."""
    monkeypatch.setattr(
        prices,
        "fetch_peer_returns",
        lambda tickers, period="1y": related_frame(list(tickers)),
    )
    yield


@pytest.fixture
def country_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A macro-routed tape with a country factor: the whole v1.5 path, no network.

    The market frame is correlated with the stock so `routing` is `macro`, and
    the country proxy carries the stock's own returns so `macro_driver` is
    `country:TW`. That is what gives the run moves with a `country:XX`
    `sub_routing` — and therefore geo events, and an open gate.
    """

    def fetch_ohlcv(ticker: str, period: str = "1y") -> pd.DataFrame:
        if ticker.upper() == "SPY":
            return correlated_market()
        return frame_for(ticker)

    monkeypatch.setattr(prices, "fetch_ohlcv", fetch_ohlcv)
    monkeypatch.setattr(
        prices,
        "fetch_info",
        lambda ticker: TickerInfo(TICKER, COMPANY_NAME, "Technology", "Semiconductors", "XLK"),
    )
    monkeypatch.setattr(prices, "fetch_earnings_dates", lambda ticker, limit=12: [])
    monkeypatch.setattr(prices, "fetch_etf_holdings", lambda etf, top_n=10: [])
    monkeypatch.setattr(
        prices,
        "fetch_peer_returns",
        lambda tickers, period="1y": related_frame(list(tickers)),
    )
    monkeypatch.setattr(
        prices,
        "fetch_factor_returns",
        lambda names, period="1y": factor_frame(dict(names)),
    )
    yield


@pytest.fixture
def news() -> FakeNewsSource:
    return FakeNewsSource()


@pytest.fixture
def provider() -> HeuristicProvider:
    return HeuristicProvider()


def run_ingest(
    session: Session,
    news: FakeNewsSource,
    provider: HeuristicProvider,
    *,
    refresh: bool = False,
) -> ingest.IngestResult:
    return ingest.ingest_ticker(
        session,
        TICKER,
        period=PERIOD,
        top_n=TOP_N,
        refresh=refresh,
        provider=provider,
        news_source=news,
    )


def shock_date() -> date:
    """The calendar date of the planted shock."""
    return frame_for(TICKER).index[SHOCK_POSITION].date()


def joined_row_count() -> int:
    """Rows surviving the stock/market/ETF inner join in `build_features`."""
    index = frame_for(TICKER).index.intersection(frame_for("SPY").index)
    return len(index.intersection(frame_for("XLK").index))


def count(session: Session, model: type, ticker: str | None = None) -> int:
    statement = select(model)
    if ticker is not None:
        statement = statement.where(col(model.ticker) == ticker)
    return len(session.exec(statement).all())


# --------------------------------------------------------------------------- #
# The feature-column contract this module relies on
# --------------------------------------------------------------------------- #


def test_every_feature_column_is_stored() -> None:
    """Every stored column is in exactly one of ingest's groups.

    v1.5 widens the contract from `FEATURE_COLUMNS` to `FEATURE_COLUMNS` plus
    `FACTOR_COLUMNS`: `build_features` always returns both blocks, both are
    columns on `prices` and on `moves`, and a column added to either tuple must
    still land in exactly one group here.
    """
    groups = ingest._FLOAT_FEATURES + ingest._STR_FEATURES + ingest._BOOL_FEATURES
    stored = FEATURE_COLUMNS + FACTOR_COLUMNS
    assert set(groups) == set(stored)
    assert len(groups) == len(stored)


# --------------------------------------------------------------------------- #
# A first ingest
# --------------------------------------------------------------------------- #


def test_first_ingest_stores_prices_moves_and_top_n_explanations(
    session: Session,
    no_network: None,
    news: FakeNewsSource,
    provider: HeuristicProvider,
) -> None:
    result = run_ingest(session, news, provider)

    assert result.ticker == TICKER
    assert result.period == PERIOD
    assert result.top_n == TOP_N
    assert result.provider == "heuristic"
    assert result.news_source == "fake"

    assert result.n_prices == joined_row_count()
    assert result.n_prices == count(session, Price, TICKER)
    assert result.n_moves >= 1
    assert result.n_moves == count(session, Move, TICKER)

    # The planted shock is a move, and the biggest one on record.
    moves = session.exec(
        select(Move).where(col(Move.ticker) == TICKER).order_by(col(Move.date).asc())
    ).all()
    dates = {move.date for move in moves}
    assert shock_date() in dates

    # Exactly min(top_n, n_moves) explanations, one per move.
    expected = min(TOP_N, result.n_moves)
    explanations = session.exec(select(Explanation)).all()
    assert len(explanations) == expected
    assert result.n_explanations == expected
    assert len({e.move_id for e in explanations}) == expected

    # The company query ran, anchored to the market as DESIGN section 3 says.
    assert news.queries
    assert any(query.startswith('"Test" stock') for query in news.queries)


def test_shock_move_explanation_cites_the_company_article(
    session: Session,
    no_network: None,
    news: FakeNewsSource,
    provider: HeuristicProvider,
) -> None:
    run_ingest(session, news, provider)

    move = session.exec(
        select(Move).where(col(Move.ticker) == TICKER, col(Move.date) == shock_date())
    ).one()
    explanation = session.exec(select(Explanation).where(col(Explanation.move_id) == move.id)).one()
    company_article = session.exec(
        select(Article).where(col(Article.title) == COMPANY_HEADLINE)
    ).first()

    assert company_article is not None
    assert company_article.id in explanation.cited_article_ids
    assert not explanation.unexplained
    assert COMPANY_HEADLINE in explanation.summary


# --------------------------------------------------------------------------- #
# The reported provider
# --------------------------------------------------------------------------- #


@dataclass
class DegradingProvider:
    """A keyed-looking provider whose explanations all come from the fallback.

    Everything is delegated to the keyless provider; `explain` only re-stamps
    the result as `degraded`, which is exactly what `AnthropicProvider` does
    when a model call fails.
    """

    name: str = "stub"
    inner: HeuristicProvider = field(default_factory=HeuristicProvider)

    def score_articles(self, move: object, articles: object) -> object:
        return self.inner.score_articles(move, articles)  # type: ignore[arg-type]

    def explain(self, move: object, scored: object) -> ExplanationResult:
        result = self.inner.explain(move, scored)  # type: ignore[arg-type]
        return replace(result, degraded=True)

    def suggest_peers(
        self, ticker: str, name: str, sector: str | None, industry: str | None
    ) -> list[str]:
        return self.inner.suggest_peers(ticker, name, sector, industry)

    def chat(self, history: object, tools: object, ticker: str | None) -> object:
        return self.inner.chat(history, tools, ticker)  # type: ignore[arg-type]


def test_all_degraded_explanations_report_the_heuristic_provider(
    session: Session,
    no_network: None,
    news: FakeNewsSource,
) -> None:
    """Every explanation degraded, so the run reports "heuristic", not "stub"."""
    result = ingest.ingest_ticker(
        session,
        TICKER,
        period=PERIOD,
        top_n=TOP_N,
        provider=DegradingProvider(),
        news_source=news,
    )

    assert result.n_explanations >= 1
    stored = {e.provider for e in session.exec(select(Explanation)).all()}
    assert stored == {"heuristic"}
    assert result.provider == "heuristic"


# --------------------------------------------------------------------------- #
# Idempotence
# --------------------------------------------------------------------------- #


def test_second_ingest_is_idempotent_and_calls_no_news(
    session: Session,
    no_network: None,
    news: FakeNewsSource,
    provider: HeuristicProvider,
) -> None:
    first = run_ingest(session, news, provider)
    calls_after_first = news.calls
    assert calls_after_first > 0

    second = run_ingest(session, news, provider)

    assert second == first
    assert count(session, Price, TICKER) == first.n_prices
    assert count(session, Move, TICKER) == first.n_moves
    assert count(session, Article) == first.n_articles
    assert len(session.exec(select(Explanation)).all()) == first.n_explanations
    # Every top move was already explained, so the source was never asked again.
    assert news.calls == calls_after_first


def test_refresh_refetches_news_and_keeps_one_explanation_per_move(
    session: Session,
    no_network: None,
    news: FakeNewsSource,
    provider: HeuristicProvider,
) -> None:
    first = run_ingest(session, news, provider)
    calls_after_first = news.calls

    refreshed = run_ingest(session, news, provider, refresh=True)

    assert news.calls > calls_after_first
    assert refreshed.n_prices == first.n_prices
    assert refreshed.n_moves == first.n_moves
    # Deduped on url, so re-fetching the same headlines adds no rows.
    assert refreshed.n_articles == first.n_articles

    explanations = session.exec(select(Explanation)).all()
    assert len(explanations) == first.n_explanations
    assert len({e.move_id for e in explanations}) == len(explanations)


# --------------------------------------------------------------------------- #
# On-demand enrichment
# --------------------------------------------------------------------------- #


def test_enrich_move_explains_a_move_outside_the_top_n(
    session: Session,
    no_network: None,
    news: FakeNewsSource,
    provider: HeuristicProvider,
) -> None:
    run_ingest(session, news, provider)

    explained = {e.move_id for e in session.exec(select(Explanation)).all()}
    unexplained = session.exec(select(Move).where(col(Move.ticker) == TICKER)).all()
    target = next(move for move in unexplained if move.id not in explained)
    company = get_or_build_company(session, TICKER, provider)

    calls_before = news.calls
    explanation = ingest.enrich_move(session, target, company, provider, news)

    assert explanation.move_id == target.id
    assert news.calls > calls_before
    assert session.exec(select(MoveArticle).where(col(MoveArticle.move_id) == target.id)).all()
    assert len(session.exec(select(Explanation)).all()) == TOP_N + 1

    # A second call is served from the cache.
    calls_after = news.calls
    again = ingest.enrich_move(session, target, company, provider, news)
    assert again.id == explanation.id
    assert news.calls == calls_after


# --------------------------------------------------------------------------- #
# Freshness (DESIGN section 5)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        # Saturday -> the Friday before it.
        (datetime(2026, 9, 12, 12, 0, tzinfo=_EASTERN), date(2026, 9, 11)),
        # Sunday -> the same Friday.
        (datetime(2026, 9, 13, 12, 0, tzinfo=_EASTERN), date(2026, 9, 11)),
        # Monday before the close -> the Friday before it.
        (datetime(2026, 9, 14, 10, 0, tzinfo=_EASTERN), date(2026, 9, 11)),
        # Monday after the close -> that Monday.
        (datetime(2026, 9, 14, 17, 0, tzinfo=_EASTERN), date(2026, 9, 14)),
        # Exactly at the close counts as completed.
        (datetime(2026, 9, 14, 16, 0, tzinfo=_EASTERN), date(2026, 9, 14)),
    ],
)
def test_last_completed_trading_day(now: datetime, expected: date) -> None:
    assert ingest.last_completed_trading_day(now) == expected


def test_last_completed_trading_day_accepts_naive_and_other_zones() -> None:
    """A naive value is read as ET; an aware one is converted to ET first."""
    naive_monday_morning = datetime(2026, 9, 14, 10, 0)  # noqa: DTZ001
    assert ingest.last_completed_trading_day(naive_monday_morning) == date(2026, 9, 11)

    # 22:00 UTC on Monday is 18:00 ET, i.e. after that day's close.
    utc_monday_evening = datetime(2026, 9, 14, 22, 0, tzinfo=ZoneInfo("UTC"))
    assert ingest.last_completed_trading_day(utc_monday_evening) == date(2026, 9, 14)


def test_needs_ingest_and_is_stale(
    session: Session,
    no_network: None,
    news: FakeNewsSource,
    provider: HeuristicProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert ingest.needs_ingest(session, TICKER) is True
    assert ingest.is_stale(session, TICKER) is True

    run_ingest(session, news, provider)

    assert ingest.needs_ingest(session, TICKER) is False

    # The synthetic history ends in the past, so pin "the last completed
    # trading day" to a day the stored data covers.
    newest = frame_for(TICKER).index[-1].date()
    monkeypatch.setattr(ingest, "last_completed_trading_day", lambda now=None: newest)
    assert ingest.is_stale(session, TICKER) is False

    # One trading day later and the ticker is stale again.
    monkeypatch.setattr(
        ingest,
        "last_completed_trading_day",
        lambda now=None: newest + timedelta(days=3),
    )
    assert ingest.is_stale(session, TICKER) is True

    # An unknown ticker is always stale: there is nothing stored at all.
    assert ingest.is_stale(session, "NOPE") is True
    assert ingest.needs_ingest(session, "NOPE") is True


# --------------------------------------------------------------------------- #
# v1.5: the relationship layer (docs/v1.5-plan.md, decisions 4 to 8)
# --------------------------------------------------------------------------- #


def edges(session: Session, relation: str | None = None) -> list[CompanyEdge]:
    """Stored edges for the test ticker, optionally one relation."""
    statement = select(CompanyEdge).where(col(CompanyEdge.src) == TICKER)
    if relation is not None:
        statement = statement.where(col(CompanyEdge.relation) == relation)
    return list(session.exec(statement).all())


def moves_of(session: Session) -> list[Move]:
    return list(session.exec(select(Move).where(col(Move.ticker) == TICKER)).all())


def shock_move(session: Session) -> Move:
    return session.exec(
        select(Move).where(col(Move.ticker) == TICKER, col(Move.date) == shock_date())
    ).one()


def test_no_factor_frame_skips_the_factor_pass_but_still_writes_edges(
    session: Session,
    no_network: None,
    news: FakeNewsSource,
    relations_provider: RelationsProvider,
) -> None:
    """An empty factor frame means no attribution, no factor edge, no geo.

    This is the keyless-and-offline shape of the run: `fetch_factor_returns`
    resolved nothing, so `build_features` is called the v1 way and every
    `macro_driver` is NULL. The model relations are still written — they never
    depended on prices — which is what keeps the two `build_edges` calls
    independently useful.
    """
    result = ingest.ingest_ticker(
        session,
        TICKER,
        period=PERIOD,
        top_n=TOP_N,
        provider=relations_provider,
        news_source=news,
    )

    assert edges(session, "factor") == []
    assert {edge.dst for edge in edges(session, "competitor")} == set(RIVALS)
    assert {edge.dst for edge in edges(session, "supplier")} == set(SUPPLIERS)
    assert {edge.dst for edge in edges(session, "customer")} == set(CUSTOMERS)
    assert [(edge.dst, edge.weight) for edge in edges(session, "country")] == [
        (COUNTRY, COUNTRY_WEIGHT)
    ]
    assert result.n_edges == len(edges(session))

    stored = session.exec(select(Price).where(col(Price.ticker) == TICKER)).all()
    assert stored
    assert all(price.macro_driver is None for price in stored)
    assert all(price.macro_driver_component is None for price in stored)

    # No macro driver and no co-movement: nothing for `sub_route` to name.
    assert all(move.sub_routing is None for move in moves_of(session))
    # No country-routed move, so no geo query was ever issued.
    assert result.n_geo_events == 0
    assert session.exec(select(GeoEvent)).all() == []
    assert not any("Taiwan" in query for query in news.queries)


def test_country_run_stores_factor_columns_sub_routing_and_geo_events(
    session: Session,
    country_network: None,
    news: FakeNewsSource,
    relations_provider: RelationsProvider,
) -> None:
    """One synthetic run through the whole v1.5 path.

    The tape is macro-routed and the country proxy is what moved, so the shock
    carries `macro_driver` / `macro_driver_component` on both `prices` and
    `moves`, a `country:TW` `sub_routing`, a fitted `factor` edge per proxy, and
    a geopolitical headline count for the window.
    """
    result = ingest.ingest_ticker(
        session,
        TICKER,
        period=PERIOD,
        top_n=TOP_N,
        provider=relations_provider,
        news_source=news,
    )

    # `FACTOR_COLUMNS` on prices.
    price = session.exec(
        select(Price).where(col(Price.ticker) == TICKER, col(Price.date) == shock_date())
    ).one()
    assert price.macro_driver == COUNTRY_FACTOR
    assert price.macro_driver_component == pytest.approx(SHOCK_RETURN, abs=1e-6)

    # `sub_routing` and the attribution on the move.
    move = shock_move(session)
    assert move.routing == "macro"
    assert move.macro_driver == COUNTRY_FACTOR
    assert move.sub_routing == f"country:{COUNTRY}"
    assert move.rival_comove == pytest.approx(RIVAL_SHOCK_RETURN)
    assert move.chain_comove == pytest.approx(RIVAL_SHOCK_RETURN)
    # The v1 number is untouched beside them: no peers, so no peer co-movement.
    assert move.peer_comove is None

    # Factor edges, one per proxy, sourced from prices and not from the model.
    factor_edges = edges(session, "factor")
    assert {edge.dst for edge in factor_edges} == set(prices.FACTOR_ETFS) | {COUNTRY_FACTOR}
    assert {edge.source for edge in factor_edges} == {"prices"}
    # The country proxy carries the stock's own returns, so its beta is ~1.
    country_beta = next(edge.weight for edge in factor_edges if edge.dst == COUNTRY_FACTOR)
    assert country_beta == pytest.approx(1.0, abs=0.05)
    assert result.n_edges == len(edges(session))

    # Geo events: one row per country-routed move window, all for TW.
    country_moves = [
        move for move in moves_of(session) if (move.sub_routing or "").startswith("country:")
    ]
    assert len(country_moves) >= 1
    geo_rows = session.exec(select(GeoEvent)).all()
    assert {row.country for row in geo_rows} == {COUNTRY}
    assert {row.date for row in geo_rows} == {move.date for move in country_moves}
    assert result.n_geo_events == len(geo_rows)
    assert any("Taiwan" in query for query in news.queries)


def test_second_country_run_duplicates_no_edges_and_no_geo_events(
    session: Session,
    country_network: None,
    news: FakeNewsSource,
    relations_provider: RelationsProvider,
) -> None:
    """Re-ingesting rewrites the edges and the counts in place, never twice."""
    first = ingest.ingest_ticker(
        session, TICKER, period=PERIOD, top_n=TOP_N, provider=relations_provider, news_source=news
    )
    n_edges = len(edges(session))
    n_geo = len(session.exec(select(GeoEvent)).all())
    assert n_edges == first.n_edges
    assert n_geo == first.n_geo_events

    second = ingest.ingest_ticker(
        session, TICKER, period=PERIOD, top_n=TOP_N, provider=relations_provider, news_source=news
    )

    assert second == first
    assert len(edges(session)) == n_edges
    assert len(session.exec(select(GeoEvent)).all()) == n_geo


def test_score_and_link_is_given_the_countries_and_the_geo_lines(
    session: Session,
    country_network: None,
    news: FakeNewsSource,
    relations_provider: RelationsProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate rule needs the edges, and only `ingest` can read them for it."""
    seen: list[MoveContext] = []

    def recording_score_and_link(*args: object, **kwargs: object) -> object:
        context = kwargs.get("context")
        assert isinstance(context, MoveContext)
        seen.append(context)
        return real_score_and_link(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ingest, "score_and_link", recording_score_and_link)

    ingest.ingest_ticker(
        session,
        TICKER,
        period=PERIOD,
        top_n=TOP_N,
        provider=relations_provider,
        news_source=news,
    )

    assert len(seen) == TOP_N
    for context in seen:
        assert context.countries == ((COUNTRY, COUNTRY_WEIGHT),)
        assert context.competitors == RIVALS
        assert context.suppliers == SUPPLIERS
        assert context.customers == CUSTOMERS
    # The geo events are fetched before the explanations, so the window's
    # counts are already there to be rendered into the context.
    assert any(context.geo_events for context in seen)
    assert all(COUNTRY in line for context in seen for line in context.geo_events)


def test_share_shift_expands_the_queries_with_the_rival_names(
    session: Session,
    share_shift_network: None,
    news: FakeNewsSource,
    relations_provider: RelationsProvider,
) -> None:
    """Rivals up on the day the stock fell: `share_shift`, and one query each.

    The routing here is `company` (the market frame is independent of the
    stock), so this is the other half of decision 6 — and decision 8's rule
    that a rival query is only asked for the names that actually moved the
    other way.
    """
    ingest.ingest_ticker(
        session,
        TICKER,
        period=PERIOD,
        top_n=TOP_N,
        provider=relations_provider,
        news_source=news,
    )

    move = shock_move(session)
    assert move.routing == "company"
    assert move.sub_routing == "share_shift"
    assert move.rival_comove == pytest.approx(RIVAL_SHOCK_RETURN)

    for rival in RIVALS:
        assert f'"{rival}" stock' in news.queries
    # The other moves have flat rivals, so they get no expansion at all.
    assert sum(1 for move in moves_of(session) if move.sub_routing) == 1
