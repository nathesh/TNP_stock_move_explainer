"""Ingest tests — the whole pipeline, with no network anywhere.

`prices` is monkeypatched at the module level (which is why `ingest.py` calls
`prices.fetch_ohlcv(...)` rather than binding the function), the news source is
a fake that records its queries, and the provider is the keyless heuristic. So
these tests exercise the real feature, detection, scoring and explanation code
against a synthetic price frame with a shock planted at a known position.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from sqlmodel import Session, col, select
from synth import synthetic_ohlcv

from stock_moves import ingest, prices
from stock_moves.models import Article, Explanation, Move, MoveArticle, Price
from stock_moves.moves import FEATURE_COLUMNS
from stock_moves.news import NewsItem
from stock_moves.ontology import get_or_build_company
from stock_moves.prices import TickerInfo
from stock_moves.providers import HeuristicProvider
from stock_moves.providers.base import ExplanationResult

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


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every `prices` entry point replaced by a deterministic local stand-in."""

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
        lambda peers, period="1y": pd.DataFrame(index=pd.DatetimeIndex([], name="date")),
    )
    monkeypatch.setattr(prices, "fetch_etf_holdings", lambda etf, top_n=10: [])
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
    """Every `FEATURE_COLUMNS` entry is in exactly one of ingest's groups."""
    groups = ingest._FLOAT_FEATURES + ingest._STR_FEATURES + ingest._BOOL_FEATURES
    assert set(groups) == set(FEATURE_COLUMNS)
    assert len(groups) == len(FEATURE_COLUMNS)


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
