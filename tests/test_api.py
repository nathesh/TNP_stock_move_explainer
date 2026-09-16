"""End-to-end tests for the ticker routes (DESIGN sections 5 and 6).

The whole application is exercised through `TestClient` — routers,
dependencies, the read layer, detection, scoring and the explainer — with no
network anywhere. Three monkeypatches buy that:

* every `stock_moves.prices` entry point is replaced by a deterministic local
  stand-in (which is why `ingest` calls `prices.fetch_ohlcv(...)` through the
  module rather than binding the function), with a synthetic frame per symbol
  and an `-8%` shock planted at a known position;
* `stock_moves.ingest.get_news_source` returns a fake that counts its calls,
  which is how "the second read did not re-fetch news" is asserted rather than
  assumed;
* `stock_moves.ingest.last_completed_trading_day` is pinned to the end of the
  synthetic history, so the freshness rule of DESIGN section 5 considers the
  stored data current and the second read is a pure SQL read.

The database is in-memory and rebuilt per test, so each test tells the whole
story from a cold start.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import col, select
from synth import synthetic_ohlcv

from stock_moves import ingest, prices
from stock_moves.api import deps
from stock_moves.api.app import create_app
from stock_moves.db import Session, configure_engine, get_engine
from stock_moves.models import CompanyEdge, Move
from stock_moves.news import NewsItem
from stock_moves.prices import PriceFetchError, TickerInfo

TICKER = "TEST"
COMPANY_NAME = "Test Corp"

#: A fixed seed per symbol; `hash()` is salted per process and would make the
#: frames differ between runs. Anything outside this dict has no price data.
SEEDS: dict[str, int] = {"TEST": 1, "SPY": 2, "XLK": 3}

#: Index position of the planted shock, well past the 60-day OLS warm-up.
SHOCK_POSITION = 250
SHOCK_RETURN = -0.08

COMPANY_HEADLINE = "Test Corp shares plunge after a guidance cut"
MACRO_HEADLINE = "Broad market selloff as Treasury yields jump"

#: The edges seeded for the relations read: one of every relation the plan
#: defines, so the filter has something to exclude in each direction and the
#: ordering (relation, then dst) has two rows inside one relation to sort.
EDGE_ROWS: tuple[tuple[str, str, float, str], ...] = (
    ("AMD", "competitor", 0.8, "model"),
    ("INTC", "competitor", 0.4, "etf_holdings"),
    ("TSM", "supplier", 0.7, "model"),
    ("MSFT", "customer", 0.6, "model"),
    ("TW", "country", 0.3, "model"),
    ("CN", "country", 0.2, "model"),
    ("oil", "factor", -0.45, "prices"),
)

#: Every key `MoveOut` carried in v1; the v1.5 read must still carry them all.
V1_MOVE_KEYS = (
    "date",
    "ret",
    "ret_z",
    "gap_ret",
    "intraday_ret",
    "vol_z",
    "direction",
    "routing",
    "near_earnings",
    "near_fomc",
    "near_cpi",
    "regime_mkt",
    "regime_sector",
    "mkt_component",
    "sector_component",
    "idio_component",
    "peer_comove",
    "explanation",
    "articles",
)

#: The five v1.5 decision 10 adds to it.
V15_MOVE_KEYS = (
    "sub_routing",
    "macro_driver",
    "macro_driver_component",
    "rival_comove",
    "chain_comove",
)

#: Every column DESIGN section 1 says is computed and stored per trading day.
DESIGN_PRICE_COLUMNS = (
    "ret",
    "ret_z",
    "gap_ret",
    "intraday_ret",
    "vol_z",
    "mkt_component",
    "sector_component",
    "idio_component",
    "routing",
    "regime_mkt",
    "regime_sector",
    "near_earnings",
    "near_fomc",
    "near_cpi",
)


def frame_for(ticker: str) -> pd.DataFrame:
    """The synthetic OHLCV frame for one symbol, shock included for the stock."""
    key = ticker.upper()
    return synthetic_ohlcv(
        seed=SEEDS[key],
        shocks={SHOCK_POSITION: SHOCK_RETURN} if key == TICKER else None,
    )


def shock_date() -> date:
    """The calendar date of the planted shock."""
    return frame_for(TICKER).index[SHOCK_POSITION].date()


def last_synthetic_day() -> date:
    """The newest day the synthetic history covers."""
    return frame_for(TICKER).index[-1].date()


@dataclass
class FakeNewsSource:
    """A `NewsSource` that records its queries and never touches the network.

    Two items per query, deduped on url across the buckets of one move: one
    naming the company (the heuristic scores it `company` at high relevance)
    and one macro headline that must not be mistaken for company news.
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
def news() -> FakeNewsSource:
    return FakeNewsSource()


@pytest.fixture
def client(
    no_api_key: None, news: FakeNewsSource, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """A client on a fresh in-memory database, with no key and no network.

    `configure_engine` runs before `create_app`, so the lifespan's `init_db()`
    and every `get_db` session share this engine's single pooled connection.
    The provider cache is cleared on both sides because `no_api_key` changes
    which provider the key decides on (DESIGN section 4).
    """

    def fetch_ohlcv(ticker: str, period: str = "1y") -> pd.DataFrame:
        if ticker.upper() not in SEEDS:
            raise PriceFetchError(f"no price data for {ticker}")
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
    monkeypatch.setattr(
        ingest,
        "get_news_source",
        lambda name="google_rss", **kwargs: news,
    )
    # The synthetic history ends in the past; pin "the last completed trading
    # day" to a day it covers so the ticker is fresh once ingested.
    monkeypatch.setattr(ingest, "last_completed_trading_day", lambda now=None: last_synthetic_day())

    engine = configure_engine("sqlite://")
    deps.reset_provider_cache()
    with TestClient(create_app()) as test_client:
        yield test_client
    deps.reset_provider_cache()
    engine.dispose()


def read_ticker(client: TestClient, **params: Any) -> dict[str, Any]:
    """`GET /tickers/TEST` with `params`, asserting a 200."""
    response = client.get(f"/tickers/{TICKER}", params=params)
    assert response.status_code == 200, response.text
    payload: dict[str, Any] = response.json()
    return payload


def move_on(payload: dict[str, Any], on: date) -> dict[str, Any]:
    """The move for one date out of a ticker payload."""
    wanted = on.isoformat()
    return next(move for move in payload["moves"] if move["date"] == wanted)


def seed_edges() -> None:
    """Write `EDGE_ROWS` for the ticker on a fresh session.

    Edges are written by an ingest from a model call, and the keyless provider
    names none (v1.5 decision 3), so a test that wants edges to read has to put
    them there itself — which is also what keeps this an assertion about the
    route rather than about the provider.
    """
    with Session(get_engine()) as session:
        for dst, relation, weight, source in EDGE_ROWS:
            session.add(
                CompanyEdge(src=TICKER, dst=dst, relation=relation, weight=weight, source=source)
            )
        session.commit()


def set_sub_routing(assignments: dict[str, str]) -> None:
    """Stamp `sub_routing` onto stored moves, keyed by ISO date.

    Same reason as `seed_edges`: without a key the routing layer has no
    competitor or country edges to decide a sub-bucket from, so the filter is
    exercised against rows written here.
    """
    with Session(get_engine()) as session:
        statement = select(Move).where(col(Move.ticker) == TICKER)
        by_date = {row.date.isoformat(): row for row in session.exec(statement).all()}
        for iso, sub_routing in assignments.items():
            move = by_date[iso]
            move.sub_routing = sub_routing
            session.add(move)
        session.commit()


# --------------------------------------------------------------------------- #
# The lazy first read (DESIGN section 5)
# --------------------------------------------------------------------------- #


def test_first_read_ingests_and_explains_the_planted_shock(
    client: TestClient,
) -> None:
    payload = read_ticker(client)

    assert payload["ticker"] == TICKER
    assert payload["ingested"] is True
    assert payload["company"]["name"] == COMPANY_NAME
    assert payload["company"]["ticker"] == TICKER
    assert payload["moves"]
    # `prices` is opt-in, so a plain read does not carry a year of bars.
    assert payload["prices"] is None
    # The filters the server actually ran are echoed back.
    assert payload["filters"]["z_threshold"] == 2.0

    move = move_on(payload, shock_date())
    assert move["direction"] == "down"
    assert move["ret"] < 0

    explanation = move["explanation"]
    assert explanation is not None
    assert explanation["primary_category"] == "company"
    assert explanation["unexplained"] is False
    assert len(explanation["cited_article_ids"]) >= 1

    assert move["articles"]
    assert any(article["title"] == COMPANY_HEADLINE for article in move["articles"])
    cited = set(explanation["cited_article_ids"])
    assert cited & {article["id"] for article in move["articles"]}


def test_second_read_is_fresh_and_asks_the_news_source_nothing(
    client: TestClient, news: FakeNewsSource
) -> None:
    first = read_ticker(client)
    assert first["ingested"] is True
    calls_after_first = news.calls
    assert calls_after_first > 0

    second = read_ticker(client)

    assert second["ingested"] is False
    assert news.calls == calls_after_first
    assert len(second["moves"]) == len(first["moves"])


# --------------------------------------------------------------------------- #
# Filters (DESIGN section 6)
# --------------------------------------------------------------------------- #


def test_direction_filter_drops_the_other_side(client: TestClient) -> None:
    payload = read_ticker(client, direction="up")

    assert payload["moves"]
    assert all(move["direction"] == "up" for move in payload["moves"])


def test_impossible_thresholds_return_no_moves(client: TestClient) -> None:
    # Both gates have to be raised: DESIGN section 1 ORs them together.
    payload = read_ticker(client, z_threshold=100, pct_threshold=1)

    assert payload["moves"] == []
    assert payload["filters"]["z_threshold"] == 100.0


def test_include_prices_returns_every_stored_column(client: TestClient) -> None:
    payload = read_ticker(client, include_prices="true")

    rows = payload["prices"]
    assert rows is not None
    assert len(rows) > 200
    first = rows[0]
    for column in ("date", "open", "high", "low", "close", "volume"):
        assert column in first
    for column in DESIGN_PRICE_COLUMNS:
        assert column in first
    # Oldest first, which is the order a chart wants.
    assert rows[0]["date"] < rows[-1]["date"]


def test_include_news_false_omits_every_article(client: TestClient) -> None:
    payload = read_ticker(client, include_news="false")

    assert payload["moves"]
    assert all(move["articles"] == [] for move in payload["moves"])


def test_category_filter_falls_back_to_routing_when_unexplained(
    client: TestClient,
) -> None:
    payload = read_ticker(client, category="company")

    assert payload["moves"]
    for move in payload["moves"]:
        explanation = move["explanation"]
        effective = move["routing"] if explanation is None else explanation["primary_category"]
        assert effective == "company"


def test_limit_caps_the_move_list(client: TestClient) -> None:
    payload = read_ticker(client, limit=1)

    assert len(payload["moves"]) == 1


# --------------------------------------------------------------------------- #
# One move (DESIGN section 6), explained on demand (DESIGN section 4)
# --------------------------------------------------------------------------- #


def test_single_move_is_returned_with_its_explanation(client: TestClient) -> None:
    read_ticker(client)

    response = client.get(f"/tickers/{TICKER}/moves/{shock_date().isoformat()}")

    assert response.status_code == 200, response.text
    move = response.json()
    assert move["date"] == shock_date().isoformat()
    assert move["direction"] == "down"
    assert move["explanation"]["primary_category"] == "company"
    assert move["articles"]


def test_a_day_that_was_not_a_move_is_a_404(client: TestClient) -> None:
    payload = read_ticker(client, include_prices="true", limit=0)

    move_dates = {move["date"] for move in payload["moves"]}
    quiet_day = next(row["date"] for row in payload["prices"] if row["date"] not in move_dates)

    response = client.get(f"/tickers/{TICKER}/moves/{quiet_day}")

    assert response.status_code == 404
    assert quiet_day in response.json()["detail"]


def test_a_move_outside_the_top_n_is_explained_on_demand(
    client: TestClient, news: FakeNewsSource
) -> None:
    payload = read_ticker(client, limit=0)

    unexplained = [move for move in payload["moves"] if move["explanation"] is None]
    assert unexplained, "expected more moves than the default top_n"
    target = unexplained[0]["date"]
    calls_before = news.calls

    response = client.get(f"/tickers/{TICKER}/moves/{target}")

    assert response.status_code == 200, response.text
    move = response.json()
    assert move["explanation"] is not None
    assert move["explanation"]["summary"]
    # The explanation was computed now, not at ingest.
    assert news.calls > calls_before

    # And cached: the second read asks the news source nothing.
    calls_after = news.calls
    again = client.get(f"/tickers/{TICKER}/moves/{target}")
    assert again.status_code == 200
    assert again.json()["explanation"] == move["explanation"]
    assert news.calls == calls_after


# --------------------------------------------------------------------------- #
# Explicit ingest, unknown tickers, case
# --------------------------------------------------------------------------- #


def test_explicit_ingest_reports_what_it_stored(client: TestClient) -> None:
    response = client.post(f"/tickers/{TICKER}/ingest", params={"top_n": 1})

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["ticker"] == TICKER
    assert result["top_n"] == 1
    assert result["n_prices"] > 200
    assert result["n_moves"] >= 1
    assert result["n_explanations"] >= 1
    assert result["provider"] == "heuristic"
    assert result["news_source"] == "fake"


def test_unknown_ticker_is_a_404(client: TestClient) -> None:
    response = client.get("/tickers/NOPE")

    assert response.status_code == 404
    assert response.json()["detail"] == "unknown ticker or no price data"


def test_lower_case_ticker_is_upper_cased(client: TestClient) -> None:
    response = client.get(f"/tickers/{TICKER.lower()}")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ticker"] == TICKER
    assert payload["company"]["ticker"] == TICKER
    assert payload["moves"]


# --------------------------------------------------------------------------- #
# v1.5: the relations read, the sub-routing filter and the new counts
# --------------------------------------------------------------------------- #


def test_relations_returns_every_edge_in_a_stable_order(client: TestClient) -> None:
    read_ticker(client)
    seed_edges()

    response = client.get(f"/tickers/{TICKER}/relations")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ticker"] == TICKER
    # Ordered by relation, then dst — the order `edges_of` promises.
    assert [(edge["relation"], edge["dst"]) for edge in payload["edges"]] == [
        ("competitor", "AMD"),
        ("competitor", "INTC"),
        ("country", "CN"),
        ("country", "TW"),
        ("customer", "MSFT"),
        ("factor", "oil"),
        ("supplier", "TSM"),
    ]
    amd = payload["edges"][0]
    assert amd == {"dst": "AMD", "relation": "competitor", "weight": 0.8, "source": "model"}


def test_relations_filter_keeps_one_relation(client: TestClient) -> None:
    read_ticker(client)
    seed_edges()

    response = client.get(f"/tickers/{TICKER}/relations", params={"relation": "country"})

    assert response.status_code == 200, response.text
    edges = response.json()["edges"]
    assert [edge["dst"] for edge in edges] == ["CN", "TW"]
    assert {edge["relation"] for edge in edges} == {"country"}


def test_relations_of_a_stored_ticker_with_no_edges_is_an_empty_list(
    client: TestClient,
) -> None:
    """The keyless case, and it is a 200: the ticker is stored, and "no edges"
    is the honest answer rather than a missing resource."""
    read_ticker(client)

    response = client.get(f"/tickers/{TICKER}/relations")

    assert response.status_code == 200, response.text
    assert response.json() == {"ticker": TICKER, "edges": []}


def test_relations_for_an_unknown_ticker_is_a_404(client: TestClient) -> None:
    response = client.get("/tickers/NOPE/relations")

    assert response.status_code == 404
    assert "NOPE" in response.json()["detail"]


def test_relations_accepts_a_lower_case_ticker(client: TestClient) -> None:
    read_ticker(client)
    seed_edges()

    response = client.get(f"/tickers/{TICKER.lower()}/relations")

    assert response.status_code == 200, response.text
    assert response.json()["ticker"] == TICKER


def test_an_unknown_relation_is_rejected_rather_than_silently_empty(
    client: TestClient,
) -> None:
    read_ticker(client)

    response = client.get(f"/tickers/{TICKER}/relations", params={"relation": "rival"})

    assert response.status_code == 422


def test_sub_routing_filters_the_move_list(client: TestClient) -> None:
    payload = read_ticker(client)
    dates = [move["date"] for move in payload["moves"]]
    assert len(dates) >= 3
    taiwan, china, rivals = dates[0], dates[1], dates[2]
    set_sub_routing({taiwan: "country:TW", china: "country:CN", rivals: "share_shift"})

    exact = read_ticker(client, sub_routing="country:TW")
    assert [move["date"] for move in exact["moves"]] == [taiwan]
    assert exact["filters"]["sub_routing"] == "country:TW"

    # The one prefix match: bare "country" takes both codes and nothing else.
    both = read_ticker(client, sub_routing="country")
    assert sorted(move["date"] for move in both["moves"]) == sorted([taiwan, china])

    assert [move["date"] for move in read_ticker(client, sub_routing="share_shift")["moves"]] == [
        rivals
    ]
    assert read_ticker(client, sub_routing="supply_chain")["moves"] == []


def test_the_move_shape_is_a_superset_of_v1(client: TestClient) -> None:
    payload = read_ticker(client)
    move = move_on(payload, shock_date())

    for key in V1_MOVE_KEYS:
        assert key in move, key
    for key in V15_MOVE_KEYS:
        assert key in move, key
    # Unset without a key, and null rather than absent.
    assert move["sub_routing"] is None
    assert move["macro_driver"] is None


def test_the_single_move_read_carries_the_new_keys_too(client: TestClient) -> None:
    read_ticker(client)

    response = client.get(f"/tickers/{TICKER}/moves/{shock_date().isoformat()}")

    assert response.status_code == 200, response.text
    move = response.json()
    for key in (*V1_MOVE_KEYS, *V15_MOVE_KEYS):
        assert key in move, key


def test_ingest_reports_the_new_counts(client: TestClient) -> None:
    response = client.post(f"/tickers/{TICKER}/ingest", params={"top_n": 1})

    assert response.status_code == 200, response.text
    result = response.json()
    # Present and integral; zero here because the keyless provider names no
    # relations and the factor proxies have no synthetic prices.
    assert result["n_edges"] == 0
    assert result["n_geo_events"] == 0
    # And every v1 count is still there.
    for key in ("n_prices", "n_moves", "n_articles", "n_explanations"):
        assert key in result
