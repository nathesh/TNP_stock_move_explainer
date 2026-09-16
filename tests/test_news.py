"""Tests for the news sources and query builders. No network: every HTTP call
goes through an `httpx.MockTransport`."""

from __future__ import annotations

from datetime import UTC, date, datetime

import httpx
import pytest

from stock_moves.news import (
    MACRO_QUERY_TERMS,
    GDELTSource,
    GoogleNewsRSS,
    NewsItem,
    build_query,
    clean_company_name,
    get_news_source,
    parse_gdelt,
    parse_rss,
    queries_for_move,
    window_for,
)
from stock_moves.news import gdelt as gdelt_module
from stock_moves.news.base import MAX_EXTRA_QUERIES
from stock_moves.news.geo import geo_query

RSS_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>"Apple" stock - Google News</title>
    <item>
      <title>Apple slides 5% after guidance cut - Reuters</title>
      <link>https://news.google.com/rss/articles/AAA</link>
      <pubDate>Mon, 08 Sep 2025 13:45:00 GMT</pubDate>
      <source url="https://www.reuters.com">Reuters</source>
    </item>
    <item>
      <title>Analysts trim Apple price targets - CNBC</title>
      <link>https://news.google.com/rss/articles/BBB</link>
      <pubDate>Tue, 09 Sep 2025 02:10:00 GMT</pubDate>
      <source url="https://www.cnbc.com">CNBC</source>
    </item>
    <item>
      <title>Apple slides 5% after guidance cut - Reuters</title>
      <link>https://news.google.com/rss/articles/AAA</link>
      <pubDate>Mon, 08 Sep 2025 13:45:00 GMT</pubDate>
      <source url="https://www.reuters.com">Reuters</source>
    </item>
  </channel>
</rss>
"""

GDELT_FIXTURE: dict = {
    "articles": [
        {
            "url": "https://www.reuters.com/apple-guidance",
            "title": "Apple slides after guidance cut",
            "domain": "reuters.com",
            "seendate": "20250908T134500Z",
            "language": "English",
        },
        {
            "url": "https://www.cnbc.com/apple-targets",
            "title": "Analysts trim Apple targets",
            "domain": "cnbc.com",
            "seendate": "20250909T021000Z",
            "language": "English",
        },
        {
            "url": "https://www.reuters.com/apple-guidance",
            "title": "Apple slides after guidance cut",
            "domain": "reuters.com",
            "seendate": "20250908T134500Z",
            "language": "English",
        },
    ]
}


@pytest.fixture(autouse=True)
def _reset_gdelt_throttle() -> None:
    """Clear the module-level last-call stamp so tests do not sleep on each other."""
    gdelt_module._last_call_monotonic = None


# --- clean_company_name -----------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Apple Inc.", "Apple"),
        ("NVIDIA Corporation", "NVIDIA"),
        ("Meta Platforms, Inc.", "Meta Platforms"),
        ("The Coca-Cola Company", "Coca-Cola"),
        ("Barclays PLC", "Barclays"),
        ("Airbus SE N.V.", "Airbus SE"),
        ("Berkshire Hathaway Inc", "Berkshire Hathaway"),
        ("JPMorgan Chase & Co.", "JPMorgan Chase"),
        ("Prosus N.V.", "Prosus"),
        ("Tesla", "Tesla"),
        ("  Alphabet Inc.  ", "Alphabet"),
        ("Group", "Group"),
        ("", ""),
    ],
)
def test_clean_company_name(raw: str, expected: str) -> None:
    assert clean_company_name(raw) == expected


# --- build_query ------------------------------------------------------------


def test_build_query_company_appends_long_ticker() -> None:
    assert build_query("company", "Apple Inc.", "AAPL") == '"Apple" stock OR AAPL'


def test_build_query_company_skips_short_ticker() -> None:
    assert build_query("company", "General Electric Co", "GE") == ('"General Electric" stock')


def test_build_query_industry_joins_name_peers_and_industry() -> None:
    query = build_query(
        "industry",
        "NVIDIA Corporation",
        "NVDA",
        peers=["AMD", "INTC"],
        industry="Semiconductors",
    )
    assert query == '("NVIDIA" OR AMD OR INTC OR "Semiconductors") stock'


def test_build_query_industry_skips_missing_parts() -> None:
    query = build_query("industry", "NVIDIA Corporation", "NVDA")
    assert query == '("NVIDIA") stock'


def test_build_query_industry_with_nothing_to_search_raises() -> None:
    with pytest.raises(ValueError):
        build_query("industry", "", "", peers=[], industry=None)


def test_build_query_macro_is_the_fixed_vocabulary() -> None:
    query = build_query("macro", "Apple Inc.", "AAPL")
    assert query.endswith(") stock market")
    assert query.startswith('("Federal Reserve" OR ')
    for term in MACRO_QUERY_TERMS:
        assert f'"{term}"' in query


def test_build_query_unknown_bucket_raises() -> None:
    with pytest.raises(ValueError):
        build_query("sentiment", "Apple Inc.", "AAPL")


# --- queries_for_move -------------------------------------------------------


def test_queries_for_move_company_routing_is_one_query() -> None:
    queries = queries_for_move("company", "Apple Inc.", "AAPL", ["MSFT"], "Electronics")
    assert queries == [("company", '"Apple" stock OR AAPL')]


@pytest.mark.parametrize("routing", ["industry", "macro"])
def test_queries_for_move_adds_the_bucket_query(routing: str) -> None:
    queries = queries_for_move(routing, "NVIDIA Corporation", "NVDA", ["AMD"], "Semiconductors")
    assert len(queries) == 2
    assert queries[0][0] == "company"
    assert queries[1][0] == routing


def test_queries_for_move_industry_without_peers_falls_back_to_company() -> None:
    assert queries_for_move("industry", "", "", peers=[], industry=None) == [
        ("company", '"" stock')
    ]


# --- queries_for_move: the v1.5 edge expansion (plan decision 8) -------------


@pytest.mark.parametrize("routing", ["company", "industry", "macro"])
def test_queries_for_move_without_new_kwargs_is_exactly_the_v1_list(routing: str) -> None:
    """The v1 call is untouched: same list, no extras, for every routing."""
    args = ("NVIDIA Corporation", "NVDA", ["AMD"], "Semiconductors")
    assert queries_for_move(routing, *args) == queries_for_move(
        routing,
        *args,
        sub_routing=None,
        rival_names=(),
        chain_names=(),
        country=None,
    )


def test_queries_for_move_share_shift_adds_one_query_per_rival() -> None:
    queries = queries_for_move(
        "company",
        "NVIDIA Corporation",
        "NVDA",
        ["AMD"],
        "Semiconductors",
        sub_routing="share_shift",
        rival_names=["Advanced Micro Devices, Inc.", "Intel Corporation"],
    )
    assert queries == [
        ("company", '"NVIDIA" stock OR NVDA'),
        ("company", '"Advanced Micro Devices" stock'),
        ("company", '"Intel" stock'),
    ]


def test_queries_for_move_share_shift_ignores_chain_names() -> None:
    queries = queries_for_move(
        "company",
        "NVIDIA Corporation",
        "NVDA",
        sub_routing="share_shift",
        rival_names=["Intel Corp"],
        chain_names=["Taiwan Semiconductor Manufacturing Company"],
    )
    assert queries[1:] == [("company", '"Intel" stock')]


def test_queries_for_move_supply_chain_adds_one_query_per_chain_name() -> None:
    queries = queries_for_move(
        "company",
        "Apple Inc.",
        "AAPL",
        sub_routing="supply_chain",
        rival_names=["Samsung Electronics Co."],
        chain_names=["Taiwan Semiconductor Manufacturing Company", "Foxconn Holdings"],
    )
    assert queries == [
        ("company", '"Apple" stock OR AAPL'),
        ("company", '"Taiwan Semiconductor Manufacturing" stock'),
        ("company", '"Foxconn" stock'),
    ]


def test_queries_for_move_blank_edge_names_are_skipped() -> None:
    queries = queries_for_move(
        "company",
        "Apple Inc.",
        "AAPL",
        sub_routing="share_shift",
        rival_names=["", "   ", "Dell Technologies Inc."],
    )
    assert queries[1:] == [("company", '"Dell Technologies" stock')]


def test_queries_for_move_country_sub_routing_adds_the_geo_query() -> None:
    queries = queries_for_move(
        "macro",
        "Apple Inc.",
        "AAPL",
        sub_routing="country:TW",
    )
    assert queries[-1] == ("macro", geo_query("TW"))
    assert len(queries) == 3


def test_queries_for_move_explicit_country_overrides_the_sub_routing_code() -> None:
    queries = queries_for_move(
        "macro",
        "Apple Inc.",
        "AAPL",
        sub_routing="country:TW",
        country="CN",
    )
    assert queries[-1] == ("macro", geo_query("CN"))


def test_queries_for_move_unknown_country_is_skipped_not_raised() -> None:
    """An edge to a country with no query is a missing expansion, not a failure."""
    v1 = queries_for_move("macro", "Apple Inc.", "AAPL")
    assert queries_for_move("macro", "Apple Inc.", "AAPL", sub_routing="country:ZZ") == v1
    assert queries_for_move("macro", "Apple Inc.", "AAPL", sub_routing="country:") == v1
    assert (
        queries_for_move("macro", "Apple Inc.", "AAPL", sub_routing="country:TW", country="ZZ")
        == v1
    )


@pytest.mark.parametrize("sub_routing", [None, "oil", "dollar", "rates", "gold"])
def test_queries_for_move_sub_routing_without_an_edge_story_adds_nothing(
    sub_routing: str | None,
) -> None:
    v1 = queries_for_move("macro", "Apple Inc.", "AAPL")
    assert (
        queries_for_move(
            "macro",
            "Apple Inc.",
            "AAPL",
            sub_routing=sub_routing,
            rival_names=["Dell Technologies"],
            chain_names=["Foxconn"],
        )
        == v1
    )


def test_queries_for_move_caps_the_extra_queries() -> None:
    queries = queries_for_move(
        "company",
        "NVIDIA Corporation",
        "NVDA",
        sub_routing="share_shift",
        rival_names=["AMD", "Intel", "Qualcomm", "Broadcom", "Marvell"],
    )
    assert len(queries) == 1 + MAX_EXTRA_QUERIES
    assert [query for _, query in queries[1:]] == [
        '"AMD" stock',
        '"Intel" stock',
        '"Qualcomm" stock',
    ]


def test_queries_for_move_dedupes_an_extra_against_the_v1_queries() -> None:
    """A rival whose cleaned name is the company's own is not fetched twice."""
    queries = queries_for_move(
        "company",
        "Ford Motor Company",
        "F",
        sub_routing="share_shift",
        rival_names=["Ford Motor Co.", "General Motors Company"],
    )
    assert queries == [
        ("company", '"Ford Motor" stock'),
        ("company", '"General Motors" stock'),
    ]


def test_queries_for_move_dedupes_the_extras_against_each_other() -> None:
    """The same name down two edges, and two spellings of it, cost one query."""
    queries = queries_for_move(
        "company",
        "Apple Inc.",
        "AAPL",
        sub_routing="supply_chain",
        chain_names=["Foxconn", "Foxconn Holdings", "Foxconn", "Qualcomm Inc"],
    )
    assert queries == [
        ("company", '"Apple" stock OR AAPL'),
        ("company", '"Foxconn" stock'),
        ("company", '"Qualcomm" stock'),
    ]


# --- window_for -------------------------------------------------------------


def test_window_for_defaults_to_plus_minus_one_day() -> None:
    assert window_for(date(2025, 9, 8)) == (date(2025, 9, 7), date(2025, 9, 9))


def test_window_for_custom_width_crosses_a_month_boundary() -> None:
    assert window_for(date(2025, 9, 1), before=3, after=2) == (
        date(2025, 8, 29),
        date(2025, 9, 3),
    )


# --- parse_rss --------------------------------------------------------------


def test_parse_rss_dedupes_and_strips_the_source_suffix() -> None:
    items = parse_rss(RSS_FIXTURE, bucket="company")

    assert len(items) == 2
    first, second = items

    assert isinstance(first, NewsItem)
    assert first.title == "Apple slides 5% after guidance cut"
    assert first.source == "Reuters"
    assert first.url == "https://news.google.com/rss/articles/AAA"
    assert first.published_at == datetime(2025, 9, 8, 13, 45, tzinfo=UTC)
    assert first.language is None
    assert first.news_source == "google_rss"
    assert first.bucket == "company"

    assert second.title == "Analysts trim Apple price targets"
    assert second.source == "CNBC"


def test_parse_rss_default_bucket_is_empty() -> None:
    assert parse_rss(RSS_FIXTURE)[0].bucket == ""


def test_parse_rss_on_garbage_returns_empty() -> None:
    assert parse_rss("<html>429 Too Many Requests</html>") == []


def test_parse_rss_tolerates_a_missing_source_and_bad_date() -> None:
    xml_text = """<?xml version="1.0"?><rss><channel><item>
      <title>Bare headline</title>
      <link>https://example.com/x</link>
      <pubDate>not a date</pubDate>
    </item></channel></rss>"""
    (item,) = parse_rss(xml_text)
    assert item.source is None
    assert item.title == "Bare headline"
    assert item.published_at is None


# --- GoogleNewsRSS.search ---------------------------------------------------


def test_google_rss_search_builds_the_window_and_parses() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        raw = str(request.url)
        assert "after%3A2025-09-07" in raw
        # Google's `before:` is exclusive, so the window's last day is end + 1.
        assert "before%3A2025-09-10" in raw
        assert request.headers["User-Agent"].startswith("Mozilla/5.0")
        return httpx.Response(200, text=RSS_FIXTURE)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    source = GoogleNewsRSS(client=client)
    items = source.search('"Apple" stock OR AAPL', date(2025, 9, 7), date(2025, 9, 9))

    assert source.name == "google_rss"
    assert len(seen) == 1
    assert [item.url for item in items] == [
        "https://news.google.com/rss/articles/AAA",
        "https://news.google.com/rss/articles/BBB",
    ]


def test_google_rss_search_honours_limit() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text=RSS_FIXTURE))
    )
    items = GoogleNewsRSS(client=client).search("q", date(2025, 9, 7), date(2025, 9, 9), limit=1)
    assert len(items) == 1


def test_google_rss_search_returns_empty_on_non_200() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(503, text="nope")))
    assert GoogleNewsRSS(client=client).search("q", date(2025, 9, 7), date(2025, 9, 9)) == []


def test_google_rss_search_returns_empty_on_transport_error() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    client = httpx.Client(transport=httpx.MockTransport(boom))
    assert GoogleNewsRSS(client=client).search("q", date(2025, 9, 7), date(2025, 9, 9)) == []


# --- parse_gdelt / GDELTSource.search ---------------------------------------


def test_parse_gdelt_maps_fields_and_dedupes() -> None:
    items = parse_gdelt(GDELT_FIXTURE, bucket="macro")

    assert len(items) == 2
    first = items[0]
    assert first.url == "https://www.reuters.com/apple-guidance"
    assert first.source == "reuters.com"
    assert first.published_at == datetime(2025, 9, 8, 13, 45, tzinfo=UTC)
    assert first.language == "English"
    assert first.news_source == "gdelt"
    assert first.bucket == "macro"


def test_parse_gdelt_on_an_empty_payload_returns_empty() -> None:
    assert parse_gdelt({}) == []
    assert parse_gdelt({"articles": None}) == []


def test_gdelt_search_sends_the_documented_params() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=GDELT_FIXTURE)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    source = GDELTSource(client=client)
    items = source.search('"Apple" stock', date(2025, 9, 7), date(2025, 9, 9), limit=5)

    assert source.name == "gdelt"
    params = seen[0].url.params
    assert params["query"] == '"Apple" stock sourcelang:english'
    assert params["mode"] == "artlist"
    assert params["format"] == "json"
    assert params["sort"] == "hybridrel"
    assert params["maxrecords"] == "5"
    assert params["startdatetime"] == "20250907000000"
    assert params["enddatetime"] == "20250909235959"
    assert len(items) == 2


def test_gdelt_search_returns_empty_on_non_json() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text="query too short"))
    )
    assert GDELTSource(client=client).search("q", date(2025, 9, 7), date(2025, 9, 9)) == []


def test_gdelt_search_throttles_consecutive_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []
    monkeypatch.setattr(gdelt_module.time, "sleep", slept.append)

    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=GDELT_FIXTURE))
    )
    source = GDELTSource(client=client, throttle_s=5.0)
    source.search("q", date(2025, 9, 7), date(2025, 9, 9))
    assert slept == []

    source.search("q", date(2025, 9, 7), date(2025, 9, 9))
    assert len(slept) == 1
    assert 0 < slept[0] <= 5.0


# --- get_news_source --------------------------------------------------------


def test_get_news_source_resolves_both_names() -> None:
    rss = get_news_source()
    assert isinstance(rss, GoogleNewsRSS)
    assert rss.timeout_s == 20.0

    gdelt = get_news_source("gdelt", timeout_s=1.0, throttle_s=0.5)
    assert isinstance(gdelt, GDELTSource)
    assert (gdelt.timeout_s, gdelt.throttle_s) == (1.0, 0.5)


def test_get_news_source_unknown_name_raises() -> None:
    with pytest.raises(ValueError):
        get_news_source("exa")
