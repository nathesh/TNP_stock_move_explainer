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
