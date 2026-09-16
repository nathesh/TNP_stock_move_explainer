"""Tests for geopolitical headline counts per country (v1.5 plan, decision 5).

No network: every fetch goes through a fake `NewsSource` that returns canned
`NewsItem`s. The database half is skipped until `models.GeoEvent` exists.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from stock_moves.db import Session
from stock_moves.news import NewsItem
from stock_moves.news.base import MACRO_QUERY_TERMS
from stock_moves.news.geo import (
    COUNTRY_ALIASES,
    COUNTRY_NAMES,
    GEO_TERMS,
    MAX_SAMPLE_TITLES,
    GeoEventDraft,
    countries_in,
    fetch_geo_events,
    geo_event_lines,
    geo_query,
    is_geo_headline,
    upsert_geo_events,
)

try:  # `models.GeoEvent` is added by another wave-1 task.
    from stock_moves.models import GeoEvent

    HAS_GEO_EVENT = True
except ImportError:  # pragma: no cover - exercised only before that task lands
    HAS_GEO_EVENT = False

needs_geo_event = pytest.mark.skipif(
    not HAS_GEO_EVENT,
    reason="models.GeoEvent is not defined yet (wave-1 task R1 owns models.py)",
)

START = date(2025, 4, 1)
END = date(2025, 4, 5)


# --------------------------------------------------------------------------- #
# A fake news source
# --------------------------------------------------------------------------- #


class FakeNewsSource:
    """A `NewsSource` that returns canned items and records its calls."""

    name = "fake_rss"

    def __init__(self, items: list[NewsItem]) -> None:
        self.items = items
        self.calls: list[tuple[str, date, date, int]] = []

    def search(self, query: str, start: date, end: date, limit: int = 30) -> list[NewsItem]:
        self.calls.append((query, start, end, limit))
        return list(self.items)[:limit]


def make_item(
    title: str,
    published_at: datetime | None,
    *,
    url: str | None = None,
    news_source: str = "fake_rss",
) -> NewsItem:
    """A `NewsItem` with only the fields `fetch_geo_events` reads set."""
    return NewsItem(
        url=url or f"https://example.test/{abs(hash(title))}",
        title=title,
        source="Reuters",
        published_at=published_at,
        language="en",
        news_source=news_source,
        bucket="macro",
    )


def at(day: int, hour: int = 12) -> datetime:
    """A UTC timestamp on 2025-04-`day`."""
    return datetime(2025, 4, day, hour, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Vocabulary and query
# --------------------------------------------------------------------------- #


def test_country_names_cover_the_required_codes() -> None:
    required = {"TW", "CN", "JP", "KR", "IN", "EU", "MX", "CA", "GB", "DE"}
    required |= {"US", "RU", "IL", "IR", "SA", "UA"}
    assert required <= set(COUNTRY_NAMES)
    assert COUNTRY_NAMES["EU"] == "European Union"
    assert set(COUNTRY_ALIASES) <= set(COUNTRY_NAMES)


def test_geo_terms_are_the_documented_vocabulary() -> None:
    assert GEO_TERMS == (
        "tariff",
        "tariffs",
        "sanctions",
        "export controls",
        "export ban",
        "embargo",
        "conflict",
        "war",
        "blockade",
        "trade dispute",
    )


def test_geo_query_names_the_country_quotes_phrases_and_anchors() -> None:
    query = geo_query("TW")
    assert query.startswith('"Taiwan" (')
    # Anchored to the market the way `base.build_query` anchors its macro query.
    assert query.endswith(" stock market")
    assert MACRO_QUERY_TERMS[-1] == "stock market"
    assert "tariff OR tariffs OR sanctions" in query
    assert '"export controls"' in query  # multi-word terms are phrases
    assert '"trade dispute"' in query
    assert " embargo OR conflict OR war " in query  # single words are bare
    for term in GEO_TERMS:
        assert term in query


def test_geo_query_accepts_a_lowercase_code_and_uses_the_full_name() -> None:
    assert geo_query("eu").startswith('"European Union" (')
    assert geo_query("kr").startswith('"South Korea" (')


@pytest.mark.parametrize("country", ["ZZ", "", "   ", "Taiwan"])
def test_geo_query_rejects_an_unknown_country(country: str) -> None:
    with pytest.raises(ValueError):
        geo_query(country)


# --------------------------------------------------------------------------- #
# Headline matching
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("US slaps new tariffs on Chinese imports", True),
        ("Washington weighs a tariff on chip tools", True),
        ("Commerce adds export controls for H20 chips", True),
        ("Sanctions relief lifts Russian crude", True),
        ("Red Sea blockade snarls freight", True),
        ("Trade dispute escalates as talks stall", True),
        ("Nvidia beats on earnings, guides higher", False),
        ("Warning signs in the jobs report", False),  # "war" must not match "Warning"
        ("Warner Bros. rallies on a bid", False),
        ("Analysts trim price targets", False),
        ("", False),
    ],
)
def test_is_geo_headline(title: str, expected: bool) -> None:
    assert is_geo_headline(title) is expected


def test_countries_in_matches_names_demonyms_and_capitals() -> None:
    codes = ("TW", "CN", "JP", "KR")
    title = "Taiwanese suppliers pause as Beijing widens export controls"
    assert countries_in(title, codes) == ["CN", "TW"]
    assert countries_in("Tokyo protests the new tariffs", codes) == ["JP"]
    assert countries_in("Korean memory makers cut output", codes) == ["KR"]
    assert countries_in("South Korea and Japan agree terms", codes) == ["JP", "KR"]


def test_countries_in_is_sorted_deduped_and_case_insensitive() -> None:
    title = "CHINA and china and Chinese and Beijing"
    assert countries_in(title, ["cn", "CN", " cn "]) == ["CN"]
    assert countries_in("Germany and Canada trade barbs", {"CA", "DE"}) == ["CA", "DE"]


def test_countries_in_only_returns_requested_codes() -> None:
    title = "China and Mexico face new tariffs"
    assert countries_in(title, ["MX"]) == ["MX"]
    assert countries_in(title, []) == []
    assert countries_in("", ["CN"]) == []


def test_countries_in_skips_codes_with_no_query_name() -> None:
    assert countries_in("Brazil raises tariffs", ["BR", "CN"]) == []


def test_countries_in_avoids_the_ambiguous_short_forms() -> None:
    # Bare "us" is an English word and bare "Korea" is also North Korea.
    assert countries_in("Tariffs hit us hard, says the CEO", ["US"]) == []
    assert countries_in("U.S. tariffs take effect", ["US"]) == ["US"]
    assert countries_in("North Korea launches a missile", ["KR"]) == []
    assert countries_in("Seoul weighs a response", ["KR"]) == ["KR"]


# --------------------------------------------------------------------------- #
# Fetching and grouping
# --------------------------------------------------------------------------- #


def test_fetch_geo_events_groups_by_published_day() -> None:
    source = FakeNewsSource(
        [
            make_item("China vows retaliation over tariffs", at(2, 9)),
            make_item("Beijing answers with export controls", at(2, 18)),
            make_item("Tariff pause lifts chipmakers", at(4)),
        ]
    )

    drafts = fetch_geo_events(source, "CN", START, END)

    assert [draft.date for draft in drafts] == [date(2025, 4, 2), date(2025, 4, 4)]
    assert [draft.headline_count for draft in drafts] == [2, 1]
    assert all(draft.country == "CN" for draft in drafts)
    assert all(draft.news_source == "fake_rss" for draft in drafts)
    assert drafts[0].sample_titles == (
        "China vows retaliation over tariffs",
        "Beijing answers with export controls",
    )


def test_fetch_geo_events_calls_the_source_with_the_geo_query() -> None:
    source = FakeNewsSource([make_item("Taiwan tension", at(3))])

    fetch_geo_events(source, "tw", START, END, limit=25)

    assert source.calls == [(geo_query("TW"), START, END, 25)]


def test_fetch_geo_events_drops_undated_items() -> None:
    source = FakeNewsSource(
        [
            make_item("Undated tariff story", None),
            make_item("Dated tariff story", at(3)),
        ]
    )

    drafts = fetch_geo_events(source, "CN", START, END)

    assert len(drafts) == 1
    assert drafts[0].headline_count == 1
    assert drafts[0].sample_titles == ("Dated tariff story",)


def test_fetch_geo_events_drops_items_outside_the_window() -> None:
    source = FakeNewsSource(
        [
            make_item("Too early", datetime(2025, 3, 30, 12, tzinfo=UTC)),
            make_item("In window", at(2)),
            make_item("Too late", datetime(2025, 4, 9, 12, tzinfo=UTC)),
        ]
    )

    drafts = fetch_geo_events(source, "CN", START, END)

    assert [draft.date for draft in drafts] == [date(2025, 4, 2)]


def test_fetch_geo_events_caps_sample_titles_but_not_the_count() -> None:
    items = [make_item(f"Tariff headline {n}", at(2, n)) for n in range(1, 8)]
    drafts = fetch_geo_events(FakeNewsSource(items), "CN", START, END)

    assert len(drafts) == 1
    assert drafts[0].headline_count == 7
    assert len(drafts[0].sample_titles) == MAX_SAMPLE_TITLES
    assert drafts[0].sample_titles[0] == "Tariff headline 1"


def test_fetch_geo_events_dedupes_sample_titles_only() -> None:
    items = [
        make_item("Same headline", at(2, 9), url="https://example.test/a"),
        make_item("SAME HEADLINE", at(2, 10), url="https://example.test/b"),
        make_item("Another headline", at(2, 11), url="https://example.test/c"),
    ]
    drafts = fetch_geo_events(FakeNewsSource(items), "CN", START, END)

    assert drafts[0].headline_count == 3
    assert drafts[0].sample_titles == ("Same headline", "Another headline")


def test_fetch_geo_events_returns_nothing_for_an_empty_or_inverted_window() -> None:
    empty = FakeNewsSource([])
    assert fetch_geo_events(empty, "CN", START, END) == []
    assert empty.calls  # an empty result is still a real search

    inverted = FakeNewsSource([make_item("Tariffs", at(3))])
    assert fetch_geo_events(inverted, "CN", END, START) == []
    assert inverted.calls == []  # never search a window that cannot contain a day


def test_fetch_geo_events_prefers_the_item_news_source() -> None:
    source = FakeNewsSource([make_item("Tariffs", at(3), news_source="gdelt")])
    assert fetch_geo_events(source, "CN", START, END)[0].news_source == "gdelt"


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


def draft(
    day: int,
    country: str = "CN",
    *,
    count: int = 3,
    titles: tuple[str, ...] = ("China vows retaliation over tariffs",),
    news_source: str = "google_rss",
) -> GeoEventDraft:
    return GeoEventDraft(
        date=date(2025, 4, day),
        country=country,
        headline_count=count,
        sample_titles=titles,
        news_source=news_source,
    )


def stored_rows(session: Session) -> list[GeoEvent]:
    from sqlmodel import select

    return list(session.exec(select(GeoEvent)).all())


@needs_geo_event
def test_upsert_geo_events_is_idempotent(session: Session) -> None:
    drafts = [draft(2), draft(3)]

    assert upsert_geo_events(session, drafts) == 2
    assert upsert_geo_events(session, drafts) == 0

    rows = stored_rows(session)
    assert len(rows) == 2
    assert {row.date for row in rows} == {date(2025, 4, 2), date(2025, 4, 3)}
    assert all(row.country == "CN" for row in rows)


@needs_geo_event
def test_upsert_geo_events_updates_a_changed_count(session: Session) -> None:
    assert upsert_geo_events(session, [draft(2, count=3)]) == 1
    assert upsert_geo_events(session, [draft(2, count=9)]) == 1

    rows = stored_rows(session)
    assert len(rows) == 1
    assert rows[0].headline_count == 9


@needs_geo_event
def test_upsert_geo_events_separates_countries_and_sources(session: Session) -> None:
    written = upsert_geo_events(
        session,
        [
            draft(2, "CN"),
            draft(2, "TW"),
            draft(2, "CN", news_source="gdelt"),
        ],
    )

    assert written == 3
    assert len(stored_rows(session)) == 3


@needs_geo_event
def test_upsert_geo_events_handles_no_drafts(session: Session) -> None:
    assert upsert_geo_events(session, []) == 0
    assert stored_rows(session) == []


@needs_geo_event
def test_geo_event_lines_renders_and_sorts(session: Session) -> None:
    upsert_geo_events(
        session,
        [
            draft(3, "TW", count=4, titles=("Taipei answers the export ban",)),
            draft(2, "CN", count=12, titles=("China vows retaliation over tariffs", "Second")),
            draft(2, "TW", count=1, titles=("Taiwan tension rises",)),
        ],
    )

    lines = geo_event_lines(session, ["CN", "TW"], date(2025, 4, 1), date(2025, 4, 5))

    assert lines == [
        "2025-04-02 CN 12 headlines: China vows retaliation over tariffs",
        "2025-04-02 TW 1 headlines: Taiwan tension rises",
        "2025-04-03 TW 4 headlines: Taipei answers the export ban",
    ]


@needs_geo_event
def test_geo_event_lines_filters_by_country_and_window(session: Session) -> None:
    upsert_geo_events(session, [draft(2, "CN"), draft(4, "TW")])

    assert geo_event_lines(session, ["TW"], date(2025, 4, 1), date(2025, 4, 5)) == [
        "2025-04-04 TW 3 headlines: China vows retaliation over tariffs"
    ]
    assert geo_event_lines(session, ["CN"], date(2025, 4, 3), date(2025, 4, 5)) == []
    assert geo_event_lines(session, [], date(2025, 4, 1), date(2025, 4, 5)) == []
    assert geo_event_lines(session, ["CN"], date(2025, 4, 5), date(2025, 4, 1)) == []


@needs_geo_event
def test_geo_event_lines_omits_a_missing_sample_title(session: Session) -> None:
    upsert_geo_events(session, [draft(2, "CN", count=2, titles=())])

    assert geo_event_lines(session, ["cn"], date(2025, 4, 1), date(2025, 4, 5)) == [
        "2025-04-02 CN 2 headlines"
    ]
