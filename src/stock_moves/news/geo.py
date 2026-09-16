"""Geopolitical headline counts per country (v1.5 plan, decision 5).

The GDELT Events CSV feed is 96 files a day and the DOC API cannot filter by
CAMEO code, so v1.5 does not have real events. What it has is a *count*: for a
country and a day, how many headlines the existing `NewsSource` returned for
that country's geopolitical vocabulary, plus a handful of sample titles. That
is `geo_events(date, country, headline_count, sample_titles_json, news_source)`
— no new API, no throttle beyond the one v1 already has.

The pure half of this module (`geo_query`, `is_geo_headline`, `countries_in`,
`fetch_geo_events`) imports nothing from the rest of `stock_moves`, in the
spirit of `news.base`. The two functions that touch the database import the
table late, inside themselves.

The country matching here is a heuristic feeding a *gate* (plan decision 7),
not an entity linker. It is deliberately conservative where a short form is an
ordinary English word: bare "US" is not matched (it collides with "us") and
bare "Korea" is not matched (it collides with "North Korea"); "U.S.",
"American", "Washington", "Korean" and "Seoul" are.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from sqlmodel import Session, col, select

from .base import NewsSource

__all__ = [
    "COUNTRY_ALIASES",
    "COUNTRY_NAMES",
    "GEO_TERMS",
    "MAX_SAMPLE_TITLES",
    "GeoEventDraft",
    "countries_in",
    "fetch_geo_events",
    "geo_event_lines",
    "geo_query",
    "is_geo_headline",
    "upsert_geo_events",
]

# `date` is a field name on `GeoEventDraft`, which shadows the imported type
# inside the class body; annotate with this alias (same trick as `models.py`).
DateT = date

# ISO-3166 alpha-2 -> the name to put in a news query. The ten countries with a
# country ETF in plan decision 4 (TW CN JP KR IN EU MX CA GB DE) plus the six
# that show up in the geopolitical vocabulary without one.
COUNTRY_NAMES: dict[str, str] = {
    "CA": "Canada",
    "CN": "China",
    "DE": "Germany",
    "EU": "European Union",
    "GB": "United Kingdom",
    "IL": "Israel",
    "IN": "India",
    "IR": "Iran",
    "JP": "Japan",
    "KR": "South Korea",
    "MX": "Mexico",
    "RU": "Russia",
    "SA": "Saudi Arabia",
    "TW": "Taiwan",
    "UA": "Ukraine",
    "US": "United States",
}

# Demonyms, capitals and short forms, used only by `countries_in`. Never by
# `geo_query`: a query wants one unambiguous name, not an OR of nicknames.
COUNTRY_ALIASES: dict[str, tuple[str, ...]] = {
    "CA": ("Canadian", "Ottawa"),
    "CN": ("Chinese", "Beijing", "PRC"),
    "DE": ("German", "Berlin"),
    "EU": ("EU", "Brussels", "European Commission"),
    "GB": ("Britain", "British", "UK", "London"),
    "IL": ("Israeli", "Jerusalem", "Tel Aviv"),
    "IN": ("Indian", "New Delhi"),
    "IR": ("Iranian", "Tehran"),
    "JP": ("Japanese", "Tokyo"),
    "KR": ("Korean", "Seoul"),
    "MX": ("Mexican", "Mexico City"),
    "RU": ("Russian", "Moscow", "Kremlin"),
    "SA": ("Saudi", "Saudi Arabian", "Riyadh"),
    "TW": ("Taiwanese", "Taipei"),
    "UA": ("Ukrainian", "Kyiv", "Kiev"),
    "US": ("U.S.", "American", "Washington"),
}

# The geopolitical vocabulary. Plan decision 5 names the query terms; this is
# also the list the gate rule (decision 7) matches a headline against, so a
# term is added here once and both halves see it.
GEO_TERMS: tuple[str, ...] = (
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

MAX_SAMPLE_TITLES: int = 5
"""How many titles a `GeoEventDraft` carries as evidence of its count."""


def _phrase_pattern(terms: Iterable[str]) -> re.Pattern[str]:
    """Case-insensitive alternation over `terms`, matched on word boundaries.

    Lookarounds rather than `\\b` so a term that ends in punctuation ("U.S.")
    still matches at the end of a title, and longest-first so "export controls"
    is not shadowed by a prefix that shares its first word.
    """
    ordered = sorted({term.strip() for term in terms if term.strip()}, key=len, reverse=True)
    if not ordered:
        # Never matches; `re` has no "match nothing" literal, so use a lookahead.
        return re.compile(r"(?!)")
    alternation = "|".join(re.escape(term).replace(r"\ ", r"\s+") for term in ordered)
    return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE)


_GEO_PATTERN: re.Pattern[str] = _phrase_pattern(GEO_TERMS)

_COUNTRY_PATTERNS: dict[str, re.Pattern[str]] = {
    code: _phrase_pattern((name, *COUNTRY_ALIASES.get(code, ())))
    for code, name in COUNTRY_NAMES.items()
}


def _country_name(country: str) -> str:
    """The query name for an alpha-2 code, or `ValueError` if it is not known."""
    code = country.strip().upper()
    if not code:
        raise ValueError("country code must not be empty")
    try:
        return COUNTRY_NAMES[code]
    except KeyError:
        raise ValueError(
            f"unknown country code {country!r}; expected one of {sorted(COUNTRY_NAMES)}"
        ) from None


def geo_query(country: str) -> str:
    """Build the geopolitical headline query for one country (plan decision 5).

    `geo_query("TW")` is `'"Taiwan" (tariff OR tariffs OR ... ) stock market'`.
    Multi-word terms are quoted so the source treats them as phrases, and the
    whole query is anchored with "stock market" the way `base.build_query`
    anchors its macro query — without the anchor, "Taiwan conflict" returns the
    whole of foreign-affairs coverage rather than the part that moved a price.
    """
    name = _country_name(country)
    terms = " OR ".join(f'"{term}"' if " " in term else term for term in GEO_TERMS)
    return f'"{name}" ({terms}) stock market'


def is_geo_headline(title: str) -> bool:
    """True if `title` contains any `GEO_TERMS` phrase, case-insensitively."""
    return bool(title) and _GEO_PATTERN.search(title) is not None


def countries_in(title: str, codes: Iterable[str]) -> list[str]:
    """Which of `codes` the title names, by country name, demonym or capital.

    Returns a sorted, de-duplicated list of alpha-2 codes. Codes with no entry
    in `COUNTRY_NAMES` can never match and are skipped silently, so a company
    with a `country` edge outside the dict simply contributes nothing here.
    """
    if not title:
        return []
    wanted = {str(code).strip().upper() for code in codes}
    return sorted(
        code
        for code in wanted
        if code in _COUNTRY_PATTERNS and _COUNTRY_PATTERNS[code].search(title) is not None
    )


@dataclass(frozen=True)
class GeoEventDraft:
    """One country-day of geopolitical headlines, before it is stored.

    `headline_count` counts every item returned for that day; `sample_titles`
    holds at most `MAX_SAMPLE_TITLES` distinct ones as the evidence a reader
    (and the prose layer) actually sees.
    """

    date: DateT
    country: str
    headline_count: int
    sample_titles: tuple[str, ...]
    news_source: str


def _published_date(published_at: datetime | None) -> date | None:
    """The UTC calendar date of a timestamp; `None` if there is no timestamp.

    Aware timestamps are converted to UTC first so an item stamped 20:00-05:00
    lands on the next day, the same day the RSS feed's UTC `pubDate` would.
    """
    if published_at is None:
        return None
    if published_at.tzinfo is not None:
        return published_at.astimezone(UTC).date()
    return published_at.date()


def _sample_titles(titles: Iterable[str]) -> tuple[str, ...]:
    """Up to `MAX_SAMPLE_TITLES` distinct non-empty titles, in arrival order."""
    seen: set[str] = set()
    kept: list[str] = []
    for title in titles:
        cleaned = (title or "").strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        kept.append(cleaned)
        if len(kept) == MAX_SAMPLE_TITLES:
            break
    return tuple(kept)


def fetch_geo_events(
    source: NewsSource,
    country: str,
    start: date,
    end: date,
    limit: int = 50,
) -> list[GeoEventDraft]:
    """One `GeoEventDraft` per day in `[start, end]` that returned a headline.

    Items with no `published_at` are dropped — a count for a day cannot include
    an item that does not say which day it belongs to — and so are items whose
    date falls outside the window, which a source is free to return. Days with
    no items get no draft at all rather than a zero row: absence of a headline
    is absence of evidence, and storing a zero would let the gate read it as a
    fact. Drafts come back sorted by date.
    """
    if start > end:
        return []

    code = country.strip().upper()
    items = source.search(geo_query(code), start, end, limit)

    by_day: dict[date, list[Any]] = {}
    for item in items:
        day = _published_date(item.published_at)
        if day is None or day < start or day > end:
            continue
        by_day.setdefault(day, []).append(item)

    source_name = getattr(source, "name", "") or ""
    drafts: list[GeoEventDraft] = []
    for day in sorted(by_day):
        day_items = by_day[day]
        news_source = next(
            (item.news_source for item in day_items if item.news_source), source_name
        )
        drafts.append(
            GeoEventDraft(
                date=day,
                country=code,
                headline_count=len(day_items),
                sample_titles=_sample_titles(item.title for item in day_items),
                news_source=news_source,
            )
        )
    return drafts


def _geo_event_table() -> Any:
    """The `geo_events` table, imported late to keep the pure half import-free."""
    from stock_moves.models import GeoEvent

    return GeoEvent


def upsert_geo_events(session: Session, drafts: Sequence[GeoEventDraft]) -> int:
    """Store `drafts`, idempotent on `(date, country, news_source)`.

    Returns the number of rows actually written: a draft that inserts a row, or
    that changes the count or the samples of one, counts; a re-run over
    unchanged drafts writes nothing and returns 0. Duplicates inside `drafts`
    keep the last occurrence, so a caller that fetched a day twice stores the
    fuller answer.
    """
    if not drafts:
        return 0

    geo_event = _geo_event_table()
    from stock_moves.models import utcnow

    deduped: dict[tuple[date, str, str], GeoEventDraft] = {
        (draft.date, draft.country, draft.news_source): draft for draft in drafts
    }

    countries = {country for _, country, _ in deduped}
    dates = [day for day, _, _ in deduped]
    existing = {
        (row.date, row.country, row.news_source): row
        for row in session.exec(
            select(geo_event).where(
                col(geo_event.country).in_(countries),
                col(geo_event.date) >= min(dates),
                col(geo_event.date) <= max(dates),
            )
        ).all()
    }

    now = utcnow()
    written = 0
    for key, draft in deduped.items():
        titles_json = json.dumps(list(draft.sample_titles))
        row = existing.get(key)
        if row is None:
            session.add(
                geo_event(
                    date=draft.date,
                    country=draft.country,
                    headline_count=draft.headline_count,
                    sample_titles_json=titles_json,
                    news_source=draft.news_source,
                    fetched_at=now,
                )
            )
            written += 1
            continue
        if row.headline_count == draft.headline_count and row.sample_titles_json == titles_json:
            continue
        row.headline_count = draft.headline_count
        row.sample_titles_json = titles_json
        row.fetched_at = now
        session.add(row)
        written += 1

    if written:
        session.commit()
    return written


def _first_sample_title(sample_titles_json: str | None) -> str:
    """The first stored sample title; `""` if there is none or the json is bad."""
    try:
        parsed = json.loads(sample_titles_json or "[]")
    except (TypeError, ValueError):
        return ""
    if not isinstance(parsed, list) or not parsed:
        return ""
    return str(parsed[0]).strip()


def geo_event_lines(
    session: Session,
    country_codes: Iterable[str],
    start: date,
    end: date,
) -> list[str]:
    """Stored geo events as one line each, for a prompt or a rendered answer.

    `"2025-04-03 CN 12 headlines: China vows retaliation over new tariffs"`, or
    the same line without the colon when the row kept no sample title. Sorted
    by date, then country, then source, so two sources for one country-day read
    in a stable order.
    """
    wanted = {str(code).strip().upper() for code in country_codes}
    wanted.discard("")
    if not wanted or start > end:
        return []

    geo_event = _geo_event_table()
    rows = session.exec(
        select(geo_event).where(
            col(geo_event.country).in_(wanted),
            col(geo_event.date) >= start,
            col(geo_event.date) <= end,
        )
    ).all()

    lines: list[str] = []
    for row in sorted(rows, key=lambda r: (r.date, r.country, r.news_source or "")):
        head = f"{row.date:%Y-%m-%d} {row.country} {row.headline_count} headlines"
        title = _first_sample_title(row.sample_titles_json)
        lines.append(f"{head}: {title}" if title else head)
    return lines
