"""Turning "this year" into two dates, on the server, from the real clock.

A language model has no clock. Asked "why did AAPL drop the most this year?"
it will still answer with a `start` and an `end`, because the tool schema
offers them and the training data is full of plausible-looking dates -- and
what comes back is the year it happened to be trained in: `2024-01-01`,
`2024-06-18`, once an `end` of `2024-06-07` that clipped the window shut and
produced "there are no notable one-day drops recorded for AAPL so far this
year" for a question whose answer was a 7.4% fall. The model was not wrong
about the data; it was wrong about the date, and it had no way to be right.

So the model is not allowed to compute one. The time window in a question is
matched here, resolved against a `today` the *caller* supplies, and applied by
the tool binding; `providers/base.py` no longer shows the model a `start` or an
`end` at all. The only date a model may pass is one the user wrote out in full,
and that goes to `get_move`, not through this module.

Like `narrate.py` this is deliberately dependency-free -- no database, no
settings, no provider imports -- and `resolve(text, today)` is still a pure
function of the `today` it is handed, which is the only reason a window can be
tested at all. The one clock here is `market_today`, kept beside the resolver
because *which* clock counts is part of the same rule: the date is the
exchange's, not the server's. `None` is the ordinary answer. Most questions
name no period, and a one-year dataset does not need one.
"""

from __future__ import annotations

import re
from calendar import monthrange
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

__all__ = ["MARKET_TZ", "Window", "market_today", "resolve"]


#: The exchange's clock, not the server's.
#:
#: The data is US equities, so "today" and "yesterday" are questions about a
#: *trading* day on the New York calendar. The process may run anywhere -- on
#: Vercel it runs in UTC, where after 5pm Pacific / 8pm Eastern every window is
#: already a day ahead of the market: "this year" ended on a date the exchange
#: had not reached, and "yesterday" resolved to what was still today in New
#: York. The server's location is an accident of deployment; the exchange's
#: date is what the question is about.
#:
#: `zoneinfo` is stdlib, so this keeps the module dependency-free.
MARKET_TZ = ZoneInfo("America/New_York")


def market_today(now: datetime | None = None) -> date:
    """The exchange's current date: `now` in `MARKET_TZ`, defaulting to the real clock.

    `now` must be timezone-aware. A naive datetime is rejected rather than
    assumed to be anything: assuming a zone is exactly the bug this function
    exists to fix, and a caller holding a wall clock knows which zone it is in.
    """
    current = datetime.now(UTC) if now is None else now
    if current.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return current.astimezone(MARKET_TZ).date()


@dataclass(frozen=True)
class Window:
    """An inclusive date range and the words that produced it.

    `phrase` is the matched text exactly as the user wrote it, so an answer --
    or the UI's "show its work" panel -- can say *which* window was applied
    rather than making the reader infer it from two ISO dates.
    """

    start: date
    end: date
    phrase: str


# --------------------------------------------------------------------------- #
# Grammar
# --------------------------------------------------------------------------- #

_MONTHS: tuple[str, ...] = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)

#: Small counts spelled out. Beyond twelve people write digits, and a longer
#: list buys false positives ("last twenty" is rarely a date).
_NUMBER_WORDS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

_COUNT = "|".join([r"\d{1,4}", *sorted(_NUMBER_WORDS, key=len, reverse=True)])
_MONTH_NAMES = "|".join(_MONTHS)

#: One pass over the text. Alternatives are ordered most specific first, so at
#: a given position "last 30 days" wins over "last", and "March 2025" over a
#: bare "March"; `finditer` then gives them to `resolve` left to right.
_PATTERN = re.compile(
    rf"""
      \b(?P<ytd>year\s+to\s+date|ytd)\b
    | \b(?:last|past)\s+(?P<count>{_COUNT})\s+(?P<unit>day|week|month)s?\b
    | \b(?P<which>this|last)\s+(?P<named>week|month|quarter|year)\b
    | \b(?P<today>today)\b
    | \b(?P<yesterday>yesterday)\b
    | \bq(?P<quarter>[1-4])\b(?:\s+(?P<qyear>\d{{4}})\b)?
    | (?:\b(?P<prep>in|during|for|of)\s+)?
      \b(?P<month>{_MONTH_NAMES})\b(?:\s+(?P<myear>\d{{4}})\b)?
    """,
    re.IGNORECASE | re.VERBOSE,
)


# --------------------------------------------------------------------------- #
# Calendar helpers
# --------------------------------------------------------------------------- #


def _month_end(year: int, month: int) -> date:
    return date(year, month, monthrange(year, month)[1])


def _shift_months(day: date, months: int) -> date:
    """`day` moved back `months` calendar months, clamped to the month's length."""
    index = day.year * 12 + (day.month - 1) - months
    year, month = divmod(index, 12)
    month += 1
    return date(year, month, min(day.day, monthrange(year, month)[1]))


def _week_start(day: date) -> date:
    """The Monday of `day`'s week. Weeks are Mon-Sun throughout."""
    return day - timedelta(days=day.weekday())


def _quarter_of(day: date) -> int:
    return (day.month - 1) // 3 + 1


def _quarter_bounds(year: int, quarter: int) -> tuple[date, date]:
    first = 3 * (quarter - 1) + 1
    return date(year, first, 1), _month_end(year, first + 2)


def _most_recent(build: Callable[[int], tuple[date, date]], today: date) -> tuple[date, date]:
    """`build(year)` for this year, or last year when this year's has not begun.

    "October" asked in September means last October, not a window that starts
    two weeks from now and is therefore empty. The bounds are rebuilt for the
    earlier year rather than shifted, so a February keeps its own last day.
    """
    start, end = build(today.year)
    return (start, end) if start <= today else build(today.year - 1)


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def resolve(text: str, today: date) -> Window | None:
    """The first time window `text` names, resolved against `today`.

    `None` when it names none, which is the common case. A window never starts
    after `today` and its `end` is always clamped to `today`: the data stops
    there, and a question about a period that has not happened yet should come
    back empty rather than wrong.
    """
    if not text:
        return None
    for match in _PATTERN.finditer(text):
        window = _window(match, today)
        if window is not None:
            return window
    return None


def _window(match: re.Match[str], today: date) -> Window | None:
    """One match as a window, or `None` when it is not really a date."""
    text = match.string
    phrase = match.group(0).strip()
    # Every alternative is matched case-insensitively, so every captured word
    # has to be folded before it is compared.
    named = (match.group("named") or "").lower()

    if match.group("ytd") or named == "year":
        start, end = _year(match, today)
    elif match.group("count"):
        start, end = _relative(match, today), today
    elif named:
        start, end = _named(match.group("which").lower(), named, today)
    elif match.group("today"):
        start = end = today
    elif match.group("yesterday"):
        start = end = today - timedelta(days=1)
    elif match.group("quarter"):
        start, end = _quarter(match, today)
    elif match.group("month"):
        bounds = _month(match, today)
        if bounds is None:
            return None
        start, end = bounds
        phrase = text[match.start("month") : match.end()].strip()
    else:  # pragma: no cover - every alternative is handled above
        return None

    if start > today:
        return None
    return Window(start, min(end, today), " ".join(phrase.split()))


def _year(match: re.Match[str], today: date) -> tuple[date, date]:
    """`year to date` / `ytd` / `this year` / `last year`.

    "This year" is January 1st to today and nothing cleverer: the whole point
    of the module is that the phrase has one boring, checkable meaning.
    """
    if (match.group("which") or "").lower() == "last":
        return date(today.year - 1, 1, 1), date(today.year - 1, 12, 31)
    return date(today.year, 1, 1), today


def _relative(match: re.Match[str], today: date) -> date:
    """The start of `last N days|weeks|months`, counted back from `today`."""
    raw = match.group("count").lower()
    count = _NUMBER_WORDS.get(raw) or int(raw)
    unit = match.group("unit").lower()
    if unit == "month":
        return _shift_months(today, count)
    return today - timedelta(days=count * (7 if unit == "week" else 1))


def _named(which: str, unit: str, today: date) -> tuple[date, date]:
    """`this`/`last` + `week`/`month`/`quarter`. The year case is `_year`."""
    if unit == "week":
        this_start = _week_start(today)
        if which == "this":
            return this_start, today
        last_start = this_start - timedelta(days=7)
        return last_start, last_start + timedelta(days=6)

    if unit == "month":
        if which == "this":
            return today.replace(day=1), today
        first_of_this = today.replace(day=1)
        last_end = first_of_this - timedelta(days=1)
        return last_end.replace(day=1), last_end

    quarter = _quarter_of(today)
    if which == "this":
        return _quarter_bounds(today.year, quarter)[0], today
    year = today.year if quarter > 1 else today.year - 1
    return _quarter_bounds(year, quarter - 1 if quarter > 1 else 4)


def _quarter(match: re.Match[str], today: date) -> tuple[date, date]:
    """`Q1`..`Q4`, with an explicit year or the most recent one that has begun."""
    quarter = int(match.group("quarter"))
    raw_year = match.group("qyear")
    if raw_year:
        return _quarter_bounds(int(raw_year), quarter)
    return _most_recent(lambda year: _quarter_bounds(year, quarter), today)


def _month(match: re.Match[str], today: date) -> tuple[date, date] | None:
    """A named month, with an explicit year or the most recent one that has begun.

    "May" is the one month that is also an ordinary English word ("it may be
    earnings"), and a stray auxiliary verb silently clipping the window is
    exactly the failure this module exists to stop. So a bare "may" is not a
    date: it counts only when a preposition or a year makes the intent explicit.
    """
    name = match.group("month").lower()
    raw_year = match.group("myear")
    if name == "may" and not raw_year and not match.group("prep"):
        return None

    month = _MONTHS.index(name) + 1

    def bounds(year: int) -> tuple[date, date]:
        return date(year, month, 1), _month_end(year, month)

    return bounds(int(raw_year)) if raw_year else _most_recent(bounds, today)
