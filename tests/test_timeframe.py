"""Tests for `stock_moves.timeframe` — the clock the model is not allowed to have.

`resolve` takes `today` as an argument precisely so it can be pinned here:
every case below is asserted against Tuesday 15 September 2026, so "this week"
and "last quarter" have exactly one right answer rather than a drifting one.

Two properties matter as much as the arithmetic. A phrase that is not there
must resolve to `None` — most questions name no period, and inventing one is
the bug this module exists to prevent — and a near-miss ("todays", "mostly")
must not be read as one.
"""

from __future__ import annotations

from datetime import date

import pytest

from stock_moves.timeframe import Window, resolve

#: A Tuesday, so the Mon-Sun week boundaries are visible rather than aligned.
TODAY = date(2026, 9, 15)


def test_the_fixed_today_is_a_tuesday() -> None:
    """If this ever fails, every week assertion below means something else."""
    assert TODAY.weekday() == 1


def window(text: str) -> Window:
    resolved = resolve(text, TODAY)
    assert resolved is not None, f"{text!r} resolved to nothing"
    return resolved


# --------------------------------------------------------------------------- #
# The phrases
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("today", date(2026, 9, 15), date(2026, 9, 15)),
        ("yesterday", date(2026, 9, 14), date(2026, 9, 14)),
        # Mon-Sun: "this week" stops at today, "last week" is the whole of the
        # previous one.
        ("this week", date(2026, 9, 14), date(2026, 9, 15)),
        ("last week", date(2026, 9, 7), date(2026, 9, 13)),
        ("this month", date(2026, 9, 1), date(2026, 9, 15)),
        ("last month", date(2026, 8, 1), date(2026, 8, 31)),
        ("this quarter", date(2026, 7, 1), date(2026, 9, 15)),
        ("last quarter", date(2026, 4, 1), date(2026, 6, 30)),
        # The phrase from the defect report. Nothing clever: January 1st.
        ("this year", date(2026, 1, 1), date(2026, 9, 15)),
        ("last year", date(2025, 1, 1), date(2025, 12, 31)),
        ("year to date", date(2026, 1, 1), date(2026, 9, 15)),
        ("ytd", date(2026, 1, 1), date(2026, 9, 15)),
        ("last 30 days", date(2026, 8, 16), date(2026, 9, 15)),
        ("past 30 days", date(2026, 8, 16), date(2026, 9, 15)),
        ("past two weeks", date(2026, 9, 1), date(2026, 9, 15)),
        ("last three months", date(2026, 6, 15), date(2026, 9, 15)),
        # A named month with no year is the most recent one that has begun.
        ("in March", date(2026, 3, 1), date(2026, 3, 31)),
        ("March 2025", date(2025, 3, 1), date(2025, 3, 31)),
        ("October", date(2025, 10, 1), date(2025, 10, 31)),
        ("Q2", date(2026, 4, 1), date(2026, 6, 30)),
        ("Q4", date(2025, 10, 1), date(2025, 12, 31)),
        ("Q1 2025", date(2025, 1, 1), date(2025, 3, 31)),
    ],
)
def test_each_phrase_resolves_to_its_window(text: str, start: date, end: date) -> None:
    resolved = window(text)
    assert (resolved.start, resolved.end) == (start, end)


def test_the_match_is_found_inside_a_real_question() -> None:
    """The phrase is a substring of a sentence, not the whole input."""
    resolved = window("Why did AAPL drop the most this year?")
    assert (resolved.start, resolved.end) == (date(2026, 1, 1), TODAY)
    # And the phrase is kept for echoing, in the user's own words.
    assert resolved.phrase == "this year"


def test_a_window_never_starts_after_today() -> None:
    """The guarantee the tool binding relies on: no empty future window."""
    for text in ("October", "Q4", "December", "this quarter", "this year"):
        assert window(text).start <= TODAY


def test_the_end_is_clamped_to_today() -> None:
    """September 2026 is not over, so "in September" stops at the 15th."""
    resolved = window("in September")
    assert (resolved.start, resolved.end) == (date(2026, 9, 1), TODAY)


# --------------------------------------------------------------------------- #
# Not a phrase
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "",
        "why did AAPL fall",
        "what were the biggest moves",
        # Whole words only: a phrase that merely starts one is not one.
        "todays close was ugly",
        "mostly quiet, then a gap down",
        "yesterdays gap",
        "this weekend",
        "last weeks",
        # "may" is an English word first; a bare one is not a month.
        "it may be the earnings report",
    ],
)
def test_text_with_no_phrase_resolves_to_nothing(text: str) -> None:
    assert resolve(text, TODAY) is None


def test_may_counts_as_a_month_when_the_wording_makes_it_one() -> None:
    """The escape hatch for the one ambiguous month name."""
    assert window("in May").start == date(2026, 5, 1)
    assert window("May 2025").start == date(2025, 5, 1)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_the_leftmost_phrase_wins() -> None:
    """Two phrases in one question resolve to the first, every time."""
    text = "last week, or was it this year?"
    assert window(text).phrase == "last week"
    assert resolve(text, TODAY) == resolve(text, TODAY)


def test_resolution_is_case_insensitive() -> None:
    assert resolve("THIS YEAR", TODAY) == Window(date(2026, 1, 1), TODAY, "THIS YEAR")


def test_a_different_today_moves_the_window() -> None:
    """No hidden clock: the same text against another day is another window."""
    resolved = resolve("this year", date(2025, 3, 2))
    assert resolved == Window(date(2025, 1, 1), date(2025, 3, 2), "this year")
