"""Hardcoded US macro event calendar for 2025 and 2026.

Two event streams matter for the ``near_fomc`` / ``near_cpi`` flags described in
DESIGN.md section 1:

* ``FOMC_DATES`` — the **decision day** of each regularly scheduled FOMC
  meeting, i.e. the second day of the two-day meeting, which is when the policy
  statement is released (Federal Reserve published calendar).
* ``CPI_DATES`` — BLS Consumer Price Index news release dates (08:30 ET).

Both lists are deliberately literal rather than computed: the schedules are
irregular (holidays, and in 2025 a lapse in appropriations) and a rule would be
wrong more often than a table. Dates that could not be confirmed against the
published schedules are omitted rather than guessed, so a missing date makes the
flag ``False`` and never invents an event.

Two 2025 CPI irregularities are reflected below:

* The September 2025 CPI, normally due 2025-10-15, was published 2025-10-24.
* The October 2025 CPI was never published, so there is no November 2025
  release date.

Dates after the last trading day in a frame are simply ignored by
``near_dates``, so the forward-looking 2026 entries are harmless.
"""

from __future__ import annotations

from datetime import date

__all__ = ["CPI_DATES", "FOMC_DATES"]

#: FOMC statement/decision days (second day of each scheduled meeting).
FOMC_DATES: tuple[date, ...] = (
    # 2025
    date(2025, 1, 29),
    date(2025, 3, 19),
    date(2025, 5, 7),
    date(2025, 6, 18),
    date(2025, 7, 30),
    date(2025, 9, 17),
    date(2025, 10, 29),
    date(2025, 12, 10),
    # 2026
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 4, 29),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12, 9),
)

#: BLS CPI news release days.
CPI_DATES: tuple[date, ...] = (
    # 2025
    date(2025, 1, 15),
    date(2025, 2, 12),
    date(2025, 3, 12),
    date(2025, 4, 10),
    date(2025, 5, 13),
    date(2025, 6, 11),
    date(2025, 7, 15),
    date(2025, 8, 12),
    date(2025, 9, 11),
    date(2025, 10, 24),  # September data, delayed from 2025-10-15
    date(2025, 12, 18),  # November data; the October report was never published
    # 2026
    date(2026, 1, 13),
    date(2026, 2, 13),
    date(2026, 3, 11),
    date(2026, 4, 10),
    date(2026, 5, 12),
    date(2026, 6, 10),
    date(2026, 7, 14),
    date(2026, 8, 12),
    date(2026, 9, 11),
    date(2026, 10, 14),
    date(2026, 11, 10),
    date(2026, 12, 10),
)
