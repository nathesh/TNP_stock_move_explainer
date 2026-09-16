"""Tests for sub-routing — v1.5 plan, decision 6.

``routing`` says *where* to look; ``sub_routing`` says *what kind of story* to
look for once you are there. Two of the three answers are arithmetic on related
names' returns — rivals moving the other way, a chain moving the same way — and
the third is the factor attribution passed through.

Everything here is a pure function over plain numbers and small frames, so the
tests assert exact labels rather than tolerances: no network, no database, no
random walks.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from stock_moves.moves import (
    MIN_RIVALS,
    SHARE_SHIFT_RATIO,
    SUB_ROUTE_SHARE_SHIFT,
    SUB_ROUTE_SUPPLY_CHAIN,
    SUPPLY_CHAIN_RATIO,
    peer_comove,
    signed_comove,
    sub_route,
)

#: The worked example from the task: a 5% fall with rivals up 2%.
STOCK_DROP = -0.05
RIVAL_RISE = 0.02
#: ... and the supply-chain case: the same fall with suppliers down 3%.
SUPPLIER_DROP = -0.03

NAN = float("nan")


def _dates(n: int = 3) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.bdate_range(start="2025-04-14", periods=n), name="date")


# --------------------------------------------------------------------------
# signed_comove
# --------------------------------------------------------------------------


def test_signed_comove_averages_across_columns_and_keeps_the_sign() -> None:
    dates = _dates()
    rivals = pd.DataFrame({"AMD": [0.02, -0.01, 0.0], "INTC": [0.04, -0.03, 0.0]}, index=dates)

    comove = signed_comove(rivals, dates)

    assert comove.tolist() == pytest.approx([0.03, -0.02, 0.0])
    assert comove.dtype == float


def test_signed_comove_needs_min_count_columns_on_the_day() -> None:
    """One name that happened to trade is an anecdote, and comes back as NaN."""
    dates = _dates()
    rivals = pd.DataFrame(
        {"AMD": [0.02, 0.02, np.nan], "INTC": [0.04, np.nan, np.nan]},
        index=dates,
    )

    comove = signed_comove(rivals, dates)

    assert comove.iloc[0] == pytest.approx(0.03)  # two reporters
    assert math.isnan(comove.iloc[1])  # one reporter, below MIN_RIVALS
    assert math.isnan(comove.iloc[2])  # none at all


def test_signed_comove_min_count_is_a_parameter() -> None:
    dates = _dates()
    rivals = pd.DataFrame({"AMD": [0.02, 0.02], "INTC": [0.04, np.nan]}, index=dates[:2])

    lenient = signed_comove(rivals, dates[:2], min_count=1)
    strict = signed_comove(rivals, dates[:2], min_count=3)

    assert lenient.iloc[1] == pytest.approx(0.02)
    assert math.isnan(strict.iloc[0])


def test_signed_comove_defaults_to_min_rivals() -> None:
    dates = _dates(1)
    one = pd.DataFrame({"AMD": [0.02]}, index=dates)

    assert MIN_RIVALS == 2
    assert math.isnan(signed_comove(one, dates).iloc[0])
    assert signed_comove(one, dates, min_count=MIN_RIVALS - 1).iloc[0] == pytest.approx(0.02)


def test_signed_comove_with_no_columns_is_all_nan() -> None:
    dates = _dates()

    comove = signed_comove(pd.DataFrame(index=dates), dates)

    assert comove.index.equals(dates)
    assert comove.isna().all()


def test_signed_comove_reindexes_onto_the_move_dates() -> None:
    """Names with a different calendar are aligned, not zipped positionally."""
    dates = _dates()
    rivals = pd.DataFrame(
        {"AMD": [0.02, 0.05], "INTC": [0.04, 0.05]},
        index=pd.DatetimeIndex([dates[0], pd.Timestamp("2025-01-02")], name="date"),
    )

    comove = signed_comove(rivals, dates)

    assert comove.iloc[0] == pytest.approx(0.03)
    assert comove.iloc[1:].isna().all()


def test_peer_comove_is_unchanged_by_the_shared_helper() -> None:
    """A single peer still averages: ``peer_comove`` keeps its v1 meaning."""
    dates = _dates()
    peers = pd.DataFrame({"AMD": [0.02, np.nan, -0.01]}, index=dates)

    comove = peer_comove(peers, dates)

    assert comove.iloc[0] == pytest.approx(0.02)
    assert math.isnan(comove.iloc[1])
    assert comove.iloc[2] == pytest.approx(-0.01)
    assert peer_comove(pd.DataFrame(index=dates), dates).isna().all()


# --------------------------------------------------------------------------
# sub_route — the two company branches
# --------------------------------------------------------------------------


def test_two_rivals_up_while_the_stock_falls_is_a_share_shift() -> None:
    """The worked case: -5% for us, +2% each for two rivals, from the frame up."""
    dates = _dates(1)
    rivals = pd.DataFrame({"AMD": [RIVAL_RISE], "INTC": [RIVAL_RISE]}, index=dates)

    rival_comove = float(signed_comove(rivals, dates).iloc[0])

    assert rival_comove == pytest.approx(RIVAL_RISE)
    assert sub_route("company", STOCK_DROP, rival_comove, None, None) == SUB_ROUTE_SHARE_SHIFT


def test_two_suppliers_down_with_the_stock_is_a_supply_chain() -> None:
    dates = _dates(1)
    suppliers = pd.DataFrame({"TSM": [SUPPLIER_DROP], "ASML": [SUPPLIER_DROP]}, index=dates)

    chain_comove = float(signed_comove(suppliers, dates).iloc[0])

    assert chain_comove == pytest.approx(SUPPLIER_DROP)
    assert sub_route("company", STOCK_DROP, None, chain_comove, None) == SUB_ROUTE_SUPPLY_CHAIN


def test_share_shift_is_symmetric_on_an_up_day() -> None:
    assert sub_route("company", 0.05, -0.02, None, None) == SUB_ROUTE_SHARE_SHIFT
    assert sub_route("company", 0.05, 0.02, None, None) is None


def test_supply_chain_is_symmetric_on_an_up_day() -> None:
    assert sub_route("company", 0.05, None, 0.03, None) == SUB_ROUTE_SUPPLY_CHAIN
    assert sub_route("company", 0.05, None, -0.03, None) is None


def test_rivals_moving_with_the_stock_is_not_a_share_shift() -> None:
    """The DeepSeek shape: everyone fell together, so nothing was taken."""
    assert sub_route("company", STOCK_DROP, -0.064, None, None) is None


def test_thresholds_are_fractions_of_the_move_and_inclusive() -> None:
    at_bar = -SHARE_SHIFT_RATIO * abs(STOCK_DROP) * -1.0  # rivals up by exactly the bar
    assert sub_route("company", STOCK_DROP, at_bar, None, None) == SUB_ROUTE_SHARE_SHIFT
    assert sub_route("company", STOCK_DROP, at_bar * 0.99, None, None) is None

    chain_bar = SUPPLY_CHAIN_RATIO * STOCK_DROP  # chain down by exactly the bar
    assert sub_route("company", STOCK_DROP, None, chain_bar, None) == SUB_ROUTE_SUPPLY_CHAIN
    assert sub_route("company", STOCK_DROP, None, chain_bar * 0.99, None) is None


def test_a_bigger_move_needs_a_bigger_comovement() -> None:
    """The same 1% rival move reads as a share shift against a small move only."""
    assert sub_route("company", -0.02, 0.01, None, None) == SUB_ROUTE_SHARE_SHIFT
    assert sub_route("company", -0.20, 0.01, None, None) is None


def test_share_shift_wins_the_tie_break() -> None:
    label = sub_route("company", STOCK_DROP, RIVAL_RISE, SUPPLIER_DROP, None)

    assert label == SUB_ROUTE_SHARE_SHIFT
    # ... and each half would have fired on its own.
    assert sub_route("company", STOCK_DROP, RIVAL_RISE, None, None) == SUB_ROUTE_SHARE_SHIFT
    assert sub_route("company", STOCK_DROP, None, SUPPLIER_DROP, None) == SUB_ROUTE_SUPPLY_CHAIN


def test_company_with_no_related_data_is_none() -> None:
    assert sub_route("company", STOCK_DROP, None, None, None) is None


def test_a_macro_driver_is_ignored_on_a_company_day() -> None:
    """``sub_routing`` never contradicts ``routing``: this day was not macro."""
    assert sub_route("company", STOCK_DROP, None, None, "oil") is None


# --------------------------------------------------------------------------
# sub_route — the macro branch and industry
# --------------------------------------------------------------------------


@pytest.mark.parametrize("driver", ["oil", "dollar", "rates", "gold", "country:TW"])
def test_macro_passes_the_driver_through_verbatim(driver: str) -> None:
    assert sub_route("macro", STOCK_DROP, None, None, driver) == driver


def test_macro_without_a_driver_is_none() -> None:
    assert sub_route("macro", STOCK_DROP, None, None, None) is None
    assert sub_route("macro", STOCK_DROP, None, None, "") is None


def test_macro_ignores_the_company_side_signals() -> None:
    assert sub_route("macro", STOCK_DROP, RIVAL_RISE, SUPPLIER_DROP, None) is None
    assert sub_route("macro", STOCK_DROP, RIVAL_RISE, SUPPLIER_DROP, "oil") == "oil"


@pytest.mark.parametrize(
    "rival,chain,driver",
    [
        (RIVAL_RISE, None, None),
        (None, SUPPLIER_DROP, None),
        (RIVAL_RISE, SUPPLIER_DROP, "oil"),
        (None, None, None),
    ],
)
def test_industry_never_gets_a_sub_bucket(
    rival: float | None,
    chain: float | None,
    driver: str | None,
) -> None:
    assert sub_route("industry", STOCK_DROP, rival, chain, driver) is None


def test_an_unknown_routing_bucket_is_none() -> None:
    assert sub_route("", STOCK_DROP, RIVAL_RISE, None, None) is None
    assert sub_route("unexplained", STOCK_DROP, RIVAL_RISE, None, None) is None


# --------------------------------------------------------------------------
# sub_route — missing and degenerate inputs
# --------------------------------------------------------------------------


def test_nan_comovements_count_as_missing() -> None:
    assert sub_route("company", STOCK_DROP, NAN, None, None) is None
    assert sub_route("company", STOCK_DROP, None, NAN, None) is None
    assert sub_route("company", STOCK_DROP, NAN, NAN, None) is None


def test_a_nan_rival_does_not_hide_a_supply_chain() -> None:
    """The two rules are independent: a missing rival average is not a veto."""
    assert sub_route("company", STOCK_DROP, NAN, SUPPLIER_DROP, None) == SUB_ROUTE_SUPPLY_CHAIN


def test_a_nan_from_the_frame_flows_through_as_missing() -> None:
    """The NaN ``signed_comove`` returns below ``MIN_RIVALS`` is the None case."""
    dates = _dates(1)
    one_rival = pd.DataFrame({"AMD": [RIVAL_RISE]}, index=dates)

    rival_comove = float(signed_comove(one_rival, dates).iloc[0])

    assert math.isnan(rival_comove)
    assert sub_route("company", STOCK_DROP, rival_comove, None, None) is None


def test_a_missing_return_yields_no_label() -> None:
    assert sub_route("company", NAN, RIVAL_RISE, SUPPLIER_DROP, None) is None


def test_a_flat_day_yields_no_label() -> None:
    """At ``ret == 0`` both tests reduce to ``0 <= 0``; that is noise, not a story."""
    assert sub_route("company", 0.0, RIVAL_RISE, None, None) is None
    assert sub_route("company", 0.0, None, SUPPLIER_DROP, None) is None
    assert sub_route("company", 0.0, 0.0, 0.0, None) is None


def test_a_nan_macro_driver_counts_as_missing() -> None:
    assert sub_route("macro", STOCK_DROP, None, None, NAN) is None
