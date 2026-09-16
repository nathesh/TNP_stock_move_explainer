"""Tests for the factor attribution — v1.5 plan, decision 4.

The factor pass answers a narrower question than ``routing``: given that a day
was a macro day, *which* macro thing was it. Two rules carry the whole design,
and both are tested here against deterministic square waves rather than random
walks, so a beta is an exact number and an eligibility decision is a fact rather
than a probability:

* a factor's contribution to a day is ``beta * that factor's return``, and
* a factor is only a candidate when the factor itself had an unusual day.

The second rule is the one that matters. Without it the proxy with the largest
beta would be named the driver of every ordinary Tuesday.

No network, no database — ``build_features`` is fed synthetic frames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from synth import synthetic_market, synthetic_ohlcv

from stock_moves.moves import (
    FACTOR_COLUMNS,
    FEATURE_COLUMNS,
    OHLCV_COLUMNS,
    build_features,
    factor_betas,
    macro_driver,
    rolling_z,
)

N_DAYS = 200
OLS_WINDOW = 60
Z_WINDOW = 20
SHOCK_POSITION = 120
QUIET_POSITION = 100

#: The amplitude of the ordinary days, and the size of the one unusual day.
BASE = 0.004
SHOCK = 0.02

#: `build_features` returns the OHLCV block, then the v1 features, then the two
#: v1.5 factor columns — present on every call, filled only when factors are given.
_EXPECTED_COLUMNS = list(OHLCV_COLUMNS) + list(FEATURE_COLUMNS) + list(FACTOR_COLUMNS)


def _index(n: int = N_DAYS) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.bdate_range(start="2025-01-02", periods=n), name="date")


def _square_wave(n: int, period: int, amplitude: float) -> np.ndarray:
    """``+amplitude`` for the first half of each period, ``-amplitude`` for the second.

    Two waves whose periods are 2 and 4 are exactly uncorrelated over any window
    that is a multiple of 4 — the 60-day OLS window is — so a univariate beta
    fitted on one of them recovers its coefficient exactly, with no sampling
    noise for a test to tolerate. Each wave's rolling z-score is a constant
    ``1 / sqrt(20/19)`` ≈ 0.97, comfortably under the 1.5 eligibility bar, so
    *nothing* is eligible unless a test injects a shock.
    """
    positions = np.arange(n)
    return amplitude * np.where((positions % period) < period // 2, 1.0, -1.0)


def _factors(shock_at: int | None = SHOCK_POSITION) -> pd.DataFrame:
    """Two proxies: ``oil`` carries the injected shock, ``dollar`` never moves unusually."""
    oil = _square_wave(N_DAYS, 2, BASE)
    if shock_at is not None:
        oil[shock_at] = SHOCK
    dollar = _square_wave(N_DAYS, 4, BASE)
    return pd.DataFrame({"oil": oil, "dollar": dollar}, index=_index())


# --------------------------------------------------------------------------
# The eligibility premise the rest of the module rests on
# --------------------------------------------------------------------------


def test_only_the_shock_day_clears_the_eligibility_bar() -> None:
    factors = _factors()
    z = rolling_z(factors["oil"], window=Z_WINDOW)

    assert abs(z.iloc[SHOCK_POSITION]) >= 1.5
    assert z.iloc[SHOCK_POSITION] > 4.0  # the injected day is a 4-sigma day
    quiet = z.drop(z.index[SHOCK_POSITION]).abs().dropna()
    assert (quiet < 1.5).all()
    assert rolling_z(factors["dollar"], window=Z_WINDOW).abs().dropna().max() < 1.5


# --------------------------------------------------------------------------
# factor_betas
# --------------------------------------------------------------------------


def test_factor_betas_recovers_a_unit_beta() -> None:
    factors = _factors()
    ret = pd.Series(factors["oil"].to_numpy(dtype=float), index=factors.index)

    betas = factor_betas(ret, factors, window=OLS_WINDOW)

    assert list(betas.columns) == ["oil", "dollar"]
    # The window ends at t-1, so the shock day's own beta was fitted without it.
    assert betas["oil"].iloc[SHOCK_POSITION] == pytest.approx(1.0)
    assert betas["oil"].iloc[QUIET_POSITION] == pytest.approx(1.0)
    assert betas["dollar"].iloc[QUIET_POSITION] == pytest.approx(0.0, abs=1e-9)


def test_factor_betas_are_nan_until_the_window_is_full() -> None:
    factors = _factors()
    ret = pd.Series(factors["oil"].to_numpy(dtype=float), index=factors.index)

    betas = factor_betas(ret, factors, window=OLS_WINDOW)

    assert betas.iloc[:OLS_WINDOW].isna().all().all()
    assert betas.iloc[OLS_WINDOW:].notna().all().all()
    assert betas.index.equals(factors.index)


def test_factor_betas_of_an_empty_frame_is_empty() -> None:
    index = _index()
    ret = pd.Series(0.0, index=index)

    betas = factor_betas(ret, pd.DataFrame(index=index), window=OLS_WINDOW)

    assert list(betas.columns) == []
    assert betas.index.equals(index)


# --------------------------------------------------------------------------
# macro_driver
# --------------------------------------------------------------------------


def test_macro_driver_names_the_shocked_factor_and_nothing_else() -> None:
    factors = _factors()
    ret = pd.Series(factors["oil"].to_numpy(dtype=float), index=factors.index)
    betas = factor_betas(ret, factors, window=OLS_WINDOW)

    attribution = macro_driver(ret, factors, betas, z_window=Z_WINDOW)

    assert list(attribution.columns) == list(FACTOR_COLUMNS)
    assert attribution["macro_driver"].iloc[SHOCK_POSITION] == "oil"
    # beta 1 on a 2% factor day is a 2% contribution.
    assert attribution["macro_driver_component"].iloc[SHOCK_POSITION] == pytest.approx(SHOCK)

    # A quiet day gets no driver: nothing macro happened, so nothing is named.
    assert attribution["macro_driver"].iloc[QUIET_POSITION] is None
    assert pd.isna(attribution["macro_driver_component"].iloc[QUIET_POSITION])
    assert attribution["macro_driver"].notna().sum() == 1


def test_macro_driver_never_names_a_factor_that_stayed_quiet() -> None:
    """A large beta is not a reason. The factor has to have moved."""
    factors = _factors()
    # Ten times the exposure to `dollar`, which never clears the bar, and unit
    # exposure to `oil`, which clears it exactly once.
    ret = pd.Series(
        factors["oil"].to_numpy(dtype=float) + 10.0 * factors["dollar"].to_numpy(dtype=float),
        index=factors.index,
    )
    betas = factor_betas(ret, factors, window=OLS_WINDOW)
    attribution = macro_driver(ret, factors, betas, z_window=Z_WINDOW)

    # The quiet factor's contribution is the larger one every single day...
    assert betas["dollar"].iloc[SHOCK_POSITION] == pytest.approx(10.0)
    dollar_contribution = abs(10.0 * BASE)
    assert dollar_contribution > abs(SHOCK)
    # ...and it is still never chosen.
    assert "dollar" not in set(attribution["macro_driver"].dropna())
    assert attribution["macro_driver"].iloc[SHOCK_POSITION] == "oil"


def test_macro_driver_is_empty_without_factors_or_betas() -> None:
    index = _index()
    ret = pd.Series(0.0, index=index)
    empty = pd.DataFrame(index=index)

    attribution = macro_driver(ret, empty, empty, z_window=Z_WINDOW)

    assert attribution["macro_driver"].isna().all()
    assert attribution["macro_driver_component"].isna().all()

    # A factor with no fitted beta is not a candidate either.
    factors = _factors()
    orphaned = macro_driver(ret, factors, pd.DataFrame(index=index), z_window=Z_WINDOW)
    assert orphaned["macro_driver"].isna().all()


# --------------------------------------------------------------------------
# build_features
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stock() -> pd.DataFrame:
    return synthetic_ohlcv(n_days=N_DAYS, seed=3)


@pytest.fixture(scope="module")
def spy(stock: pd.DataFrame) -> pd.DataFrame:
    return synthetic_market(N_DAYS, seed=4, correlate_with=stock["close"].pct_change(), beta=0.8)


def test_build_features_without_factors_leaves_both_columns_empty(
    stock: pd.DataFrame, spy: pd.DataFrame
) -> None:
    features = build_features(stock, spy, None)

    assert list(features.columns) == _EXPECTED_COLUMNS
    # The factor columns sit outside FEATURE_COLUMNS, which stays the v1 storage
    # contract `ingest` is written against until task R9 extends it.
    assert not set(FACTOR_COLUMNS) & set(FEATURE_COLUMNS)
    assert features["macro_driver"].isna().all()
    assert all(value is None for value in features["macro_driver"])
    assert features["macro_driver_component"].isna().all()


def test_build_features_with_factors_fills_the_driver_columns(
    stock: pd.DataFrame, spy: pd.DataFrame
) -> None:
    factors = _factors().reindex(stock.index)

    features = build_features(stock, spy, None, factors=factors)

    assert list(features.columns) == _EXPECTED_COLUMNS
    day = stock.index[SHOCK_POSITION]
    assert features.loc[day, "macro_driver"] == "oil"
    assert pd.notna(features.loc[day, "macro_driver_component"])
    assert features["macro_driver"].notna().sum() == 1
    # The factor pass is a second attribution: routing is untouched by it.
    baseline = build_features(stock, spy, None)
    assert features["routing"].equals(baseline["routing"])
    assert features["idio_component"].equals(baseline["idio_component"])
