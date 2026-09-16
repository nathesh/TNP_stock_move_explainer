"""Tests for DESIGN.md section 1 — prices, decomposition and move detection.

Everything runs against synthetic frames from ``tests/synth.py``: no network,
no database, no yfinance. A move is planted at a known index position and the
tests check that detection finds it, that the thresholds behave at both
extremes, and that the routing and proximity flags are exactly as specified.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from synth import synthetic_market, synthetic_ohlcv

from stock_moves.macro_calendar import CPI_DATES, FOMC_DATES
from stock_moves.moves import (
    FACTOR_COLUMNS,
    FEATURE_COLUMNS,
    OHLCV_COLUMNS,
    build_features,
    compute_returns,
    decompose,
    detect_moves,
    near_dates,
    near_earnings,
    peer_comove,
    regime,
    rolling_z,
    route,
    volume_z,
)

N_DAYS = 320
SHOCK_POSITION = 250
SHOCK_RETURN = -0.08
EARNINGS_POSITION = 100

# Roughly the number of trading days yfinance returns for `period="2y"`, which is
# the default since v1.5 (`Settings.default_period`).
TWO_YEAR_DAYS = 504
ONE_YEAR_DAYS = 252
# `regime`'s slow SMA window: the first `REGIME_SLOW_WINDOW - 1` rows are unlabelled.
REGIME_SLOW_WINDOW = 200


@pytest.fixture(scope="module")
def stock() -> pd.DataFrame:
    return synthetic_ohlcv(n_days=N_DAYS, seed=0, shocks={SHOCK_POSITION: SHOCK_RETURN})


@pytest.fixture(scope="module")
def spy(stock: pd.DataFrame) -> pd.DataFrame:
    return synthetic_market(N_DAYS, seed=1, correlate_with=stock["close"].pct_change(), beta=0.8)


@pytest.fixture(scope="module")
def etf(stock: pd.DataFrame) -> pd.DataFrame:
    return synthetic_market(N_DAYS, seed=2, correlate_with=stock["close"].pct_change(), beta=0.5)


@pytest.fixture(scope="module")
def features(stock: pd.DataFrame, spy: pd.DataFrame, etf: pd.DataFrame) -> pd.DataFrame:
    return build_features(
        stock,
        spy,
        etf,
        earnings_dates=[stock.index[EARNINGS_POSITION].date()],
    )


# --------------------------------------------------------------------------
# build_features
# --------------------------------------------------------------------------


def test_build_features_has_exactly_the_designed_columns(
    features: pd.DataFrame, stock: pd.DataFrame, spy: pd.DataFrame, etf: pd.DataFrame
) -> None:
    expected = list(OHLCV_COLUMNS) + list(FEATURE_COLUMNS) + list(FACTOR_COLUMNS)
    assert list(features.columns) == expected

    joined = stock.index.intersection(spy.index).intersection(etf.index)
    assert len(features) == len(joined)
    assert features.index.name == "date"
    assert features.index.is_monotonic_increasing
    assert features.index.tz is None


def test_build_features_populates_the_feature_columns(features: pd.DataFrame) -> None:
    # Warm-up rows are NaN by design; every column must have real values after it.
    for column in ("ret", "gap_ret", "intraday_ret", "ret_z", "vol_z"):
        assert features[column].notna().any(), column
    for column in ("mkt_component", "sector_component", "idio_component"):
        assert features[column].notna().any(), column

    assert features["routing"].iloc[0] is None  # no return on the first row
    assert set(features["routing"].dropna()) <= {"company", "industry", "macro"}
    assert set(features["regime_mkt"].dropna()) <= {"bull", "bear"}
    assert set(features["regime_sector"].dropna()) <= {"bull", "bear"}

    for column in ("near_earnings", "near_fomc", "near_cpi"):
        assert features[column].dtype == bool, column
    # The synthetic window spans 2025-01 to 2026-03, so both macro calendars fire.
    assert features["near_fomc"].sum() > 0
    assert features["near_cpi"].sum() > 0


def test_build_features_without_an_etf_leaves_regime_sector_none(
    stock: pd.DataFrame, spy: pd.DataFrame
) -> None:
    features = build_features(stock, spy, None)
    assert list(features.columns) == list(OHLCV_COLUMNS) + list(FEATURE_COLUMNS) + list(
        FACTOR_COLUMNS
    )
    assert features["regime_sector"].isna().all()
    assert features["regime_mkt"].notna().any()


def test_build_features_over_two_years_labels_the_regime_after_the_warm_up() -> None:
    """Why `Settings.default_period` is `2y`: a 1y window is mostly regime-less.

    `regime` needs a 200-day SMA, so the first 199 rows of any window carry no
    label at all. Over two years that warm-up is a fifth of the sample and the
    rest is labelled; over one year it is four fifths of it.
    """
    stock = synthetic_ohlcv(n_days=TWO_YEAR_DAYS, seed=5)
    stock_ret = stock["close"].pct_change()
    spy = synthetic_market(TWO_YEAR_DAYS, seed=6, correlate_with=stock_ret, beta=0.9)
    etf = synthetic_market(TWO_YEAR_DAYS, seed=7, correlate_with=stock_ret, beta=0.6)

    features = build_features(stock, spy, etf)
    assert len(features) == TWO_YEAR_DAYS

    regime_mkt = features["regime_mkt"]
    assert regime_mkt.iloc[: REGIME_SLOW_WINDOW - 1].isna().all()
    assert regime_mkt.iloc[REGIME_SLOW_WINDOW - 1 :].notna().all()
    assert set(regime_mkt.dropna()) <= {"bull", "bear"}
    assert features["regime_sector"].iloc[REGIME_SLOW_WINDOW - 1 :].notna().all()

    # Most of a 2y window is labelled; most of a 1y window is not.
    assert regime_mkt.notna().mean() > 0.5
    one_year = build_features(
        stock.iloc[:ONE_YEAR_DAYS], spy.iloc[:ONE_YEAR_DAYS], etf.iloc[:ONE_YEAR_DAYS]
    )
    assert one_year["regime_mkt"].notna().mean() < 0.25


# --------------------------------------------------------------------------
# detect_moves
# --------------------------------------------------------------------------


def test_detect_moves_finds_the_planted_shock(features: pd.DataFrame, stock: pd.DataFrame) -> None:
    moves = detect_moves(features)
    shock_date = stock.index[SHOCK_POSITION]

    assert shock_date in moves.index
    assert moves.loc[shock_date, "direction"] == "down"
    assert moves.loc[shock_date, "ret"] == pytest.approx(SHOCK_RETURN, abs=1e-9)
    assert set(moves["direction"].dropna()) <= {"up", "down"}

    # Sorted by abs(ret_z) descending, NaN last.
    magnitudes = moves["ret_z"].abs().to_numpy(dtype=float)
    ranked = magnitudes[~np.isnan(magnitudes)]
    assert np.all(np.diff(ranked) <= 1e-12)
    assert not np.isnan(magnitudes[: len(ranked)]).any()
    # An 8% single-day drop is the largest move in the sample.
    assert moves.index[0] == shock_date


def test_detect_moves_returns_nothing_at_absurd_thresholds(
    features: pd.DataFrame,
) -> None:
    moves = detect_moves(features, z_threshold=50.0, pct_threshold=0.5)
    assert moves.empty
    assert "direction" in moves.columns


def test_detect_moves_with_a_zero_pct_threshold_takes_every_return(
    features: pd.DataFrame,
) -> None:
    moves = detect_moves(features, pct_threshold=0.0)
    assert len(moves) == int(features["ret"].notna().sum())
    assert moves["ret"].notna().all()


# --------------------------------------------------------------------------
# route
# --------------------------------------------------------------------------


def test_route_picks_the_largest_absolute_component() -> None:
    assert route(0.01, 0.02, 0.005) == "industry"
    assert route(0.03, 0.0, 0.01) == "macro"
    assert route(0.001, 0.002, 0.05) == "company"
    assert route(-0.04, 0.01, 0.01) == "macro"  # sign is irrelevant


def test_route_falls_back_to_company_on_bad_or_tied_input() -> None:
    assert route(math.nan, 1, 1) == "company"
    assert route(1, math.nan, 1) == "company"
    assert route(1, 1, math.nan) == "company"
    assert route(None, 1, 1) == "company"
    assert route(0.01, 0.01, 0.01) == "company"  # exact three-way tie
    assert route(0.02, 0.02, 0.01) == "company"  # exact tie between leaders


# --------------------------------------------------------------------------
# returns and z-scores
# --------------------------------------------------------------------------


def test_compute_returns_matches_the_definitions(stock: pd.DataFrame) -> None:
    out = compute_returns(stock)
    assert math.isnan(out["ret"].iloc[0])
    assert math.isnan(out["gap_ret"].iloc[0])

    position = 40
    prev_close = stock["close"].iloc[position - 1]
    open_ = stock["open"].iloc[position]
    close = stock["close"].iloc[position]
    assert out["ret"].iloc[position] == pytest.approx(close / prev_close - 1.0)
    assert out["gap_ret"].iloc[position] == pytest.approx(open_ / prev_close - 1.0)
    assert out["intraday_ret"].iloc[position] == pytest.approx(close / open_ - 1.0)
    # The two legs compose back into the daily return.
    composed = (1 + out["gap_ret"].iloc[position]) * (1 + out["intraday_ret"].iloc[position]) - 1
    assert composed == pytest.approx(out["ret"].iloc[position])


def test_rolling_z_warms_up_and_survives_a_flat_window() -> None:
    window = 20
    values = pd.Series(np.linspace(1.0, 5.0, 60))
    z = rolling_z(values, window=window)
    assert z.iloc[:window].isna().all()
    assert z.iloc[window:].notna().all()

    flat = pd.Series([5.0] * 60)
    flat_z = rolling_z(flat, window=window)
    assert not np.isinf(flat_z.to_numpy(dtype=float)).any()
    assert flat_z.isna().all()  # zero dispersion is undefined, not infinite


def test_volume_z_is_a_multiple_of_the_trailing_mean() -> None:
    window = 20
    volume = pd.Series([1e6] * 30 + [5e6])
    z = volume_z(volume, window=window)
    assert z.iloc[:window].isna().all()
    assert z.iloc[25] == pytest.approx(1.0)
    assert z.iloc[30] == pytest.approx(5.0)


# --------------------------------------------------------------------------
# regime
# --------------------------------------------------------------------------


def test_regime_reads_a_trend_in_both_directions() -> None:
    rising = synthetic_ohlcv(n_days=N_DAYS, seed=3, daily_vol=0.001, drift=0.004)
    falling = synthetic_ohlcv(n_days=N_DAYS, seed=3, daily_vol=0.001, drift=-0.004)

    bull = regime(rising["close"])
    bear = regime(falling["close"])

    assert bull.iloc[-1] == "bull"
    assert bear.iloc[-1] == "bear"
    assert bull.iloc[0] is None  # SMA(200) undefined during warm-up
    assert bull.iloc[198] is None
    assert bull.iloc[199] == "bull"


def test_regime_labels_a_history_shorter_than_the_slow_window() -> None:
    short = synthetic_ohlcv(n_days=252, seed=4, daily_vol=0.001, drift=0.004)
    labels = regime(short["close"], fast=50, slow=400)
    # Fallback min_periods = max(50, 252 // 2) = 126, so the label starts there.
    assert labels.iloc[124] is None
    assert labels.iloc[125] == "bull"


# --------------------------------------------------------------------------
# event proximity
# --------------------------------------------------------------------------


def test_near_earnings_marks_only_the_adjacent_trading_days(
    stock: pd.DataFrame,
) -> None:
    flags = near_earnings(stock.index, [stock.index[EARNINGS_POSITION]])
    assert flags.dtype == bool

    hits = set(np.flatnonzero(flags.to_numpy()))
    assert hits == {EARNINGS_POSITION - 1, EARNINGS_POSITION, EARNINGS_POSITION + 1}


def test_near_dates_snaps_a_weekend_event_to_the_next_trading_day(
    stock: pd.DataFrame,
) -> None:
    saturday = pd.Timestamp("2025-03-08")
    assert saturday.dayofweek == 5
    assert saturday not in stock.index

    flags = near_dates(stock.index, [saturday.date()])
    monday = stock.index.get_loc(pd.Timestamp("2025-03-10"))
    hits = set(np.flatnonzero(flags.to_numpy()))
    # Snapped forward to Monday, then +/- one trading day: Friday, Monday, Tuesday.
    assert hits == {monday - 1, monday, monday + 1}
    assert stock.index[monday].dayofweek == 0


def test_near_dates_ignores_events_past_the_end_of_the_index(
    stock: pd.DataFrame,
) -> None:
    beyond = pd.Timestamp("2030-01-02").date()
    assert not near_dates(stock.index, [beyond]).any()
    assert not near_dates(stock.index, []).any()


def test_macro_calendar_is_a_plausible_published_schedule() -> None:
    for dates in (FOMC_DATES, CPI_DATES):
        assert len(dates) == len(set(dates))
        assert list(dates) == sorted(dates)
        assert all(d.weekday() < 5 for d in dates)  # releases are weekdays
    # Eight scheduled FOMC decisions per year.
    assert sum(d.year == 2025 for d in FOMC_DATES) == 8
    assert sum(d.year == 2026 for d in FOMC_DATES) == 8


# --------------------------------------------------------------------------
# peers and decomposition
# --------------------------------------------------------------------------


def test_peer_comove_averages_across_peers(stock: pd.DataFrame) -> None:
    dates = stock.index[:5]
    peers = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, np.nan, 0.0, -0.01],
            "BBB": [0.03, np.nan, np.nan, 0.04, -0.03],
        },
        index=dates,
    )
    comove = peer_comove(peers, dates)
    assert comove.iloc[0] == pytest.approx(0.02)
    assert comove.iloc[1] == pytest.approx(0.02)  # skips the missing peer
    assert math.isnan(comove.iloc[2])  # no peer data at all that day

    empty = peer_comove(pd.DataFrame(index=dates), dates)
    assert empty.isna().all()


def test_decompose_without_a_sector_zeroes_the_sector_component(
    stock: pd.DataFrame, spy: pd.DataFrame
) -> None:
    window = 60
    stock_ret = stock["close"].pct_change()
    market_ret = spy["close"].pct_change()
    out = decompose(stock_ret, market_ret, None, window=window)

    assert list(out.columns) == [
        "beta_mkt",
        "beta_sec",
        "alpha",
        "mkt_component",
        "sector_component",
        "idio_component",
    ]
    # First return is NaN, so the first full window ends at position window + 1.
    assert out["beta_mkt"].iloc[: window + 1].isna().all()
    warm = out.iloc[window + 1 :]
    assert warm["beta_mkt"].notna().all()
    assert (warm["sector_component"] == 0.0).all()
    assert (warm["beta_sec"] == 0.0).all()

    # idio = ret - mkt - sector, exactly.
    rebuilt = warm["mkt_component"] + warm["sector_component"] + warm["idio_component"]
    pd.testing.assert_series_equal(rebuilt, stock_ret.iloc[window + 1 :], check_names=False)


def test_decompose_recovers_a_known_market_beta(stock: pd.DataFrame, spy: pd.DataFrame) -> None:
    truth = 0.8  # spy was built as 0.8 * stock returns + variance-matched noise
    out = decompose(stock["close"].pct_change(), spy["close"].pct_change(), None)
    fitted = out["beta_mkt"].iloc[-1]
    assert not math.isnan(fitted)
    assert abs(fitted - truth) < 0.4


def test_decompose_uses_only_history_strictly_before_the_day(
    stock: pd.DataFrame, spy: pd.DataFrame
) -> None:
    window = 60
    stock_ret = stock["close"].pct_change()
    market_ret = spy["close"].pct_change()
    full = decompose(stock_ret, market_ret, None, window=window)

    # Corrupting day t's own return must not change day t's betas.
    tampered = stock_ret.copy()
    tampered.iloc[-1] = 5.0
    after = decompose(tampered, market_ret, None, window=window)
    assert after["beta_mkt"].iloc[-1] == pytest.approx(full["beta_mkt"].iloc[-1])
