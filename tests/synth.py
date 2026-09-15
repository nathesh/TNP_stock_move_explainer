"""Deterministic synthetic OHLCV frames for the move-detection tests.

Plain module, not a package: pytest's default import mode puts ``tests/`` on
``sys.path``, so ``from synth import synthetic_ohlcv`` works from a test module
in the same directory. Nothing here touches the network, which is the point —
the whole of DESIGN.md section 1 is a pure function over a frame, so it can be
tested against a frame we made up.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

__all__ = ["synthetic_market", "synthetic_ohlcv"]

_BASE_PRICE = 100.0
_BASE_VOLUME = 1e6


def _ohlcv_from_returns(
    returns: np.ndarray,
    index: pd.DatetimeIndex,
    rng: np.random.Generator,
    daily_vol: float,
) -> pd.DataFrame:
    """Build a consistent OHLCV frame from a close-to-close return path."""
    n = len(returns)
    close = _BASE_PRICE * np.cumprod(1.0 + returns)
    prev_close = np.concatenate(([_BASE_PRICE], close[:-1]))
    open_ = prev_close * (1.0 + rng.normal(0.0, daily_vol / 4.0, n))

    upper = np.maximum(open_, close)
    lower = np.minimum(open_, close)
    high = upper * (1.0 + np.abs(rng.normal(0.0, daily_vol / 2.0, n)))
    low = lower * (1.0 - np.abs(rng.normal(0.0, daily_vol / 2.0, n)))
    volume = rng.lognormal(math.log(_BASE_VOLUME), 0.25, n)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def synthetic_ohlcv(
    n_days: int = 320,
    seed: int = 0,
    daily_vol: float = 0.01,
    drift: float = 0.0003,
    shocks: dict[int, float] | None = None,
    start: str = "2025-01-02",
) -> pd.DataFrame:
    """A geometric random walk as a daily OHLCV frame.

    The index is a business-day ``DatetimeIndex`` named ``date`` (tz-naive,
    ascending). ``shocks`` maps an index position to the close-to-close return
    forced on that day, which is how a test plants a move it can then look for.
    Output is fully determined by ``seed``.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start=start, periods=n_days, name="date")

    returns = rng.normal(drift, daily_vol, n_days)
    returns[0] = 0.0  # no prior close, so no first-day return
    for position, forced in (shocks or {}).items():
        if 0 <= position < n_days:
            returns[position] = forced

    return _ohlcv_from_returns(returns, index, rng, daily_vol)


def synthetic_market(
    n_days: int,
    seed: int,
    correlate_with: pd.Series | None = None,
    beta: float = 0.8,
    daily_vol: float = 0.01,
    start: str = "2025-01-02",
) -> pd.DataFrame:
    """An index/ETF frame whose returns carry a known relationship to a stock.

    With ``correlate_with`` given (a series of daily returns), this frame's
    returns are ``beta * those returns + noise``. The noise scale is set to
    ``std * sqrt(1 - beta**2)`` so the two series end up with the same variance
    and the OLS slope of the *stock* on this market recovers ``beta`` — which is
    the direction :func:`stock_moves.moves.decompose` regresses in, so the test
    has a truth to compare its ``beta_mkt`` against.
    """
    rng = np.random.default_rng(seed)

    if correlate_with is None:
        index = pd.bdate_range(start=start, periods=n_days, name="date")
        returns = rng.normal(0.0, daily_vol, n_days)
    else:
        index = pd.DatetimeIndex(correlate_with.index[:n_days], name="date")
        base = np.nan_to_num(correlate_with.to_numpy(dtype=float)[:n_days], nan=0.0, copy=True)
        noise_std = float(np.std(base)) * math.sqrt(max(0.0, 1.0 - beta * beta))
        returns = beta * base + rng.normal(0.0, noise_std, len(base))

    returns[0] = 0.0
    return _ohlcv_from_returns(returns, index, rng, daily_vol)
