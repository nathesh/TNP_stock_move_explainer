"""Price features and move detection — DESIGN.md section 1.

Pure functions over pandas DataFrames: no I/O, no network, no database. Every
frame here is indexed by a tz-naive ``DatetimeIndex`` named ``date``, sorted
ascending, with lower-case ``open``, ``high``, ``low``, ``close``, ``volume``
columns. The only sibling module imported is :mod:`stock_moves.macro_calendar`,
which is a static table rather than a dependency.

The point of the module is the routing decision: decompose a day's return into
market, sector and idiosyncratic pieces so the news layer knows whether to ask
about the company, its industry, or the macro tape. The decomposition is a
rough factor model, not a causal claim.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import date, datetime

import numpy as np
import pandas as pd

from .macro_calendar import CPI_DATES, FOMC_DATES

__all__ = [
    "FACTOR_COLUMNS",
    "FEATURE_COLUMNS",
    "MIN_RIVALS",
    "OHLCV_COLUMNS",
    "SHARE_SHIFT_RATIO",
    "SUB_ROUTE_SHARE_SHIFT",
    "SUB_ROUTE_SUPPLY_CHAIN",
    "SUPPLY_CHAIN_RATIO",
    "build_features",
    "compute_returns",
    "decompose",
    "detect_moves",
    "factor_betas",
    "macro_driver",
    "near_dates",
    "near_earnings",
    "peer_comove",
    "regime",
    "rolling_z",
    "route",
    "signed_comove",
    "sub_route",
    "volume_z",
]

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

#: Exactly the feature columns ``build_features`` appends to the OHLCV block.
FEATURE_COLUMNS: tuple[str, ...] = (
    "ret",
    "gap_ret",
    "intraday_ret",
    "ret_z",
    "vol_z",
    "mkt_component",
    "sector_component",
    "idio_component",
    "routing",
    "regime_mkt",
    "regime_sector",
    "near_earnings",
    "near_fomc",
    "near_cpi",
)

#: The two columns :func:`macro_driver` produces (v1.5), kept deliberately
#: *outside* :data:`FEATURE_COLUMNS`: that tuple is the v1 storage contract
#: `ingest` is written against, and the factor attribution is a second and
#: separate pass. ``build_features`` always appends these two to its output — so
#: the frame is one shape whether or not a factor frame was supplied — and task
#: R9 wires them into `ingest`'s storage groups, which is where they start being
#: persisted. Until then they are computed and returned but not stored.
FACTOR_COLUMNS: tuple[str, ...] = ("macro_driver", "macro_driver_component")

#: A factor is a candidate driver only if its own return that day is this many
#: trailing standard deviations from flat (v1.5 plan, decision 4). Without the
#: gate a large beta would nominate a driver out of an ordinary day's noise.
MACRO_DRIVER_Z_MIN = 1.5

#: Sub-routing thresholds (v1.5 plan, decision 6), both expressed as a fraction
#: of the move's own size so they scale with the day: a 1% fall needs rivals up
#: 0.25% to read as a share shift, a 10% fall needs them up 2.5%. The supply
#: chain bar is the higher of the two because "everyone in the chain fell
#: together" is a weaker claim than "the money went to a rival" and should not
#: be made on a sympathetic twitch.
SHARE_SHIFT_RATIO: float = 0.25
SUPPLY_CHAIN_RATIO: float = 0.5

#: How many related names must have traded that day before their mean return is
#: usable at all. One name moving is an anecdote; two is the smallest thing that
#: can be called a co-movement.
MIN_RIVALS: int = 2

#: The two company-side sub-buckets. The macro-side ones are not constants here:
#: they are whatever :func:`macro_driver` named (``oil``, ``dollar``, ``rates``,
#: ``gold``, ``country:XX``), passed through verbatim.
SUB_ROUTE_SHARE_SHIFT: str = "share_shift"
SUB_ROUTE_SUPPLY_CHAIN: str = "supply_chain"

_ROUTE_COMPANY = "company"
_ROUTE_INDUSTRY = "industry"
_ROUTE_MACRO = "macro"


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the three return columns to a copy of ``df``.

    ``ret`` is close-to-close, ``gap_ret`` is previous close to open (overnight
    news) and ``intraday_ret`` is open to close (in-session news). The split
    matters because a gap and an intraday slide point at different stories.
    """
    out = df.copy()
    close = out["close"].astype(float)
    open_ = out["open"].astype(float)
    out["ret"] = close.pct_change()
    out["gap_ret"] = open_ / close.shift(1) - 1.0
    out["intraday_ret"] = close / open_ - 1.0
    return out


def rolling_z(series: pd.Series, window: int = 20) -> pd.Series:
    """``series`` divided by the trailing standard deviation of ``series``.

    The window ends at ``t-1``: today is excluded from its own dispersion
    estimate, otherwise a large move inflates the denominator it is measured
    against. The first ``window`` positions are NaN, and a flat window (zero
    standard deviation) yields NaN rather than an infinity.
    """
    std = series.shift(1).rolling(window, min_periods=window).std()
    return series / std.where(std > 0.0)


def volume_z(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume as a multiple of its trailing ``window``-day mean, today excluded."""
    mean = volume.shift(1).rolling(window, min_periods=window).mean()
    return volume / mean.where(mean > 0.0)


def decompose(
    ret: pd.Series,
    ret_mkt: pd.Series,
    ret_sec: pd.Series | None,
    window: int = 60,
) -> pd.DataFrame:
    """Trailing OLS of ``ret`` on the market and sector returns.

    For each day ``t`` the regression (with intercept) is fitted on the
    ``window`` days ending at ``t-1``, so the betas applied to day ``t`` never
    saw day ``t``. The full ``window`` rows must be present and finite,
    otherwise the row is NaN.

    When ``ret_sec`` is None the regression is on the market alone and
    ``beta_sec`` / ``sector_component`` are 0.0 on every fitted row.

    Returns columns ``beta_mkt``, ``beta_sec``, ``alpha``, ``mkt_component``,
    ``sector_component``, ``idio_component`` on ``ret``'s index.
    """
    index = ret.index
    n = len(index)
    has_sec = ret_sec is not None

    y_all = ret.to_numpy(dtype=float)
    x_mkt = ret_mkt.reindex(index).to_numpy(dtype=float)
    x_sec = (
        ret_sec.reindex(index).to_numpy(dtype=float) if ret_sec is not None else np.full(n, np.nan)
    )

    beta_mkt = np.full(n, np.nan)
    beta_sec = np.full(n, np.nan)
    alpha = np.full(n, np.nan)

    for t in range(window, n):
        lo = t - window
        y = y_all[lo:t]
        xm = x_mkt[lo:t]
        ok = np.isfinite(y) & np.isfinite(xm)
        if has_sec:
            xs = x_sec[lo:t]
            ok &= np.isfinite(xs)
        rows = int(ok.sum())
        if rows < window:
            continue
        columns = [np.ones(rows), xm[ok]]
        if has_sec:
            columns.append(x_sec[lo:t][ok])
        design = np.column_stack(columns)
        coef, *_ = np.linalg.lstsq(design, y[ok], rcond=None)
        alpha[t] = coef[0]
        beta_mkt[t] = coef[1]
        beta_sec[t] = float(coef[2]) if has_sec else 0.0

    mkt_component = beta_mkt * x_mkt
    # Without a sector series the component is exactly 0.0 on fitted rows; going
    # through x_sec (all NaN) would poison it.
    sector_component = beta_sec * x_sec if has_sec else beta_sec.copy()
    idio_component = y_all - mkt_component - sector_component

    return pd.DataFrame(
        {
            "beta_mkt": beta_mkt,
            "beta_sec": beta_sec,
            "alpha": alpha,
            "mkt_component": mkt_component,
            "sector_component": sector_component,
            "idio_component": idio_component,
        },
        index=index,
    )


def route(
    mkt_component: float | None,
    sector_component: float | None,
    idio_component: float | None,
) -> str:
    """Name the dominant component of one day: macro, industry or company.

    The largest absolute component wins. Anything unusable — None, NaN, a
    non-numeric value, or an exact tie between the leaders — falls back to
    ``company``, the bucket whose news query is the most specific and therefore
    the least likely to fabricate a macro story out of noise.
    """
    values: list[float] = []
    for raw in (mkt_component, sector_component, idio_component):
        if raw is None:
            return _ROUTE_COMPANY
        try:
            value = abs(float(raw))
        except (TypeError, ValueError):
            return _ROUTE_COMPANY
        if math.isnan(value):
            return _ROUTE_COMPANY
        values.append(value)

    mkt, sector, idio = values
    if mkt > sector and mkt > idio:
        return _ROUTE_MACRO
    if sector > mkt and sector > idio:
        return _ROUTE_INDUSTRY
    return _ROUTE_COMPANY


def regime(close: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """``bull`` / ``bear`` from the fast versus slow simple moving average.

    ``bull`` when SMA(``fast``) >= SMA(``slow``), ``bear`` otherwise, and None
    wherever the slow average is undefined.

    With a full history the slow average uses ``min_periods=slow``, so the
    first ``slow-1`` rows are None. A history **shorter** than ``slow`` would
    otherwise be labelled nowhere at all, so the slow average then falls back to
    ``min_periods=max(fast, len(close) // 2)`` — a shorter but still meaningful
    trend baseline. The returned series is object dtype holding str or None.
    """
    n = len(close)
    slow_min = slow if n >= slow else max(fast, n // 2)
    fast_min = min(fast, slow_min)

    sma_fast = close.rolling(fast, min_periods=fast_min).mean()
    sma_slow = close.rolling(slow, min_periods=slow_min).mean()

    labels = pd.Series([None] * n, index=close.index, dtype=object)
    defined = sma_fast.notna() & sma_slow.notna()
    labels[defined & (sma_fast >= sma_slow)] = "bull"
    labels[defined & (sma_fast < sma_slow)] = "bear"
    return labels


def near_dates(
    index: pd.DatetimeIndex,
    event_dates: Iterable[date | datetime],
    tolerance: int = 1,
) -> pd.Series:
    """Flag trading days within ``tolerance`` **index positions** of an event.

    Proximity is counted in trading days, not calendar days, so a Friday event
    reaches the following Monday. An event that is not itself a trading day
    (a weekend or holiday announcement) is snapped forward to the first trading
    day at or after it — that is the session which can react to it. Events past
    the end of the index are ignored.
    """
    positions = pd.DatetimeIndex(index)
    n = len(positions)
    flags = np.zeros(n, dtype=bool)
    if n == 0:
        return pd.Series(flags, index=index, dtype=bool)

    normalised = positions.normalize()
    for event in event_dates:
        stamp = pd.Timestamp(event).normalize()
        at_or_after = int(normalised.searchsorted(stamp, side="left"))
        if at_or_after >= n:
            continue  # beyond the last trading day we hold
        lo = max(0, at_or_after - tolerance)
        hi = min(n - 1, at_or_after + tolerance)
        flags[lo : hi + 1] = True

    return pd.Series(flags, index=index, dtype=bool)


def near_earnings(
    index: pd.DatetimeIndex,
    earnings_dates: Iterable[date | datetime],
    tolerance: int = 1,
) -> pd.Series:
    """Earnings proximity — :func:`near_dates` applied to the earnings calendar."""
    return near_dates(index, earnings_dates, tolerance=tolerance)


def _mean_comove(
    returns: pd.DataFrame | None,
    dates: pd.DatetimeIndex,
    min_count: int,
) -> pd.Series:
    """Mean across ``returns``' columns per date, NaN below ``min_count`` reporters.

    The one arithmetic shared by :func:`peer_comove` and :func:`signed_comove`.
    At ``min_count=1`` the mask is a no-op — a date where nothing reported is
    already NaN — which is why the older function can sit on top of it unchanged.
    """
    if returns is None or returns.shape[1] == 0:
        return pd.Series(np.nan, index=dates, dtype=float)
    aligned = returns.reindex(dates)
    mean = aligned.mean(axis=1, skipna=True).astype(float)
    return mean.where(aligned.count(axis=1) >= min_count)


def peer_comove(peer_returns: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """Mean same-day peer return, the second and independent industry signal.

    ``peer_returns`` has one column of daily returns per peer. The mean skips
    missing peers; a date with no peer data at all is NaN, as is every date when
    there are no peers.
    """
    return _mean_comove(peer_returns, dates, min_count=1)


def signed_comove(
    other_returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    min_count: int = MIN_RIVALS,
) -> pd.Series:
    """Mean same-day return across a set of related names, or NaN if too thin.

    The average :func:`peer_comove` computes, with one rule added: a date where
    fewer than ``min_count`` columns reported is NaN rather than the mean of
    whatever did. :func:`sub_route` turns this number into a claim — *the money
    went to a rival*, *the whole chain fell* — and a claim of that shape should
    not rest on one name that happened to trade.

    The sign is kept, hence the name: the caller needs the direction relative to
    the stock, not the size of the co-movement.
    """
    return _mean_comove(other_returns, dates, min_count=min_count)


def factor_betas(
    ret: pd.Series,
    factors: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """Trailing univariate OLS beta of ``ret`` on each column of ``factors``.

    One separate regression per factor — ``ret`` on that factor alone, with an
    intercept — fitted on the ``window`` days ending at ``t-1``, the same
    look-ahead-free convention as :func:`decompose`. Univariate on purpose: the
    proxies are correlated (a dollar move and an oil move are not independent),
    and a joint fit on four collinear ETFs would hand back betas nobody can
    defend. The v1.5 plan calls this a heuristic and so does this docstring.

    A row is NaN until the full ``window`` of finite pairs is available, so an
    exposure is never asserted from a half-filled window. Columns and index come
    out exactly as ``factors.columns`` and ``ret.index``.
    """
    index = ret.index
    n = len(index)
    y_all = ret.to_numpy(dtype=float)
    names = [str(name) for name in factors.columns]

    columns: dict[str, np.ndarray] = {}
    for name, raw_name in zip(names, factors.columns, strict=True):
        x_all = factors[raw_name].reindex(index).to_numpy(dtype=float)
        betas = np.full(n, np.nan)
        for t in range(window, n):
            lo = t - window
            y = y_all[lo:t]
            x = x_all[lo:t]
            ok = np.isfinite(y) & np.isfinite(x)
            rows = int(ok.sum())
            if rows < window:
                continue
            design = np.column_stack([np.ones(rows), x[ok]])
            coef, *_ = np.linalg.lstsq(design, y[ok], rcond=None)
            betas[t] = coef[1]
        columns[name] = betas

    return pd.DataFrame(columns, index=index, columns=names, dtype=float)


def macro_driver(
    ret: pd.Series,
    factors: pd.DataFrame,
    betas: pd.DataFrame,
    z_window: int = 20,
    z_min: float = MACRO_DRIVER_Z_MIN,
) -> pd.DataFrame:
    """Name the factor that best accounts for each day, or nothing.

    Per day and per factor the contribution is ``beta[t] * factor_return[t]``:
    what the stock's measured exposure says that factor did to it. A factor is
    eligible only when the factor itself had an unusual day —
    ``abs(rolling_z(factor_return)[t]) >= z_min`` on a ``z_window`` history —
    and the driver is the eligible factor with the largest absolute
    contribution. When no factor moved, the driver is None: the honest answer on
    a quiet macro tape is that nothing macro happened, not the least quiet of
    four quiet things.

    ``ret`` supplies the index only; the attribution is entirely a statement
    about the factors and the betas already fitted to them.

    Returns ``macro_driver`` (object dtype, a factor name or None) and
    ``macro_driver_component`` (float, the signed contribution, NaN where there
    is no driver) on ``ret``'s index.
    """
    index = ret.index
    n = len(index)
    names = [str(name) for name in factors.columns if str(name) in set(betas.columns)]

    drivers: list[str | None] = [None] * n
    components = np.full(n, np.nan)

    if names:
        contributions = np.full((n, len(names)), np.nan)
        eligible = np.zeros((n, len(names)), dtype=bool)
        for j, name in enumerate(names):
            factor_ret = factors[name].reindex(index).astype(float)
            beta = betas[name].reindex(index).to_numpy(dtype=float)
            contribution = beta * factor_ret.to_numpy(dtype=float)
            z = rolling_z(factor_ret, window=z_window).to_numpy(dtype=float)
            contributions[:, j] = contribution
            eligible[:, j] = np.isfinite(contribution) & np.isfinite(z) & (np.abs(z) >= z_min)

        for t in range(n):
            row = eligible[t]
            if not row.any():
                continue
            magnitude = np.where(row, np.abs(contributions[t]), -np.inf)
            winner = int(np.argmax(magnitude))
            drivers[t] = names[winner]
            components[t] = float(contributions[t, winner])

    return pd.DataFrame(
        {
            "macro_driver": pd.Series(drivers, index=index, dtype=object),
            "macro_driver_component": pd.Series(components, index=index, dtype=float),
        },
        index=index,
    )


def _finite_or_none(value: float | None) -> float | None:
    """``value`` as a float, or None when it is missing, non-numeric or not finite."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def sub_route(
    routing: str,
    ret: float,
    rival_comove: float | None,
    chain_comove: float | None,
    macro_driver: str | None,
) -> str | None:
    """Name the sub-bucket of one move, or None (v1.5 plan, decision 6).

    ``routing`` keeps its three v1 buckets; this is the finer label stored
    beside it, and it is only ever a refinement — never a contradiction — of
    the bucket :func:`route` already chose.

    * ``company`` and rivals moved *against* the stock by at least
      :data:`SHARE_SHIFT_RATIO` of the move → ``share_shift``. The tape is
      saying the money went somewhere specific.
    * ``company`` and suppliers or customers moved *with* the stock by at least
      :data:`SUPPLY_CHAIN_RATIO` of the move → ``supply_chain``. The story is
      the chain the company sits in.
    * ``macro`` and :func:`macro_driver` named something → that name verbatim
      (``oil``, ``dollar``, ``rates``, ``gold``, ``country:XX``).
    * anything else → None. ``industry`` in particular has no sub-bucket: the
      sector was the answer, and there is nothing finer to say.

    ``share_shift`` wins when both company rules hold. A rival moving the other
    way is the more specific and the more falsifiable claim of the two, so it is
    the one worth printing.

    Pure, and deliberately unforgiving about missing inputs: None, NaN and a
    non-finite ``comove`` all mean "no signal", not "no effect". A ``ret`` of
    zero or NaN is treated the same way, because at ``ret == 0`` both company
    tests reduce to ``0 <= 0`` and would fire on any rivals at all — a rule
    carrying no information should not be allowed to produce a label.
    """
    if routing == _ROUTE_MACRO:
        return macro_driver if isinstance(macro_driver, str) and macro_driver else None
    if routing != _ROUTE_COMPANY:
        return None

    move = _finite_or_none(ret)
    if move is None or move == 0.0:
        return None
    sign = 1.0 if move > 0.0 else -1.0
    size = abs(move)

    rival = _finite_or_none(rival_comove)
    if rival is not None and rival * sign <= -SHARE_SHIFT_RATIO * size:
        return SUB_ROUTE_SHARE_SHIFT

    chain = _finite_or_none(chain_comove)
    if chain is not None and chain * sign >= SUPPLY_CHAIN_RATIO * size:
        return SUB_ROUTE_SUPPLY_CHAIN

    return None


def build_features(
    stock: pd.DataFrame,
    spy: pd.DataFrame,
    etf: pd.DataFrame | None,
    earnings_dates: Iterable[date] = (),
    z_window: int = 20,
    ols_window: int = 60,
    *,
    factors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble the per-day feature table stored in ``prices``.

    Inner-joins ``stock`` with ``spy`` (and ``etf`` when given) so every row has
    a market return to regress against. Output is the OHLCV block of ``stock``
    plus exactly :data:`FEATURE_COLUMNS` and then :data:`FACTOR_COLUMNS`.
    ``regime_sector`` is None throughout when ``etf`` is None; ``routing`` is
    None on rows with no return.

    ``factors`` is the optional frame of factor-proxy returns from
    :func:`stock_moves.prices.fetch_factor_returns`, one column per proxy. It is
    a **second and separate** attribution: the SPY-plus-sector regression that
    decides ``routing`` is untouched, and the factor pass only fills
    :data:`FACTOR_COLUMNS`. Factor rows are aligned to the joined trading
    days, so a proxy that does not trade on one of them simply has no return
    that day. Without ``factors`` — every v1 caller — the two columns are still
    present and empty, which is what keeps the stored schema one shape.
    """
    index = stock.index.intersection(spy.index)
    if etf is not None:
        index = index.intersection(etf.index)
    index = index.sort_values()

    stock_rows = stock.loc[index, list(OHLCV_COLUMNS)]
    market_close = spy.loc[index, "close"].astype(float)
    sector_close = etf.loc[index, "close"].astype(float) if etf is not None else None

    out = compute_returns(stock_rows)
    out["ret_z"] = rolling_z(out["ret"], window=z_window)
    out["vol_z"] = volume_z(out["volume"].astype(float), window=z_window)

    components = decompose(
        out["ret"],
        market_close.pct_change(),
        sector_close.pct_change() if sector_close is not None else None,
        window=ols_window,
    )
    out["mkt_component"] = components["mkt_component"]
    out["sector_component"] = components["sector_component"]
    out["idio_component"] = components["idio_component"]

    routing: list[str | None] = [
        None if pd.isna(day_ret) else route(mkt, sector, idio)
        for day_ret, mkt, sector, idio in zip(
            out["ret"].to_numpy(),
            out["mkt_component"].to_numpy(),
            out["sector_component"].to_numpy(),
            out["idio_component"].to_numpy(),
            strict=True,
        )
    ]
    out["routing"] = pd.Series(routing, index=index, dtype=object)

    out["regime_mkt"] = regime(market_close)
    out["regime_sector"] = (
        regime(sector_close)
        if sector_close is not None
        else pd.Series([None] * len(index), index=index, dtype=object)
    )

    out["near_earnings"] = near_earnings(index, earnings_dates)
    out["near_fomc"] = near_dates(index, FOMC_DATES)
    out["near_cpi"] = near_dates(index, CPI_DATES)

    if factors is not None and factors.shape[1] > 0:
        aligned = factors.reindex(index)
        attribution = macro_driver(
            out["ret"],
            aligned,
            factor_betas(out["ret"], aligned, window=ols_window),
            z_window=z_window,
        )
        out["macro_driver"] = attribution["macro_driver"]
        out["macro_driver_component"] = attribution["macro_driver_component"]
    else:
        out["macro_driver"] = pd.Series([None] * len(index), index=index, dtype=object)
        out["macro_driver_component"] = pd.Series(np.nan, index=index, dtype=float)

    return out[list(OHLCV_COLUMNS) + list(FEATURE_COLUMNS) + list(FACTOR_COLUMNS)]


def detect_moves(
    features: pd.DataFrame,
    z_threshold: float = 2.0,
    pct_threshold: float = 0.02,
) -> pd.DataFrame:
    """Select the major days: ``abs(ret_z) >= z_threshold`` OR ``abs(ret) >= pct_threshold``.

    The z test is the real definition — 2% is a big day for a utility and a
    quiet one for a small cap — but the percentage test is kept as a second,
    independent gate and is what still fires during the z warm-up, where
    ``ret_z`` is NaN and therefore never passes on its own.

    Adds ``direction`` and sorts by ``abs(ret_z)`` descending, NaN last.
    """
    abs_z = features["ret_z"].abs()
    abs_ret = features["ret"].abs()
    selected = features.loc[(abs_z >= z_threshold) | (abs_ret >= pct_threshold)].copy()

    returns = selected["ret"].to_numpy(dtype=float)
    direction = pd.Series(np.where(returns < 0.0, "down", "up"), index=selected.index, dtype=object)
    direction[selected["ret"].isna()] = None
    selected["direction"] = direction

    # argsort on the negated magnitude: descending, and NaN always sorts last.
    order = np.argsort(-selected["ret_z"].abs().to_numpy(dtype=float), kind="stable")
    return selected.iloc[order]
