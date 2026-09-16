"""Price and metadata fetch (DESIGN sections 1 and 2 inputs).

Everything that touches `yfinance` lives here, behind module-level functions so
other modules can do `from stock_moves import prices; prices.fetch_ohlcv(...)`
and monkeypatch a single name in tests. `normalize_ohlcv` and `sector_etf_for`
are pure and carry the logic worth unit-testing without the network.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

MARKET_ETF = "SPY"

SECTOR_ETF: dict[str, str] = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

#: Tradable proxies for the four macro factors (v1.5 plan, decision 4). Factor
#: exposure is measured from prices, not asked of a model, so each factor needs
#: a liquid ETF whose daily return *is* the factor's return for our purposes.
FACTOR_ETFS: dict[str, str] = {
    "oil": "USO",
    "dollar": "UUP",
    "rates": "TLT",
    "gold": "GLD",
}

#: One country ETF per ISO-3166 alpha-2 code a `country` edge can point at.
#: A company exposed to a country outside this dict keeps the edge but gets no
#: `country:XX` driver — the plan names that limitation rather than hiding it.
COUNTRY_ETFS: dict[str, str] = {
    "TW": "EWT",
    "CN": "FXI",
    "JP": "EWJ",
    "KR": "EWY",
    "IN": "INDA",
    "EU": "VGK",
    "MX": "EWW",
    "CA": "EWC",
    "GB": "EWU",
    "DE": "EWG",
}

#: Prefix that marks a country column in a factor-return frame.
COUNTRY_FACTOR_PREFIX = "country:"

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
DATE_INDEX_NAME = "date"


class PriceFetchError(RuntimeError):
    """Raised when a ticker has no usable price data."""


@dataclass(frozen=True)
class TickerInfo:
    """Company metadata from `yfinance` `info`, plus the mapped sector ETF."""

    ticker: str
    name: str
    sector: str | None
    industry: str | None
    sector_etf: str | None


def sector_etf_for(sector: str | None) -> str | None:
    """Map a `yfinance` sector string to its sector ETF, or None if unknown."""
    if not sector:
        return None
    return SECTOR_ETF.get(sector.strip())


def _empty_ohlcv() -> pd.DataFrame:
    frame = pd.DataFrame(
        {column: pd.Series(dtype="float64") for column in OHLCV_COLUMNS},
        index=pd.DatetimeIndex([], name=DATE_INDEX_NAME),
    )
    return frame


def normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn a `yfinance` price frame into the canonical daily OHLCV frame.

    Accepts a `Ticker.history` frame or a `download` frame with MultiIndex
    columns (level 0 is taken). Returns lower-cased open/high/low/close/volume
    on a tz-naive DatetimeIndex named "date", normalized to midnight, sorted
    ascending, duplicate dates collapsed (last wins), rows with a NaN close
    dropped, all values float.

    Raises PriceFetchError if a non-empty frame is missing OHLCV columns.
    """
    if raw is None or len(raw.columns) == 0 or len(raw.index) == 0:
        return _empty_ohlcv()

    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = pd.Index([str(column).strip().lower() for column in frame.columns])
    frame = frame.loc[:, ~frame.columns.duplicated(keep="first")]

    missing = [column for column in OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise PriceFetchError(f"price frame is missing columns: {', '.join(missing)}")
    frame = frame.loc[:, list(OHLCV_COLUMNS)]

    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    frame.index = index.normalize()
    frame.index.name = DATE_INDEX_NAME

    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.astype("float64")
    frame = frame[frame["close"].notna()]
    return frame


def fetch_ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch daily OHLCV for one ticker and normalize it.

    `auto_adjust=False` keeps the raw closes, so returns are the close-to-close
    returns a reader would see quoted.
    """
    try:
        raw = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    except Exception as exc:  # network, rate limit, bad symbol
        raise PriceFetchError(f"no price data for {ticker}") from exc

    if raw is None or len(raw.index) == 0:
        raise PriceFetchError(f"no price data for {ticker}")

    frame = normalize_ohlcv(raw)
    if frame.empty:
        raise PriceFetchError(f"no price data for {ticker}")
    return frame


def fetch_info(ticker: str) -> TickerInfo:
    """Fetch company metadata; degrade to a bare TickerInfo on any failure."""
    try:
        info = yf.Ticker(ticker).info or {}
        name = info.get("shortName") or info.get("longName") or ticker
        sector = info.get("sector") or None
        industry = info.get("industry") or None
        return TickerInfo(
            ticker=ticker,
            name=str(name),
            sector=sector,
            industry=industry,
            sector_etf=sector_etf_for(sector),
        )
    except Exception:  # noqa: BLE001 - metadata is optional; degrade, never fail ingest
        return TickerInfo(ticker=ticker, name=ticker, sector=None, industry=None, sector_etf=None)


def fetch_earnings_dates(ticker: str, limit: int = 12) -> list[date]:
    """Fetch earnings dates as sorted unique dates; [] when unavailable.

    Future dates are kept — the caller decides whether to ignore them.
    """
    try:
        handle = yf.Ticker(ticker)
        frame: pd.DataFrame | None = None
        getter = getattr(handle, "get_earnings_dates", None)
        if callable(getter):
            frame = getter(limit=limit)
        if frame is None:
            frame = getattr(handle, "earnings_dates", None)
        if frame is None or len(frame.index) == 0:
            return []

        index = pd.DatetimeIndex(pd.to_datetime(frame.index))
        if getattr(index, "tz", None) is not None:
            index = index.tz_localize(None)
        return sorted({stamp.date() for stamp in index if pd.notna(stamp)})
    except Exception:  # noqa: BLE001 - earnings are missing or malformed for many tickers
        return []


def _close_returns(frame: pd.DataFrame) -> pd.Series:
    """Close-to-close daily returns of one normalized OHLCV frame.

    The one definition of a daily return outside `moves.compute_returns`, shared
    by the peer and factor fetchers so a peer's return and a factor's return are
    computed the same way.
    """
    return frame["close"].astype(float).pct_change()


def _returns_frame(columns: dict[str, pd.Series]) -> pd.DataFrame:
    """Assemble named return series into the canonical returns frame."""
    if not columns:
        return pd.DataFrame(index=pd.DatetimeIndex([], name=DATE_INDEX_NAME))

    returns = pd.concat(columns, axis=1)
    returns.index.name = DATE_INDEX_NAME
    return returns.sort_index()


def fetch_peer_returns(peers: Sequence[str], period: str = "2y") -> pd.DataFrame:
    """Daily close-to-close returns for each peer, one column per peer.

    Peers that fail to fetch are skipped; an empty frame comes back when none
    succeed, so peer co-movement is simply absent rather than fatal.
    """
    columns: dict[str, pd.Series] = {}
    for peer in dict.fromkeys(symbol.strip() for symbol in peers):
        if not peer:
            continue
        try:
            frame = fetch_ohlcv(peer, period=period)
        except PriceFetchError:
            continue
        columns[peer] = _close_returns(frame)

    return _returns_frame(columns)


def factor_column_for(name: str) -> str:
    """The frame column a factor key is stored under.

    A country key — a code in :data:`COUNTRY_ETFS`, in any case, or a key that
    already carries the prefix — becomes ``country:XX``, so a driver name is
    self-describing wherever it travels (`sub_routing`, the gate rule, the
    prose). Every other key is stored verbatim, which is what the four names in
    :data:`FACTOR_ETFS` want.
    """
    key = name.strip()
    if key.lower().startswith(COUNTRY_FACTOR_PREFIX):
        return COUNTRY_FACTOR_PREFIX + key[len(COUNTRY_FACTOR_PREFIX) :].strip().upper()
    if key.upper() in COUNTRY_ETFS:
        return COUNTRY_FACTOR_PREFIX + key.upper()
    return key


def fetch_factor_returns(names: Mapping[str, str], period: str = "2y") -> pd.DataFrame:
    """Daily close-to-close returns of the factor proxies, one column per key.

    `names` maps a factor key to its proxy symbol — :data:`FACTOR_ETFS` plus the
    :data:`COUNTRY_ETFS` entries for the countries a company has an edge to.
    Country keys are stored as ``country:XX`` columns (see
    :func:`factor_column_for`); the four macro keys keep their own names.

    A proxy that fails to fetch is dropped with a logged warning and never
    raises: a missing factor means one fewer candidate driver, which is a worse
    attribution, not a failed ingest. The frame is empty when every symbol fails
    or `names` is empty, and :func:`stock_moves.moves.macro_driver` reads that as
    "no driver", not as an error.
    """
    columns: dict[str, pd.Series] = {}
    for raw_name, raw_symbol in names.items():
        column = factor_column_for(str(raw_name))
        symbol = str(raw_symbol).strip()
        if not column or not symbol or column in columns:
            continue
        try:
            frame = fetch_ohlcv(symbol, period=period)
        except PriceFetchError:
            logger.warning("factor proxy %s (%s) unavailable; dropped", column, symbol)
            continue
        columns[column] = _close_returns(frame)

    return _returns_frame(columns)


def fetch_etf_holdings(etf: str, top_n: int = 10) -> list[str]:
    """Top holdings of an ETF, used as the keyless peer fallback (DESIGN 2).

    Returns [] on any failure or if this `yfinance` build has no funds data.
    """
    try:
        funds_data = getattr(yf.Ticker(etf), "funds_data", None)
        if funds_data is None:
            return []
        holdings = getattr(funds_data, "top_holdings", None)
        if holdings is None or len(holdings.index) == 0:
            return []

        if isinstance(holdings, pd.DataFrame) and "Symbol" in holdings.columns:
            raw_symbols = list(holdings["Symbol"])
        else:
            raw_symbols = list(holdings.index)

        symbols = [str(symbol).strip().upper() for symbol in raw_symbols]
        return [symbol for symbol in dict.fromkeys(symbols) if symbol][:top_n]
    except Exception:  # noqa: BLE001 - funds data is absent in older yfinance builds
        return []
