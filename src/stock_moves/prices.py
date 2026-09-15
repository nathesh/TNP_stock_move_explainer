"""Price and metadata fetch (DESIGN sections 1 and 2 inputs).

Everything that touches `yfinance` lives here, behind module-level functions so
other modules can do `from stock_moves import prices; prices.fetch_ohlcv(...)`
and monkeypatch a single name in tests. `normalize_ohlcv` and `sector_etf_for`
are pure and carry the logic worth unit-testing without the network.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd
import yfinance as yf

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


def fetch_ohlcv(ticker: str, period: str = "1y") -> pd.DataFrame:
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


def fetch_peer_returns(peers: Sequence[str], period: str = "1y") -> pd.DataFrame:
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
        columns[peer] = frame["close"].pct_change()

    if not columns:
        return pd.DataFrame(index=pd.DatetimeIndex([], name=DATE_INDEX_NAME))

    returns = pd.concat(columns, axis=1)
    returns.index.name = DATE_INDEX_NAME
    return returns.sort_index()


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
