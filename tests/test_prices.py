"""Tests for stock_moves.prices. No network: `yfinance` is monkeypatched."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from stock_moves import prices
from stock_moves.prices import (
    COUNTRY_ETFS,
    FACTOR_ETFS,
    MARKET_ETF,
    SECTOR_ETF,
    PriceFetchError,
    TickerInfo,
    factor_column_for,
    fetch_earnings_dates,
    fetch_etf_holdings,
    fetch_factor_returns,
    fetch_info,
    fetch_ohlcv,
    fetch_peer_returns,
    normalize_ohlcv,
    sector_etf_for,
)

TZ = "America/New_York"


def _raw_history() -> pd.DataFrame:
    """A yfinance-shaped history frame: tz-aware index, capitalised columns,
    a duplicate date (2026-01-06 twice) and a NaN close (2026-01-07)."""
    index = pd.DatetimeIndex(
        [
            "2026-01-05 00:00:00",
            "2026-01-06 00:00:00",
            "2026-01-06 00:00:00",
            "2026-01-07 00:00:00",
            "2026-01-02 00:00:00",
        ],
        name="Date",
    ).tz_localize(TZ)
    return pd.DataFrame(
        {
            "Open": [10.0, 11.0, 11.5, 12.0, 9.0],
            "High": [10.5, 11.8, 11.9, 12.5, 9.4],
            "Low": [9.8, 10.9, 11.4, 11.8, 8.9],
            "Close": [10.2, 11.1, 11.7, float("nan"), 9.1],
            "Volume": [1000, 2000, 2500, 3000, 500],
            "Dividends": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Stock Splits": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )


def test_normalize_ohlcv_shape_and_index() -> None:
    frame = normalize_ohlcv(_raw_history())

    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame.index.name == "date"
    assert frame.index.tz is None
    assert list(frame.index) == [
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-05"),
        pd.Timestamp("2026-01-06"),
    ]
    assert frame.index.is_monotonic_increasing
    # all bars sit at midnight
    assert (frame.index == frame.index.normalize()).all()


def test_normalize_ohlcv_duplicate_keeps_last_and_drops_nan_close() -> None:
    frame = normalize_ohlcv(_raw_history())

    assert not frame.index.has_duplicates
    assert frame.loc[pd.Timestamp("2026-01-06"), "close"] == pytest.approx(11.7)
    assert pd.Timestamp("2026-01-07") not in frame.index
    assert frame["close"].notna().all()


def test_normalize_ohlcv_volume_is_float() -> None:
    frame = normalize_ohlcv(_raw_history())

    assert frame["volume"].dtype == "float64"
    assert frame.loc[pd.Timestamp("2026-01-02"), "volume"] == pytest.approx(500.0)


def test_normalize_ohlcv_handles_multiindex_columns() -> None:
    raw = _raw_history()[["Open", "High", "Low", "Close", "Volume"]]
    raw.columns = pd.MultiIndex.from_product(
        [list(raw.columns), ["AAPL"]], names=["Price", "Ticker"]
    )

    frame = normalize_ohlcv(raw)

    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert len(frame) == 3


def test_normalize_ohlcv_empty_frame_is_canonical_and_empty() -> None:
    frame = normalize_ohlcv(pd.DataFrame())

    assert frame.empty
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame.index.name == "date"


def test_normalize_ohlcv_missing_columns_raises() -> None:
    raw = pd.DataFrame({"Open": [1.0], "Close": [1.1]}, index=[pd.Timestamp("2026-01-02")])

    with pytest.raises(PriceFetchError):
        normalize_ohlcv(raw)


@pytest.mark.parametrize(
    ("sector", "expected"),
    [
        ("Technology", "XLK"),
        ("Financial Services", "XLF"),
        ("Real Estate", "XLRE"),
        (" Energy ", "XLE"),
        ("Blockchain", None),
        ("", None),
        (None, None),
    ],
)
def test_sector_etf_for(sector: str | None, expected: str | None) -> None:
    assert sector_etf_for(sector) == expected


def test_static_maps() -> None:
    assert MARKET_ETF == "SPY"
    assert len(SECTOR_ETF) == 11
    assert set(SECTOR_ETF.values()) == {
        "XLK",
        "XLF",
        "XLE",
        "XLV",
        "XLY",
        "XLP",
        "XLI",
        "XLB",
        "XLU",
        "XLRE",
        "XLC",
    }


class _FakeFundsData:
    def __init__(self, top_holdings: pd.DataFrame | None) -> None:
        self.top_holdings = top_holdings


class _FakeTicker:
    """Stands in for yf.Ticker; each attribute is either data or an exception."""

    def __init__(
        self,
        symbol: str,
        *,
        history: pd.DataFrame | None = None,
        info: dict[str, Any] | None = None,
        earnings: pd.DataFrame | None = None,
        funds_data: _FakeFundsData | None = None,
        boom: bool = False,
    ) -> None:
        self.symbol = symbol
        self._history = history
        self._info = info
        self._earnings = earnings
        self.funds_data = funds_data
        self._boom = boom

    def history(self, period: str = "1y", auto_adjust: bool = False) -> pd.DataFrame:
        if self._boom:
            raise RuntimeError("network down")
        assert auto_adjust is False
        return pd.DataFrame() if self._history is None else self._history

    @property
    def info(self) -> dict[str, Any]:
        if self._boom:
            raise RuntimeError("no info")
        return self._info or {}

    def get_earnings_dates(self, limit: int = 12) -> pd.DataFrame | None:
        if self._boom:
            raise RuntimeError("no earnings")
        if self._earnings is None:
            return None
        return self._earnings.head(limit)


def _install(monkeypatch: pytest.MonkeyPatch, tickers: dict[str, _FakeTicker]) -> None:
    def factory(symbol: str) -> _FakeTicker:
        if symbol not in tickers:
            raise RuntimeError(f"unknown symbol {symbol}")
        return tickers[symbol]

    monkeypatch.setattr(prices.yf, "Ticker", factory)


def test_fetch_ohlcv_normalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"AAPL": _FakeTicker("AAPL", history=_raw_history())})

    frame = fetch_ohlcv("AAPL", period="6mo")

    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert len(frame) == 3


def test_fetch_ohlcv_raises_on_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"ZZZZ": _FakeTicker("ZZZZ", history=pd.DataFrame())})

    with pytest.raises(PriceFetchError, match="no price data for ZZZZ"):
        fetch_ohlcv("ZZZZ")


def test_fetch_ohlcv_raises_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"BOOM": _FakeTicker("BOOM", boom=True)})

    with pytest.raises(PriceFetchError, match="no price data for BOOM"):
        fetch_ohlcv("BOOM")


def test_fetch_info_maps_sector_etf(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(
        monkeypatch,
        {
            "AAPL": _FakeTicker(
                "AAPL",
                info={
                    "shortName": "Apple Inc.",
                    "longName": "Apple Incorporated",
                    "sector": "Technology",
                    "industry": "Consumer Electronics",
                },
            )
        },
    )

    info = fetch_info("AAPL")

    assert info == TickerInfo(
        ticker="AAPL",
        name="Apple Inc.",
        sector="Technology",
        industry="Consumer Electronics",
        sector_etf="XLK",
    )


def test_fetch_info_falls_back_to_long_name_then_ticker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(
        monkeypatch,
        {
            "A": _FakeTicker("A", info={"longName": "Agilent"}),
            "B": _FakeTicker("B", info={}),
        },
    )

    assert fetch_info("A") == TickerInfo("A", "Agilent", None, None, None)
    assert fetch_info("B") == TickerInfo("B", "B", None, None, None)


def test_fetch_info_degrades_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"BOOM": _FakeTicker("BOOM", boom=True)})

    assert fetch_info("BOOM") == TickerInfo("BOOM", "BOOM", None, None, None)


def test_fetch_earnings_dates_sorted_unique_tz_naive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.DatetimeIndex(
        [
            "2026-10-29 16:30:00",
            "2026-07-30 16:30:00",
            "2026-07-30 16:30:00",
            "2027-01-28 16:30:00",
        ],
        name="Earnings Date",
    ).tz_localize(TZ)
    earnings = pd.DataFrame({"EPS Estimate": [1.0, 2.0, 2.0, 3.0]}, index=index)
    _install(monkeypatch, {"AAPL": _FakeTicker("AAPL", earnings=earnings)})

    assert fetch_earnings_dates("AAPL") == [
        date(2026, 7, 30),
        date(2026, 10, 29),
        date(2027, 1, 28),
    ]


def test_fetch_earnings_dates_empty_on_none_or_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(
        monkeypatch,
        {
            "NONE": _FakeTicker("NONE", earnings=None),
            "BOOM": _FakeTicker("BOOM", boom=True),
        },
    )

    assert fetch_earnings_dates("NONE") == []
    assert fetch_earnings_dates("BOOM") == []


def test_fetch_peer_returns_skips_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(
        monkeypatch,
        {
            "AMD": _FakeTicker("AMD", history=_raw_history()),
            "DEAD": _FakeTicker("DEAD", history=pd.DataFrame()),
        },
    )

    returns = fetch_peer_returns(["AMD", " AMD ", "DEAD"])

    assert list(returns.columns) == ["AMD"]
    assert returns.index.name == "date"
    # first row of a pct_change is NaN, the rest are real returns
    assert pd.isna(returns["AMD"].iloc[0])
    assert returns["AMD"].iloc[1] == pytest.approx(10.2 / 9.1 - 1.0)


def test_fetch_peer_returns_empty_when_all_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, {"DEAD": _FakeTicker("DEAD", history=pd.DataFrame())})

    returns = fetch_peer_returns(["DEAD", ""])

    assert returns.empty
    assert list(returns.columns) == []


def test_fetch_etf_holdings(monkeypatch: pytest.MonkeyPatch) -> None:
    holdings = pd.DataFrame(
        {
            "Name": ["Nvidia", "Microsoft", "Apple"],
            "Holding Percent": [0.15, 0.14, 0.13],
        },
        index=pd.Index(["NVDA", "msft", "AAPL"], name="Symbol"),
    )
    _install(
        monkeypatch,
        {
            "XLK": _FakeTicker("XLK", funds_data=_FakeFundsData(holdings)),
            "EMPTY": _FakeTicker("EMPTY", funds_data=_FakeFundsData(pd.DataFrame())),
            "NOFUNDS": _FakeTicker("NOFUNDS", funds_data=None),
        },
    )

    assert fetch_etf_holdings("XLK") == ["NVDA", "MSFT", "AAPL"]
    assert fetch_etf_holdings("XLK", top_n=2) == ["NVDA", "MSFT"]
    assert fetch_etf_holdings("EMPTY") == []
    assert fetch_etf_holdings("NOFUNDS") == []
    assert fetch_etf_holdings("UNKNOWN") == []


# --------------------------------------------------------------------------
# Factor proxies (v1.5 plan, decision 4)
# --------------------------------------------------------------------------


def _returns_history(closes: list[float], start: str = "2026-01-02") -> pd.DataFrame:
    """A minimal normalized OHLCV frame with a known close path."""
    index = pd.DatetimeIndex(pd.bdate_range(start=start, periods=len(closes)), name="date")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000.0] * len(closes),
        },
        index=index,
    )


def _install_factor_fetch(
    monkeypatch: pytest.MonkeyPatch,
    frames: dict[str, pd.DataFrame],
    seen: list[tuple[str, str]] | None = None,
) -> None:
    """Monkeypatch `prices.fetch_ohlcv`; a symbol not in `frames` raises."""

    def fake_fetch(ticker: str, period: str = "2y") -> pd.DataFrame:
        if seen is not None:
            seen.append((ticker, period))
        if ticker not in frames:
            raise PriceFetchError(f"no price data for {ticker}")
        return frames[ticker]

    monkeypatch.setattr(prices, "fetch_ohlcv", fake_fetch)


def test_factor_and_country_etf_tables() -> None:
    assert FACTOR_ETFS == {"oil": "USO", "dollar": "UUP", "rates": "TLT", "gold": "GLD"}
    assert COUNTRY_ETFS == {
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


def test_factor_column_for_prefixes_only_countries() -> None:
    assert factor_column_for("oil") == "oil"
    assert factor_column_for(" rates ") == "rates"
    assert factor_column_for("TW") == "country:TW"
    assert factor_column_for("tw") == "country:TW"
    assert factor_column_for("country:tw") == "country:TW"
    # Not a country we hold an ETF for: stored verbatim, never invented.
    assert factor_column_for("BR") == "BR"


def test_fetch_factor_returns_columns_and_values(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_factor_fetch(
        monkeypatch,
        {
            "USO": _returns_history([100.0, 101.0, 99.99]),
            "EWT": _returns_history([50.0, 52.0, 52.0]),
        },
    )

    returns = fetch_factor_returns({"oil": "USO", "TW": "EWT"})

    assert list(returns.columns) == ["oil", "country:TW"]
    assert returns.index.name == "date"
    assert returns.index.is_monotonic_increasing
    assert pd.isna(returns["oil"].iloc[0])  # no prior close
    assert returns["oil"].iloc[1] == pytest.approx(0.01)
    assert returns["country:TW"].iloc[1] == pytest.approx(0.04)
    assert returns["country:TW"].iloc[2] == pytest.approx(0.0)


def test_fetch_factor_returns_drops_a_failed_symbol_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _install_factor_fetch(monkeypatch, {"USO": _returns_history([100.0, 101.0])})

    with caplog.at_level("WARNING", logger="stock_moves.prices"):
        returns = fetch_factor_returns({"oil": "USO", "gold": "GLD", "CN": "FXI"})

    # A missing proxy is one fewer candidate driver, never a failed ingest.
    assert list(returns.columns) == ["oil"]
    assert "gold" in caplog.text
    assert "country:CN" in caplog.text


def test_fetch_factor_returns_empty_when_nothing_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_factor_fetch(monkeypatch, {})

    assert fetch_factor_returns({"oil": "USO"}).empty
    assert fetch_factor_returns({}).empty
    assert list(fetch_factor_returns({}).columns) == []
    assert fetch_factor_returns({}).index.name == "date"


def test_fetch_factor_returns_skips_blanks_and_forwards_the_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, str]] = []
    _install_factor_fetch(monkeypatch, {"USO": _returns_history([100.0, 101.0])}, seen)

    assert fetch_factor_returns({"oil": " ", "  ": "USO"}).empty
    assert seen == []

    fetch_factor_returns({"oil": "USO"}, period="6mo")
    assert seen == [("USO", "6mo")]

    fetch_factor_returns({"oil": "USO"})
    assert seen[-1] == ("USO", "2y")


@pytest.mark.skip(reason="network")
def test_fetch_aapl_smoke() -> None:
    """Hand-run smoke test against the live yfinance API (T15)."""
    frame = fetch_ohlcv("AAPL", period="1mo")
    assert not frame.empty
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame.index.tz is None

    info = fetch_info("AAPL")
    assert info.ticker == "AAPL"
    assert info.sector_etf == "XLK"

    assert fetch_earnings_dates("AAPL")
    assert fetch_etf_holdings("XLK")
    assert not fetch_peer_returns(["MSFT"], period="1mo").empty
