"""Tests for stock_moves.ontology. No network: `prices` is monkeypatched."""

from __future__ import annotations

from typing import Any

import pytest
from sqlmodel import select

from stock_moves import ontology, prices
from stock_moves.db import Session
from stock_moves.models import Company
from stock_moves.ontology import get_or_build_company, peers_of, upsert_company
from stock_moves.prices import TickerInfo

INFO = TickerInfo(
    ticker="TEST",
    name="Test Corp",
    sector="Technology",
    industry="Semiconductors",
    sector_etf="XLK",
)

# Holdings include the ticker itself and a lower-cased symbol, as the real
# fallback would: both have to be cleaned up by the ontology layer.
HOLDINGS = ["ETF1", "TEST", "etf2"]


class _StubProvider:
    """A provider that only answers `suggest_peers` — the one method this
    module calls. `raises` makes the call fail, which must route to the ETF
    fallback rather than fail the build."""

    name = "stub"

    def __init__(self, peers: list[str] | None = None, *, raises: bool = False) -> None:
        self._peers = peers or []
        self._raises = raises
        self.calls = 0

    def suggest_peers(
        self, ticker: str, name: str, sector: str | None, industry: str | None
    ) -> list[str]:
        self.calls += 1
        if self._raises:
            raise RuntimeError("model unavailable")
        return list(self._peers)


def _patch_prices(
    monkeypatch: pytest.MonkeyPatch,
    *,
    holdings: list[str] | None = None,
) -> dict[str, Any]:
    """Monkeypatch `prices.fetch_info` / `prices.fetch_etf_holdings` and return
    a counter of how often each was called, plus the ETF asked for."""
    counts: dict[str, Any] = {"info": 0, "holdings": 0, "etf": None, "top_n": None}

    def fake_fetch_info(ticker: str) -> TickerInfo:
        counts["info"] += 1
        return INFO

    def fake_fetch_etf_holdings(etf: str, top_n: int = 10) -> list[str]:
        counts["holdings"] += 1
        counts["etf"] = etf
        counts["top_n"] = top_n
        return list(HOLDINGS if holdings is None else holdings)

    monkeypatch.setattr(prices, "fetch_info", fake_fetch_info)
    monkeypatch.setattr(prices, "fetch_etf_holdings", fake_fetch_etf_holdings)
    return counts


# --------------------------------------------------------------------------- #
# upsert_company
# --------------------------------------------------------------------------- #


def test_upsert_company_inserts_normalised_peers(session: Session) -> None:
    company = upsert_company(session, INFO, ["bbb", "AAA", "aaa", ""], "model")

    assert company.ticker == "TEST"
    assert company.name == "Test Corp"
    assert company.sector == "Technology"
    assert company.industry == "Semiconductors"
    assert company.sector_etf == "XLK"
    # Upper-cased, de-duplicated, sorted, blanks dropped.
    assert company.peers == ["AAA", "BBB"]
    assert company.peers_source == "model"
    assert session.get(Company, "TEST") is not None


def test_upsert_company_updates_in_place(session: Session) -> None:
    first = upsert_company(session, INFO, ["AAA"], "model")
    stamp = first.updated_at

    renamed = TickerInfo("TEST", "Test Corporation", "Technology", "Software", "XLK")
    second = upsert_company(session, renamed, ["CCC"], "etf_holdings")

    assert second.ticker == "TEST"
    assert second.name == "Test Corporation"
    assert second.industry == "Software"
    assert second.peers == ["CCC"]
    assert second.peers_source == "etf_holdings"
    assert second.updated_at >= stamp
    assert len(session.exec(select(Company)).all()) == 1


# --------------------------------------------------------------------------- #
# get_or_build_company
# --------------------------------------------------------------------------- #


def test_provider_peers_are_cleaned_and_sourced_model(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts = _patch_prices(monkeypatch)
    provider = _StubProvider(["AAA", "test", "bbb"])

    company = get_or_build_company(session, "test", provider)

    # Own ticker excluded, upper-cased, sorted.
    assert company.peers == ["AAA", "BBB"]
    assert company.peers_source == "model"
    assert company.ticker == "TEST"
    assert provider.calls == 1
    # The provider answered, so the ETF fallback was never consulted.
    assert counts["holdings"] == 0


def test_second_call_is_cached(session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    counts = _patch_prices(monkeypatch)
    provider = _StubProvider(["AAA"])

    first = get_or_build_company(session, "TEST", provider)
    second = get_or_build_company(session, "TEST", provider)

    assert counts["info"] == 1
    assert provider.calls == 1
    assert second.ticker == first.ticker
    assert second.peers == ["AAA"]


def test_refresh_refetches(session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    counts = _patch_prices(monkeypatch)
    provider = _StubProvider(["AAA"])

    get_or_build_company(session, "TEST", provider)
    refreshed = get_or_build_company(session, "TEST", provider, refresh=True)

    assert counts["info"] == 2
    assert provider.calls == 2
    assert refreshed.peers == ["AAA"]


def test_provider_error_falls_back_to_etf_holdings(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts = _patch_prices(monkeypatch)
    provider = _StubProvider(raises=True)

    company = get_or_build_company(session, "TEST", provider)

    assert company.peers == ["ETF1", "ETF2"]
    assert company.peers_source == "etf_holdings"
    assert counts["etf"] == "XLK"
    assert counts["top_n"] == ontology.ETF_PEER_TOP_N


def test_empty_provider_falls_back_to_etf_holdings(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_prices(monkeypatch)
    company = get_or_build_company(session, "TEST", _StubProvider([]))

    assert company.peers == ["ETF1", "ETF2"]
    assert company.peers_source == "etf_holdings"


def test_no_peers_anywhere_leaves_source_none(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_prices(monkeypatch, holdings=[])
    company = get_or_build_company(session, "TEST", _StubProvider([]))

    assert company.peers == []
    assert company.peers_json == "[]"
    assert company.peers_source is None


def test_peers_of(session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_prices(monkeypatch)
    company = get_or_build_company(session, "TEST", _StubProvider(["AAA", "bbb"]))

    assert peers_of(company) == ["AAA", "BBB"]
    assert peers_of(Company(ticker="X", name="X")) == []
