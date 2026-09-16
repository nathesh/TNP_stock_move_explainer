"""Tests for stock_moves.ontology. No network: `prices` is monkeypatched."""

from __future__ import annotations

from typing import Any

import pytest
from sqlmodel import select

from stock_moves import ontology, prices
from stock_moves.db import Session
from stock_moves.models import Company, CompanyEdge
from stock_moves.ontology import (
    build_edges,
    country_codes,
    edges_of,
    edges_to_dicts,
    get_or_build_company,
    peers_of,
    relation_tickers,
    upsert_company,
)
from stock_moves.prices import TickerInfo
from stock_moves.providers import Relations

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
    """A provider that answers `suggest_peers` and `suggest_relations` — the two
    methods this module calls. `raises` makes both calls fail, which must route
    to the ETF fallback rather than fail the build."""

    name = "stub"

    def __init__(
        self,
        peers: list[str] | None = None,
        *,
        raises: bool = False,
        relations: Relations | None = None,
    ) -> None:
        self._peers = peers or []
        self._raises = raises
        self._relations = relations or Relations(
            competitors=(), suppliers=(), customers=(), countries=()
        )
        self.calls = 0
        self.relation_calls = 0

    def suggest_peers(
        self, ticker: str, name: str, sector: str | None, industry: str | None
    ) -> list[str]:
        self.calls += 1
        if self._raises:
            raise RuntimeError("model unavailable")
        return list(self._peers)

    def suggest_relations(
        self, ticker: str, name: str, sector: str | None, industry: str | None
    ) -> Relations:
        self.relation_calls += 1
        if self._raises:
            raise RuntimeError("model unavailable")
        return self._relations


MODEL_RELATIONS = Relations(
    competitors=("amd", "INTC"),
    suppliers=("TSM",),
    customers=("MSFT",),
    countries=(("tw", 0.4), ("CN", 0.25)),
)


def _company(session: Session, peers: list[str] | None = None) -> Company:
    """A stored `companies` row with `peers` as its ETF-sourced peer list."""
    return upsert_company(session, INFO, peers or [], "etf_holdings")


def _rows(session: Session, relation: str | None = None) -> list[tuple[str, str, float, str]]:
    """Stored edges as comparable tuples, in `edges_of` order."""
    return [
        (edge.relation, edge.dst, edge.weight, edge.source)
        for edge in edges_of(session, "TEST", relation)
    ]


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


# --------------------------------------------------------------------------- #
# build_edges
# --------------------------------------------------------------------------- #


def test_build_edges_writes_typed_model_edges(session: Session) -> None:
    company = _company(session, ["OLD1"])
    provider = _StubProvider(relations=MODEL_RELATIONS)

    edges = build_edges(session, company, provider)

    assert provider.relation_calls == 1
    # The model named competitors, so the stored peer list is not consulted.
    assert _rows(session) == [
        ("competitor", "AMD", 1.0, "model"),
        ("competitor", "INTC", 1.0, "model"),
        ("country", "CN", 0.25, "model"),
        ("country", "TW", 0.4, "model"),
        ("customer", "MSFT", 1.0, "model"),
        ("supplier", "TSM", 1.0, "model"),
    ]
    assert {(edge.relation, edge.dst) for edge in edges} == {
        ("competitor", "AMD"),
        ("competitor", "INTC"),
        ("country", "CN"),
        ("country", "TW"),
        ("customer", "MSFT"),
        ("supplier", "TSM"),
    }
    assert all(edge.src == "TEST" for edge in edges)


def test_build_edges_falls_back_to_etf_competitors(session: Session) -> None:
    company = _company(session, ["ETF1", "ETF2"])

    build_edges(session, company, _StubProvider())

    assert _rows(session) == [
        ("competitor", "ETF1", 1.0, "etf_holdings"),
        ("competitor", "ETF2", 1.0, "etf_holdings"),
    ]
    # An empty model answer means no supplier, customer or country edges at
    # all: without a key the geopolitical gate can never open.
    assert country_codes(session, "TEST") == []


def test_build_edges_survives_a_failing_provider(session: Session) -> None:
    company = _company(session, ["ETF1"])

    build_edges(session, company, _StubProvider(raises=True))

    assert _rows(session) == [("competitor", "ETF1", 1.0, "etf_holdings")]


def test_build_edges_replaces_rather_than_duplicates(session: Session) -> None:
    company = _company(session, ["ETF1"])
    provider = _StubProvider(relations=MODEL_RELATIONS)

    build_edges(session, company, provider)
    build_edges(session, company, provider)

    assert len(session.exec(select(CompanyEdge)).all()) == 6
    assert _rows(session, "competitor") == [
        ("competitor", "AMD", 1.0, "model"),
        ("competitor", "INTC", 1.0, "model"),
    ]

    # A later, different answer replaces the earlier one instead of unioning.
    narrowed = _StubProvider(
        relations=Relations(
            competitors=("AVGO",), suppliers=(), customers=(), countries=(("JP", 0.1),)
        )
    )
    build_edges(session, company, narrowed)

    assert _rows(session) == [
        ("competitor", "AVGO", 1.0, "model"),
        ("country", "JP", 0.1, "model"),
    ]


def test_build_edges_writes_factor_edges_from_betas(session: Session) -> None:
    company = _company(session)

    build_edges(session, company, _StubProvider(), {"oil": 0.8, "country:TW": -0.25})

    assert _rows(session, "factor") == [
        ("factor", "country:TW", -0.25, "prices"),
        ("factor", "oil", 0.8, "prices"),
    ]


def test_factor_betas_none_leaves_factor_rows_untouched(session: Session) -> None:
    company = _company(session)
    build_edges(session, company, _StubProvider(), {"oil": 0.8})

    # No betas this time: prices had nothing to say, so the fitted rows stand.
    build_edges(session, company, _StubProvider(relations=MODEL_RELATIONS))

    assert _rows(session, "factor") == [("factor", "oil", 0.8, "prices")]

    # Betas handed in replace them; an empty mapping clears them.
    build_edges(session, company, _StubProvider(), {"gold": 0.1})
    assert _rows(session, "factor") == [("factor", "gold", 0.1, "prices")]

    build_edges(session, company, _StubProvider(), {})
    assert _rows(session, "factor") == []


# --------------------------------------------------------------------------- #
# The reads
# --------------------------------------------------------------------------- #


def test_edges_of_filters_by_relation(session: Session) -> None:
    company = _company(session)
    build_edges(session, company, _StubProvider(relations=MODEL_RELATIONS), {"oil": 0.5})

    assert [edge.dst for edge in edges_of(session, "TEST", "competitor")] == ["AMD", "INTC"]
    assert [edge.dst for edge in edges_of(session, "TEST", "supplier")] == ["TSM"]
    assert [edge.dst for edge in edges_of(session, "TEST", "factor")] == ["oil"]
    assert len(edges_of(session, "TEST")) == 7
    assert edges_of(session, "OTHER") == []
    # Lower case in, same edges out.
    assert len(edges_of(session, "test")) == 7


def test_relation_tickers_are_sorted_and_upper_case(session: Session) -> None:
    company = _company(session)
    build_edges(session, company, _StubProvider(relations=MODEL_RELATIONS))

    assert relation_tickers(session, "TEST", "competitor") == ["AMD", "INTC"]
    assert relation_tickers(session, "TEST", "supplier") == ["TSM"]
    assert relation_tickers(session, "TEST", "customer") == ["MSFT"]
    assert relation_tickers(session, "TEST", "factor") == []


def test_country_codes_are_ordered_by_weight(session: Session) -> None:
    company = _company(session)
    relations = Relations(
        competitors=(),
        suppliers=(),
        customers=(),
        countries=(("CN", 0.2), ("TW", 0.55), ("JP", 0.2)),
    )
    build_edges(session, company, _StubProvider(relations=relations))

    # Largest weight first; ties break on the code, so the order is total.
    assert country_codes(session, "TEST") == [("TW", 0.55), ("CN", 0.2), ("JP", 0.2)]
    assert country_codes(session, "OTHER") == []


def test_edges_to_dicts_shape(session: Session) -> None:
    company = _company(session)
    build_edges(session, company, _StubProvider(relations=MODEL_RELATIONS), {"oil": 0.5})

    payload = edges_to_dicts(edges_of(session, "TEST", "country"))

    assert payload == [
        {"dst": "CN", "relation": "country", "weight": 0.25, "source": "model"},
        {"dst": "TW", "relation": "country", "weight": 0.4, "source": "model"},
    ]
    assert edges_to_dicts([]) == []
