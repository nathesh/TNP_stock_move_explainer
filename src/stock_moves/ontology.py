"""Company ontology — the `companies` row (DESIGN section 2).

Identity (name/sector/industry) and the sector-ETF mapping come from
`yfinance`; peers come from **one** provider call per company and are cached in
the row, because peers are a second, data-driven industry signal (same-day peer
co-movement) rather than something worth re-deriving per request.

Peers have a keyless fallback so the app runs with zero keys: when the provider
returns nothing — `HeuristicProvider.suggest_peers` returns `[]` by design, and
a model call can fail — the top holdings of the sector ETF stand in.
`peers_source` records which path produced the list ("model", "etf_holdings",
or None when neither did), so the provenance of a peer set is never guesswork.

`prices` is imported as a module and called as `prices.fetch_info(...)` so a
test monkeypatches one name and this layer never touches the network.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

from sqlmodel import Session

from stock_moves import prices
from stock_moves.models import Company, utcnow
from stock_moves.prices import TickerInfo
from stock_moves.providers.base import ModelProvider

__all__ = [
    "ETF_PEER_TOP_N",
    "PEERS_SOURCE_ETF",
    "PEERS_SOURCE_MODEL",
    "get_or_build_company",
    "peers_of",
    "upsert_company",
]

ETF_PEER_TOP_N = 8
"""How many ETF holdings to keep when peers come from the fallback."""

PEERS_SOURCE_MODEL = "model"
PEERS_SOURCE_ETF = "etf_holdings"


def _clean_tickers(raw: Iterable[str], *, exclude: str = "") -> list[str]:
    """Upper-case, de-duplicate and sort tickers, dropping blanks and `exclude`."""
    cleaned = {str(item).strip().upper() for item in raw if item}
    cleaned.discard("")
    cleaned.discard(exclude.strip().upper())
    return sorted(cleaned)


def upsert_company(
    session: Session,
    info: TickerInfo,
    peers: Sequence[str],
    peers_source: str | None = None,
) -> Company:
    """Insert or update the `companies` row for `info.ticker` and return it.

    Peers are stored normalised (upper-cased, de-duplicated, sorted) so the
    stored JSON is stable and comparable regardless of provider ordering.
    """
    ticker = info.ticker.strip().upper()
    company = session.get(Company, ticker)
    if company is None:
        company = Company(ticker=ticker, name=info.name)

    company.name = info.name
    company.sector = info.sector
    company.industry = info.industry
    company.sector_etf = info.sector_etf
    company.peers_json = json.dumps(_clean_tickers(peers))
    company.peers_source = peers_source
    company.updated_at = utcnow()

    session.add(company)
    session.commit()
    session.refresh(company)
    return company


def _provider_peers(provider: ModelProvider, ticker: str, info: TickerInfo) -> list[str]:
    """One provider call for peers; `[]` on any provider failure.

    A broken or rate-limited model must never fail ingest — the ETF fallback
    covers it.
    """
    try:
        suggested = provider.suggest_peers(ticker, info.name, info.sector, info.industry)
    except Exception:  # noqa: BLE001 - peers are optional; degrade to the fallback
        return []
    return _clean_tickers(suggested or [], exclude=ticker)


def _etf_peers(etf: str, ticker: str) -> list[str]:
    """Sector-ETF top holdings as the keyless peer set; `[]` on any failure."""
    try:
        holdings = prices.fetch_etf_holdings(etf, top_n=ETF_PEER_TOP_N)
    except Exception:  # noqa: BLE001 - funds data is absent in older yfinance builds
        return []
    return _clean_tickers(holdings or [], exclude=ticker)


def get_or_build_company(
    session: Session,
    ticker: str,
    provider: ModelProvider,
    *,
    refresh: bool = False,
) -> Company:
    """Return the cached `companies` row, building it on first sight.

    The metadata fetch and the provider call happen once per company: a stored
    row short-circuits both unless `refresh` is set.
    """
    key = ticker.strip().upper()
    existing = session.get(Company, key)
    if existing is not None and not refresh:
        return existing

    info = prices.fetch_info(key)

    peers = _provider_peers(provider, key, info)
    peers_source: str | None = PEERS_SOURCE_MODEL if peers else None
    if not peers and info.sector_etf:
        peers = _etf_peers(info.sector_etf, key)
        peers_source = PEERS_SOURCE_ETF if peers else None

    return upsert_company(session, info, peers, peers_source)


def peers_of(company: Company) -> list[str]:
    """Peer tickers for a company row; `[]` when none are stored."""
    return list(company.peers)
