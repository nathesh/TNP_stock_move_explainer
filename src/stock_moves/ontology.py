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

v1.5 adds the typed-fact layer on top of that row: `build_edges` turns one
`suggest_relations` call plus the fitted factor betas into `company_edges`
rows, and `edges_of` / `relation_tickers` / `country_codes` / `edges_to_dicts`
are the reads over them. `competitor` replaces the idea of a peer (v1.5
decision 2): `companies.peers_json` stays as the keyless ETF fallback and is
where `build_edges` gets competitors when the model names none.

`prices` is imported as a module and called as `prices.fetch_info(...)` so a
test monkeypatches one name and this layer never touches the network.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from sqlmodel import Session, col, select

from stock_moves import prices
from stock_moves.models import Company, CompanyEdge, utcnow
from stock_moves.prices import TickerInfo
from stock_moves.providers.base import ModelProvider, Relations

__all__ = [
    "EDGE_SOURCE_ETF",
    "EDGE_SOURCE_MODEL",
    "EDGE_SOURCE_PRICES",
    "ETF_PEER_TOP_N",
    "MODEL_RELATIONS",
    "PEERS_SOURCE_ETF",
    "PEERS_SOURCE_MODEL",
    "build_edges",
    "country_codes",
    "edges_of",
    "edges_to_dicts",
    "get_or_build_company",
    "peers_of",
    "relation_tickers",
    "upsert_company",
]

ETF_PEER_TOP_N = 8
"""How many ETF holdings to keep when peers come from the fallback."""

PEERS_SOURCE_MODEL = "model"
PEERS_SOURCE_ETF = "etf_holdings"

# `company_edges.source` values (v1.5 decision 1). The first two are the same
# two strings peers already use, because a competitor edge has exactly the two
# provenances a peer list does.
EDGE_SOURCE_MODEL = "model"
EDGE_SOURCE_ETF = "etf_holdings"
EDGE_SOURCE_PRICES = "prices"

MODEL_RELATIONS: tuple[str, ...] = ("competitor", "supplier", "customer", "country")
"""The relations one `suggest_relations` call speaks for, and therefore the
relations `build_edges` always replaces. `factor` is not among them: it comes
from prices (v1.5 decision 4) and is replaced only when betas are handed in."""

FACTOR_RELATION = "factor"


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


# --------------------------------------------------------------------------- #
# v1.5: typed edges
# --------------------------------------------------------------------------- #


def _relations(provider: ModelProvider, company: Company) -> Relations:
    """One `suggest_relations` call; all-empty on any provider failure.

    Same contract as `_provider_peers`: a broken or rate-limited model degrades
    to the keyless behaviour — competitors from the ETF fallback and no other
    edges — rather than failing the build.
    """
    try:
        return provider.suggest_relations(
            company.ticker, company.name, company.sector, company.industry
        )
    except Exception:  # noqa: BLE001 - edges are optional; degrade to the fallback
        return Relations(competitors=(), suppliers=(), customers=(), countries=())


def _clean_country_weights(
    raw: Sequence[tuple[str, float]],
) -> list[tuple[str, float]]:
    """`(ISO alpha-2, weight)` pairs, upper-cased and de-duplicated.

    A model that names the same country twice gets the larger of its two
    weights rather than whichever came last, so the stored row does not depend
    on ordering. Blank codes and unparsable weights are dropped.
    """
    best: dict[str, float] = {}
    for code, weight in raw:
        key = str(code).strip().upper()
        if not key:
            continue
        try:
            value = float(weight)
        except (TypeError, ValueError):
            continue
        if key not in best or value > best[key]:
            best[key] = value
    return sorted(best.items())


def _clean_factor_betas(betas: Mapping[str, float]) -> list[tuple[str, float]]:
    """`(factor name, beta)` pairs, sorted by name; unparsable betas dropped.

    Names are left exactly as the factor layer spells its columns — `"oil"`,
    `"country:TW"` — because `moves.macro_driver` is compared against them
    verbatim (v1.5 decision 6).
    """
    cleaned: dict[str, float] = {}
    for name, beta in betas.items():
        key = str(name).strip()
        if not key:
            continue
        try:
            cleaned[key] = float(beta)
        except (TypeError, ValueError):
            continue
    return sorted(cleaned.items())


def build_edges(
    session: Session,
    company: Company,
    provider: ModelProvider,
    factor_betas: Mapping[str, float] | None = None,
) -> list[CompanyEdge]:
    """Build and store the company's `company_edges` rows, and return them.

    One `suggest_relations` call supplies the `competitor`, `supplier`,
    `customer` and `country` edges (source `"model"`); when the model names no
    competitors, the stored peer list — the sector-ETF fallback — supplies them
    instead (source `"etf_holdings"`), because competitor *is* the peer concept
    in v1.5 (decision 2). `factor` edges come from `factor_betas` (source
    `"prices"`, decision 4).

    Writes are replace-all per `(src, relation)`: every relation this call
    speaks for is deleted and re-inserted, so a second build with the same
    inputs leaves the same rows rather than a second copy. The four model
    relations are always spoken for — an empty answer clears stale rows. The
    `factor` relation is spoken for only when `factor_betas` is given: `None`
    means "prices had nothing to say this time" and leaves fitted betas alone,
    while an empty mapping means "no factors" and clears them.

    Commits, like `upsert_company`: this is a cache-filling write the caller
    should not have to remember to flush.
    """
    src = company.ticker.strip().upper()
    relations = _relations(provider, company)

    competitors = _clean_tickers(relations.competitors, exclude=src)
    competitor_source = EDGE_SOURCE_MODEL
    if not competitors:
        competitors = _clean_tickers(peers_of(company), exclude=src)
        competitor_source = EDGE_SOURCE_ETF

    stamp = utcnow()
    rows: list[CompanyEdge] = [
        CompanyEdge(
            src=src,
            dst=dst,
            relation="competitor",
            weight=1.0,
            source=competitor_source,
            updated_at=stamp,
        )
        for dst in competitors
    ]
    for relation, names in (
        ("supplier", relations.suppliers),
        ("customer", relations.customers),
    ):
        rows.extend(
            CompanyEdge(
                src=src,
                dst=dst,
                relation=relation,
                weight=1.0,
                source=EDGE_SOURCE_MODEL,
                updated_at=stamp,
            )
            for dst in _clean_tickers(names, exclude=src)
        )
    rows.extend(
        CompanyEdge(
            src=src,
            dst=code,
            relation="country",
            weight=weight,
            source=EDGE_SOURCE_MODEL,
            updated_at=stamp,
        )
        for code, weight in _clean_country_weights(relations.countries)
    )

    written = list(MODEL_RELATIONS)
    if factor_betas is not None:
        written.append(FACTOR_RELATION)
        rows.extend(
            CompanyEdge(
                src=src,
                dst=name,
                relation=FACTOR_RELATION,
                weight=beta,
                source=EDGE_SOURCE_PRICES,
                updated_at=stamp,
            )
            for name, beta in _clean_factor_betas(factor_betas)
        )

    # Replace, do not accumulate. Flush the deletes before the inserts, or the
    # new rows collide with the old ones on the (src, dst, relation) key.
    stale = session.exec(
        select(CompanyEdge)
        .where(col(CompanyEdge.src) == src)
        .where(col(CompanyEdge.relation).in_(written))
    ).all()
    for row in stale:
        session.delete(row)
    session.flush()

    for row in rows:
        session.add(row)
    session.commit()
    for row in rows:
        session.refresh(row)
    return rows


def edges_of(session: Session, ticker: str, relation: str | None = None) -> list[CompanyEdge]:
    """The company's edges, optionally just one relation, in a stable order.

    Ordered by relation then `dst` so two calls — and two processes — return
    the same list; callers that want the strongest edge first sort on `weight`
    themselves, as `country_codes` does.
    """
    key = ticker.strip().upper()
    statement = select(CompanyEdge).where(col(CompanyEdge.src) == key)
    if relation is not None:
        statement = statement.where(col(CompanyEdge.relation) == relation.strip())
    statement = statement.order_by(col(CompanyEdge.relation), col(CompanyEdge.dst))
    return list(session.exec(statement).all())


def relation_tickers(session: Session, ticker: str, relation: str) -> list[str]:
    """The `dst` tickers of one relation, upper-cased, de-duplicated and sorted.

    The retrieval and routing layers want "who are the rivals" as a plain list
    of symbols; this is that read.
    """
    return sorted({edge.dst.strip().upper() for edge in edges_of(session, ticker, relation)})


def country_codes(session: Session, ticker: str) -> list[tuple[str, float]]:
    """`(ISO alpha-2, weight)` for the `country` edges, largest weight first.

    Ties break on the code so the order is total: the gate rule and the geo
    queries both walk this list and must walk it the same way every time.
    """
    edges = edges_of(session, ticker, "country")
    return sorted(((edge.dst, edge.weight) for edge in edges), key=lambda pair: (-pair[1], pair[0]))


def edges_to_dicts(edges: Sequence[CompanyEdge]) -> list[dict[str, Any]]:
    """Edges as plain dicts for the API read and the chat tool (v1.5 decision 10).

    `src` is left out: every caller has already named the ticker it asked
    about, and repeating it in each row is noise.
    """
    return [
        {
            "dst": edge.dst,
            "relation": edge.relation,
            "weight": edge.weight,
            "source": edge.source,
        }
        for edge in edges
    ]
