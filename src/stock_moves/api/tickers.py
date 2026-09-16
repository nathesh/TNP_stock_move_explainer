"""Ticker routes: lazy population (DESIGN section 5) and reads (DESIGN section 6).

Four routes, and between them they own exactly two decisions.

1. **When to ingest.** DESIGN section 5 makes population lazy: a read ingests
   when the ticker has never been seen, when the newest stored bar is older
   than the last completed trading day, or when `?refresh=true` says so.
   Otherwise the read is pure SQL. The response says which happened
   (`ingested`), because "why is this list empty" and "why was this slow" are
   the two questions a reviewer asks first.
2. **What 404 means.** A ticker with no price data *after* an ingest attempt,
   and a `prices.PriceFetchError` raised during that attempt, are both 404:
   from the client's side "unknown symbol" and "no data for this symbol" are
   the same answer. A date that is simply not a major move is also 404 — the
   move resource does not exist.

Everything else is delegated. The filters are applied in SQL by
`stock_moves.queries`, the payloads are the dicts that module serialises (the
same dicts the chat tools hand a model), and the response models only validate
and document them. So these handlers are argument plumbing plus the freshness
rule, which is the whole point of keeping them thin.

The fourth route, `GET /{ticker}/relations` (v1.5 decision 10), is the one
exception to the first decision: it is a pure read that never ingests, because
edges are a by-product of an ingest rather than something a window of them can
be computed for on demand.

`ingest.get_news_source` is called through the module rather than imported by
name so that monkeypatching `stock_moves.ingest.get_news_source` in a test
reaches the on-demand enrichment path here too, exactly as it reaches
`ingest_ticker`.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from stock_moves import ingest, ontology, prices
from stock_moves.api.deps import get_db, get_provider_dep, get_settings_dep
from stock_moves.api.schemas import (
    CompanyOut,
    IngestResponse,
    MoveOut,
    RelationsResponse,
    TickerResponse,
)
from stock_moves.db import Session
from stock_moves.ingest import (
    IngestResult,
    enrich_move,
    ingest_ticker,
    is_stale,
    needs_ingest,
)
from stock_moves.models import Company
from stock_moves.news import NewsSource
from stock_moves.providers import ModelProvider
from stock_moves.queries import (
    MoveFilters,
    article_to_dict,
    articles_for_move,
    explanation_to_dict,
    get_company,
    get_explanation,
    get_move,
    has_prices,
    list_moves,
    list_prices,
    move_to_dict,
    price_to_dict,
)
from stock_moves.settings import Settings, get_settings

__all__ = ["router"]

router = APIRouter(prefix="/tickers", tags=["tickers"])

SessionDep = Annotated[Session, Depends(get_db)]
ProviderDep = Annotated[ModelProvider, Depends(get_provider_dep)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
TickerPath = Annotated[str, Path(description="Ticker symbol, case-insensitive.")]

# A query parameter's default has to be a literal in the signature (FastAPI
# reads it at import, and a call there would be a mutable-default bug), so the
# configured defaults are captured once here rather than per request. They are
# the *defaults* only: `Settings` still owns the values, and any request may
# override them.
_DEFAULTS = get_settings()
DEFAULT_PERIOD = _DEFAULTS.default_period
DEFAULT_TOP_N = _DEFAULTS.default_top_n
DEFAULT_Z_THRESHOLD = _DEFAULTS.default_z_threshold
DEFAULT_PCT_THRESHOLD = _DEFAULTS.default_pct_threshold

#: The move list is capped so a wide window cannot return a year of rows.
DEFAULT_LIMIT = 50

Direction = Literal["up", "down"]
Category = Literal["company", "industry", "macro", "unexplained"]
#: The `company_edges.relation` values, as a filter on the relations read.
#: Spelled out rather than taken from `models.EDGE_RELATIONS` because FastAPI
#: needs a literal type at import, and a wrong value is then a 422 with the
#: five names in it instead of a silently empty list.
Relation = Literal["competitor", "supplier", "customer", "country", "factor"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _key(ticker: str) -> str:
    """Tickers are stored upper-case; a URL may spell one any way it likes."""
    return ticker.strip().upper()


def _run_ingest(
    session: Session,
    ticker: str,
    provider: ModelProvider,
    *,
    period: str,
    top_n: int,
    z_threshold: float,
    pct_threshold: float,
    refresh: bool,
) -> IngestResult:
    """Ingest one ticker, turning an unusable symbol into a 404.

    `news_source` is deliberately left unset: `ingest_ticker` builds it from
    the settings, which keeps the news-source choice in one place.
    """
    try:
        return ingest_ticker(
            session,
            ticker,
            period=period,
            top_n=top_n,
            z_threshold=z_threshold,
            pct_threshold=pct_threshold,
            refresh=refresh,
            provider=provider,
        )
    except prices.PriceFetchError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="unknown ticker or no price data",
        ) from exc


def _news_source(settings: Settings) -> NewsSource:
    """The configured news source, resolved through the `ingest` module."""
    return ingest.get_news_source(
        settings.news_source,
        timeout_s=settings.http_timeout_s,
        throttle_s=settings.gdelt_throttle_s,
    )


def _require_company(session: Session, ticker: str) -> Company:
    """The company row, or 404 — it is written by the first ingest, so its
    absence means the ticker has no stored data at all."""
    company = get_company(session, ticker)
    if company is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"unknown ticker or no price data for {ticker}",
        )
    return company


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@router.post("/{ticker}/ingest", response_model=IngestResponse)
def ingest_endpoint(
    ticker: TickerPath,
    session: SessionDep,
    provider: ProviderDep,
    period: Annotated[str, Query(description="yfinance period, e.g. 1y or 5y.")] = (DEFAULT_PERIOD),
    top_n: Annotated[
        int, Query(ge=0, description="Moves explained now; the rest on demand.")
    ] = DEFAULT_TOP_N,
    z_threshold: Annotated[float, Query(ge=0.0)] = DEFAULT_Z_THRESHOLD,
    pct_threshold: Annotated[float, Query(ge=0.0)] = DEFAULT_PCT_THRESHOLD,
    refresh: Annotated[
        bool, Query(description="Re-fetch news and re-explain cached moves.")
    ] = True,
) -> IngestResponse:
    """Ingest a ticker explicitly: prices, moves, news and the top-N explanations.

    Idempotent on `(ticker, date)`, so calling it twice stores nothing new and
    returns the same counts. `refresh` defaults to true here — an explicit
    ingest is a request to redo the work, unlike the lazy one behind a read.
    """
    result = _run_ingest(
        session,
        _key(ticker),
        provider,
        period=period,
        top_n=top_n,
        z_threshold=z_threshold,
        pct_threshold=pct_threshold,
        refresh=refresh,
    )
    return IngestResponse.model_validate(asdict(result))


@router.get("/{ticker}", response_model=TickerResponse)
def read_ticker(
    ticker: TickerPath,
    session: SessionDep,
    provider: ProviderDep,
    start: Annotated[date | None, Query(description="Inclusive lower bound.")] = None,
    end: Annotated[date | None, Query(description="Inclusive upper bound.")] = None,
    z_threshold: Annotated[float, Query(ge=0.0)] = DEFAULT_Z_THRESHOLD,
    pct_threshold: Annotated[float, Query(ge=0.0)] = DEFAULT_PCT_THRESHOLD,
    direction: Annotated[Direction | None, Query()] = None,
    category: Annotated[Category | None, Query()] = None,
    sub_routing: Annotated[
        str | None,
        Query(
            description=(
                "Stored sub-bucket: share_shift, supply_chain, oil, dollar, "
                "rates, gold or country:XX. The bare value 'country' matches "
                "every country:XX."
            )
        ),
    ] = None,
    min_relevance: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
    include_prices: Annotated[bool, Query()] = False,
    include_news: Annotated[bool, Query()] = True,
    limit: Annotated[int, Query(ge=0)] = DEFAULT_LIMIT,
    refresh: Annotated[bool, Query(description="Force an ingest first.")] = False,
    period: Annotated[str, Query(description="Only used if this read ingests.")] = (DEFAULT_PERIOD),
    top_n: Annotated[int, Query(ge=0, description="Only used if this read ingests.")] = (
        DEFAULT_TOP_N
    ),
) -> TickerResponse:
    """A ticker's major moves with their news and explanations, ingesting first
    if the stored data is missing or stale (DESIGN section 5).

    The thresholds filter stored columns, so tightening or loosening them is a
    `WHERE` and never a recomputation: a move detected by the ingest stays in
    the table and a stricter query simply hides it. `filters` echoes the query
    that actually ran, so an empty list explains itself.
    """
    key = _key(ticker)

    ingested = False
    # `is_stale` already covers the never-ingested case; `needs_ingest` is
    # named anyway because it is the DESIGN section 5 wording and it short-
    # circuits the date query on a cold ticker.
    if needs_ingest(session, key) or refresh or is_stale(session, key):
        _run_ingest(
            session,
            key,
            provider,
            period=period,
            top_n=top_n,
            z_threshold=z_threshold,
            pct_threshold=pct_threshold,
            refresh=refresh,
        )
        ingested = True

    if not has_prices(session, key):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"unknown ticker or no price data for {key}",
        )

    filters = MoveFilters(
        start=start,
        end=end,
        z_threshold=z_threshold,
        pct_threshold=pct_threshold,
        direction=direction,
        category=category,
        sub_routing=sub_routing,
        min_relevance=min_relevance,
        limit=limit,
    )

    moves: list[dict[str, Any]] = []
    for move in list_moves(session, key, filters):
        if move.id is None:  # pragma: no cover - a queried row always has an id
            continue
        articles = articles_for_move(session, move.id, min_relevance) if include_news else []
        payload = move_to_dict(move)
        payload["explanation"] = explanation_to_dict(get_explanation(session, move.id))
        payload["articles"] = [article_to_dict(a, link) for a, link in articles]
        moves.append(payload)

    company = get_company(session, key)
    return TickerResponse.model_validate(
        {
            "ticker": key,
            # `peers` is a property, and `from_attributes` reads it like any
            # other attribute.
            "company": None if company is None else CompanyOut.model_validate(company),
            "ingested": ingested,
            "filters": {
                "start": None if start is None else start.isoformat(),
                "end": None if end is None else end.isoformat(),
                "z_threshold": z_threshold,
                "pct_threshold": pct_threshold,
                "direction": direction,
                "category": category,
                "sub_routing": sub_routing,
                "min_relevance": min_relevance,
                "include_prices": include_prices,
                "include_news": include_news,
                "limit": limit,
                "refresh": refresh,
                "period": period,
                "top_n": top_n,
            },
            "moves": moves,
            "prices": (
                [price_to_dict(p) for p in list_prices(session, key, start, end)]
                if include_prices
                else None
            ),
        }
    )


@router.get("/{ticker}/relations", response_model=RelationsResponse)
def read_relations(
    ticker: TickerPath,
    session: SessionDep,
    relation: Annotated[
        Relation | None,
        Query(description="Keep only this relation; omit for all five."),
    ] = None,
) -> RelationsResponse:
    """The company's stored relationship edges (v1.5 decision 10).

    A pure read: unlike the two routes around it this one never ingests,
    because edges are written by an ingest rather than derived from a window,
    and a request for them is not a request to go and build them. So the 404
    is the plain "this ticker has no stored data at all" of `_require_company`,
    and a ticker that *is* stored with no edges answers `[]` — which is the
    expected answer with no key, where the model relations are empty and the
    ETF fallback may be too (v1.5 decision 3).
    """
    key = _key(ticker)
    _require_company(session, key)
    edges = ontology.edges_of(session, key, relation)
    return RelationsResponse.model_validate(
        {"ticker": key, "edges": ontology.edges_to_dicts(edges)}
    )


@router.get("/{ticker}/moves/{move_date}", response_model=MoveOut)
def read_move(
    ticker: TickerPath,
    move_date: Annotated[date, Path(description="Trading day, YYYY-MM-DD.")],
    session: SessionDep,
    provider: ProviderDep,
    settings: SettingsDep,
    min_relevance: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
    refresh: Annotated[
        bool, Query(description="Re-fetch news and re-write the explanation.")
    ] = False,
) -> MoveOut:
    """One move with everything attached, explaining it on demand if needed.

    This is the other half of the cost control in DESIGN section 4: the ingest
    explains the top-N moves by `abs(ret_z)`, and any other move is explained
    the first time somebody asks for it — about one model call, then cached.
    """
    key = _key(ticker)

    if needs_ingest(session, key):
        _run_ingest(
            session,
            key,
            provider,
            period=settings.default_period,
            top_n=settings.default_top_n,
            z_threshold=settings.default_z_threshold,
            pct_threshold=settings.default_pct_threshold,
            refresh=False,
        )

    move = get_move(session, key, move_date)
    if move is None or move.id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no move for {key} on {move_date.isoformat()}",
        )

    explanation = get_explanation(session, move.id)
    if explanation is None or refresh:
        explanation = enrich_move(
            session,
            move,
            _require_company(session, key),
            provider,
            _news_source(settings),
            refresh=refresh,
        )

    return MoveOut.model_validate(
        move_to_dict(
            move,
            explanation,
            articles_for_move(session, move.id, min_relevance),
        )
    )
