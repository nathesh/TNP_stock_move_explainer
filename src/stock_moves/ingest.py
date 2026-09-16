"""Ingest — DESIGN sections 1 to 5 wired into one idempotent run.

This is the only module that composes the pipeline end to end: prices ->
features -> moves -> news -> scoring -> explanation. Everything it calls is a
pure function or a single-purpose module, so the orchestration logic here is
worth reading on its own:

* **Idempotent on `(ticker, date)`.** A re-ingest fetches the same `period`
  again (`yfinance` has no "since" parameter for daily bars) but the writes are
  upserts on `(ticker, date)`: existing price rows have their computed columns
  refreshed, existing moves are updated, and new days are inserted. Nothing is
  duplicated and nothing that stopped qualifying is deleted — a move detected
  under a looser threshold stays in the table, because thresholds are API
  filters (DESIGN section 1), not storage decisions.
* **Explanations are the cost control.** Only the top-N moves by `abs(ret_z)`
  across *all* stored moves for the ticker are explained at ingest; every other
  move is enriched on demand by `enrich_move`, which is what
  `GET /tickers/{ticker}/moves/{date}` calls. So a first ingest is about N
  model calls, not one per move.
* **Two knobs, not one.** `score_and_link` keeps its own `top_k` (how many
  headlines a model re-scores, default 15); `settings.top_k_articles` (8) is
  how many reach the explainer. They are deliberately different numbers.
* **Freshness and concurrency (DESIGN section 5).** `is_stale` decides whether
  a read needs an ingest, and a per-ticker `threading.Lock` stops two
  simultaneous first requests from both ingesting the same ticker.

**v1.5: the relationship layer, wired in here and nowhere else.** The edge
table, the factor attribution and the geopolitical counts are all built by
other modules; this one is where they meet the run, in one fixed order:

1. `ontology.fetch_relations` once, then `ontology.build_edges` twice with
   that one answer. The first build needs no prices and gives us the `country`
   edges; only then do we know which country ETFs to fetch, so the fitted
   factor betas — which are the `factor` edges — can only be handed to a second
   build. The two-build shape is the price of deriving factor exposure from
   prices rather than asking a model for it (plan decision 4); fetching the
   model's answer once and passing it down is what keeps that price at zero
   extra model calls — one `suggest_relations` per run, not two.
2. `build_features(..., factors=...)`, so every stored price row carries
   `macro_driver` and `macro_driver_component`.
3. `moves.sub_route` per detected move, from the competitor and
   supplier/customer co-movements, stored beside the v1 `routing`.
4. `news.geo` for the moves that came out `country:XX`, so a gate that opens
   has evidence behind it.
5. `queries_for_move` expanded along the edges, and a `MoveContext` carrying
   the countries and the geo lines into `score_and_link`, which is what lets
   the gate rule (plan decision 7) see anything at all.

`prices` is imported as a module and called as `prices.fetch_ohlcv(...)` so a
test monkeypatches one module attribute and this layer never touches the
network.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import func
from sqlmodel import Session, col, select

from stock_moves import prices
from stock_moves.explain import FALLBACK_PROVIDER_NAME, get_or_create_explanation
from stock_moves.models import Company, CompanyEdge, Explanation, GeoEvent, Move, MoveArticle, Price
from stock_moves.moves import (
    SUB_ROUTE_SHARE_SHIFT,
    SUB_ROUTE_SUPPLY_CHAIN,
    build_features,
    compute_returns,
    detect_moves,
    factor_betas,
    peer_comove,
    signed_comove,
    sub_route,
)
from stock_moves.news import (
    NewsItem,
    NewsSource,
    get_news_source,
    queries_for_move,
    window_for,
)
from stock_moves.news import geo as news_geo
from stock_moves.ontology import (
    build_edges,
    country_codes,
    fetch_relations,
    get_or_build_company,
    relation_tickers,
)
from stock_moves.prices import MARKET_ETF, PriceFetchError
from stock_moves.providers import ModelProvider, get_provider
from stock_moves.providers.base import MoveContext
from stock_moves.queries import has_prices
from stock_moves.scoring import linked_articles, score_and_link, upsert_articles
from stock_moves.settings import get_settings

__all__ = [
    "MARKET_CLOSE_HOUR_ET",
    "IngestResult",
    "enrich_move",
    "ingest_ticker",
    "is_stale",
    "last_completed_trading_day",
    "needs_ingest",
]

logger = logging.getLogger(__name__)

#: Daily bars are final after the 4pm ET close (DESIGN section 5).
MARKET_CLOSE_HOUR_ET = 16

_EASTERN = ZoneInfo("America/New_York")

_OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

#: Feature columns split by how they are stored, so one loop can build the
#: column dict for both `prices` and `moves` rows.
_FLOAT_FEATURES: tuple[str, ...] = (
    "ret",
    "gap_ret",
    "intraday_ret",
    "ret_z",
    "vol_z",
    "mkt_component",
    "sector_component",
    "idio_component",
    # v1.5: the factor attribution (`moves.FACTOR_COLUMNS`), stored on both
    # `prices` and `moves` exactly like the v1 columns beside it.
    "macro_driver_component",
)
_STR_FEATURES: tuple[str, ...] = ("routing", "regime_mkt", "regime_sector", "macro_driver")
_BOOL_FEATURES: tuple[str, ...] = ("near_earnings", "near_fomc", "near_cpi")

# The three groups together are exactly `moves.FEATURE_COLUMNS` plus
# `moves.FACTOR_COLUMNS`; `test_ingest` asserts that, so a column added to
# either tuple cannot be silently dropped here.

#: `sub_routing` values that name a country, and the rest of the string is the
#: ISO-3166 alpha-2 code (v1.5 decision 6).
_COUNTRY_SUB_ROUTING_PREFIX = "country:"

# --------------------------------------------------------------------------- #
# Per-ticker locks (DESIGN section 5)
# --------------------------------------------------------------------------- #

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(ticker: str) -> threading.Lock:
    """The lock for one ticker, created on first sight.

    The guard is only around the dict lookup; the returned lock is held for the
    whole ingest by the caller. Two simultaneous first requests for the same
    ticker therefore serialise, while different tickers ingest in parallel.
    """
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(ticker, threading.Lock())


# --------------------------------------------------------------------------- #
# Freshness (DESIGN section 5)
# --------------------------------------------------------------------------- #


def last_completed_trading_day(now: datetime | None = None) -> date:
    """The most recent day whose daily bar is final.

    Today if today is a weekday and it is at or past the 4pm ET close;
    otherwise the last weekday strictly before today. `now` is interpreted in
    America/New_York (a naive value is assumed to already be ET) and defaults
    to the current time.

    **Market holidays are ignored.** A holiday calendar is a dependency and a
    yearly maintenance burden, and the cost of ignoring it is one wasted
    re-fetch on the day after a holiday: the ticker is reported stale, the
    fetch returns no new bar, and the upsert inserts nothing. Being wrong in
    that direction is cheap; missing a real trading day would not be.
    """
    if now is None:
        current = datetime.now(_EASTERN)
    elif now.tzinfo is None:
        current = now.replace(tzinfo=_EASTERN)
    else:
        current = now.astimezone(_EASTERN)

    today = current.date()
    if today.weekday() < 5 and current.hour >= MARKET_CLOSE_HOUR_ET:
        return today

    day = today - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def needs_ingest(session: Session, ticker: str) -> bool:
    """Whether the ticker has never been ingested at all."""
    return not has_prices(session, ticker)


def is_stale(session: Session, ticker: str) -> bool:
    """Whether a read should trigger an ingest: no prices, or a missing day.

    True when nothing is stored, or when the newest stored price date is older
    than `last_completed_trading_day()`. False means "at most one ingest per
    trading day per ticker" has already been paid.
    """
    statement = (
        select(col(Price.date))
        .where(col(Price.ticker) == ticker.strip().upper())
        .order_by(col(Price.date).desc())
        .limit(1)
    )
    newest = session.exec(statement).first()
    if newest is None:
        return True
    return newest < last_completed_trading_day()


# --------------------------------------------------------------------------- #
# Row building
# --------------------------------------------------------------------------- #


def _opt_float(value: Any) -> float | None:
    """A finite float, or None for NaN/inf/None — SQLite cannot store a NaN."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _req_float(value: Any) -> float:
    """A finite float with 0.0 as the fallback, for the NOT NULL columns."""
    number = _opt_float(value)
    return 0.0 if number is None else number


def _opt_str(value: Any) -> str | None:
    """A string, or None for None/NaN (pandas object columns hold both)."""
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    text = str(value)
    return None if text in ("nan", "None", "") else text


def _price_columns(row: pd.Series) -> dict[str, Any]:
    """The stored `prices` columns for one feature row.

    OHLCV is NOT NULL, so a non-finite value there falls back to 0.0 rather
    than failing the insert; `normalize_ohlcv` has already dropped rows with no
    close, so this is belt-and-braces. Derived columns are nullable and a NaN
    (the rolling warm-up) becomes None.
    """
    values: dict[str, Any] = {name: _req_float(row[name]) for name in _OHLCV_COLUMNS}
    values.update({name: _opt_float(row[name]) for name in _FLOAT_FEATURES})
    values.update({name: _opt_str(row[name]) for name in _STR_FEATURES})
    values.update({name: bool(row[name]) for name in _BOOL_FEATURES})
    return values


def _move_columns(
    row: pd.Series,
    peer_return: Any,
    rival_return: Any = None,
    chain_return: Any = None,
) -> dict[str, Any]:
    """The stored `moves` columns for one detected row.

    `ret_z` is NOT NULL but is NaN during the 20-day warm-up, where the
    percentage gate is what fired. It is stored as 0.0: SQLite turns a NaN into
    a NULL and the insert would fail, and 0.0 sorts such a day last in the
    `abs(ret_z)` ordering, which is exactly where a day with no dispersion
    estimate belongs.

    v1.5 adds three stored numbers and one stored label: the competitor and
    supplier/customer co-movements, and the `sub_routing` they and the day's
    `macro_driver` imply. `peer_comove` is untouched beside them — it is still
    the mean over `companies.peers_json`, and plan decision 2 keeps its name
    and its meaning.
    """
    values: dict[str, Any] = {
        name: _opt_float(row[name]) for name in _FLOAT_FEATURES if name != "ret"
    }
    ret = _req_float(row["ret"])
    values["ret"] = ret
    values["ret_z"] = _req_float(row["ret_z"])
    values.update({name: _opt_str(row[name]) for name in _STR_FEATURES})
    values.update({name: bool(row[name]) for name in _BOOL_FEATURES})
    # `routing` and `direction` are NOT NULL; both are only ever None on a row
    # with no return, which cannot be detected as a move.
    values["routing"] = values["routing"] or "company"
    values["direction"] = _opt_str(row.get("direction")) or ("down" if ret < 0 else "up")
    values["peer_comove"] = _opt_float(peer_return)
    values["rival_comove"] = _opt_float(rival_return)
    values["chain_comove"] = _opt_float(chain_return)
    values["sub_routing"] = sub_route(
        str(values["routing"]),
        ret,
        values["rival_comove"],
        values["chain_comove"],
        values["macro_driver"],
    )
    return values


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #


@dataclass
class IngestResult:
    """What one ingest run stored, counted from the tables afterwards.

    The counts are totals for the ticker rather than deltas for this run, which
    is what makes them useful as an idempotence check: a second run with the
    same arguments returns the same numbers.

    `provider` is the name stored on this ticker's explanations, not the
    configured provider: a run whose every model call degraded reports the
    keyless provider.
    """

    ticker: str
    period: str
    top_n: int
    n_prices: int
    n_moves: int
    n_articles: int
    n_explanations: int
    provider: str
    news_source: str
    #: v1.5. `n_edges` is every `company_edges` row for this ticker — the four
    #: model relations plus the fitted `factor` edges. `n_geo_events` counts
    #: `geo_events` rows for the countries this company has an edge to, which
    #: is the only slice of that table the ticker is responsible for; it is 0
    #: for a company with no country edges, and therefore 0 without a key.
    n_edges: int = 0
    n_geo_events: int = 0


# --------------------------------------------------------------------------- #
# v1.5: reading the edges back for one move
# --------------------------------------------------------------------------- #


def _country_of(sub_routing: str | None) -> str | None:
    """The ISO-3166 code in a `country:XX` sub-routing, or None for any other."""
    text = (sub_routing or "").strip()
    if not text.startswith(_COUNTRY_SUB_ROUTING_PREFIX):
        return None
    code = text[len(_COUNTRY_SUB_ROUTING_PREFIX) :].strip().upper()
    return code or None


def _display_name(session: Session, ticker: str) -> str:
    """A related ticker's company name if we happen to have stored one.

    "Only if cheap" is the rule: a `companies` row is one primary-key lookup,
    and a related ticker usually has none, in which case the symbol itself is
    handed on — `news.clean_company_name` takes either.
    """
    key = ticker.strip().upper()
    company = session.get(Company, key)
    name = (company.name or "").strip() if company is not None else ""
    return name or key


def _names_by_sign(
    returns: pd.DataFrame | None,
    tickers: Sequence[str],
    day: date,
    ret: float,
    *,
    opposite: bool,
) -> list[str]:
    """Which of `tickers` moved against (or with) a `ret` of `ret` on `day`.

    The mean is what `sub_route` judged; this is the same tape read name by
    name, because a query is per rival. A name with no bar that day, a flat
    day, and a move of exactly zero all yield nothing: the sign of zero is not
    a direction.
    """
    move = _opt_float(ret)
    if returns is None or returns.empty or not tickers or not move:
        return []
    stamp = pd.Timestamp(day)
    if stamp not in returns.index:
        return []
    row = returns.loc[stamp]
    wanted = (-1.0 if opposite else 1.0) * (1.0 if move > 0 else -1.0)

    names: list[str] = []
    for ticker in tickers:
        if ticker not in returns.columns:
            continue
        value = _opt_float(row[ticker])
        if not value:
            continue
        if (1.0 if value > 0 else -1.0) == wanted:
            names.append(ticker)
    return names


def _chain_tickers(session: Session, ticker: str) -> list[str]:
    """Suppliers and customers as one de-duplicated, sorted list."""
    suppliers = relation_tickers(session, ticker, "supplier")
    customers = relation_tickers(session, ticker, "customer")
    return sorted(set(suppliers) | set(customers))


def _edge_names(
    session: Session,
    move: Move,
    related_returns: pd.DataFrame | None,
    period: str,
) -> tuple[list[str], list[str]]:
    """`(rival_names, chain_names)` for the move's retrieval expansion.

    Only the sub-bucket that fired asks for names, so at most one of the two
    lists is ever non-empty and a move with no edge story costs nothing.
    `related_returns` is the frame `ingest_ticker` already fetched for the
    co-movements; the on-demand path (`GET /tickers/{t}/moves/{date}`) has no
    such frame and fetches the few names it needs itself.
    """
    sub_routing = (move.sub_routing or "").strip()
    if sub_routing == SUB_ROUTE_SHARE_SHIFT:
        tickers = relation_tickers(session, move.ticker, "competitor")
        opposite = True
    elif sub_routing == SUB_ROUTE_SUPPLY_CHAIN:
        tickers = _chain_tickers(session, move.ticker)
        opposite = False
    else:
        return [], []

    if not tickers:
        return [], []

    frame = related_returns
    if frame is None:
        frame = prices.fetch_peer_returns(tickers, period)
    moved = _names_by_sign(frame, tickers, move.date, move.ret, opposite=opposite)
    names = [_display_name(session, ticker) for ticker in moved]
    return (names, []) if opposite else ([], names)


def _move_context(
    session: Session,
    move: Move,
    company: Company,
    start: date,
    end: date,
) -> MoveContext:
    """The v1 context plus the company's edges and the window's geo lines.

    `MoveContext.from_objects` knows only the two rows it is handed, and the
    edges and the geopolitical counts live in two other tables — so they are
    read here and replaced onto it. Without this the gate rule sees no
    countries and shuts on every geopolitical headline.
    """
    key = company.ticker.strip().upper()
    countries = country_codes(session, key)
    return replace(
        MoveContext.from_objects(move, company),
        competitors=tuple(relation_tickers(session, key, "competitor")),
        suppliers=tuple(relation_tickers(session, key, "supplier")),
        customers=tuple(relation_tickers(session, key, "customer")),
        countries=tuple(countries),
        geo_events=tuple(
            news_geo.geo_event_lines(session, [code for code, _ in countries], start, end)
        ),
    )


# --------------------------------------------------------------------------- #
# Enrichment of one move
# --------------------------------------------------------------------------- #


def enrich_move(
    session: Session,
    move: Move,
    company: Company,
    provider: ModelProvider,
    news_source: NewsSource,
    *,
    refresh: bool = False,
    news_limit: int | None = None,
    top_k: int | None = None,
    related_returns: pd.DataFrame | None = None,
) -> Explanation:
    """Attach news to one move and explain it, reusing whatever is cached.

    Three layers of cache, cheapest first: a stored explanation short-circuits
    everything; stored article links short-circuit the news fetch; and only
    then is the source queried. `refresh` skips all three.

    A failing news source is logged and swallowed per query — the move is still
    explained, from the decomposition alone, which is the case the
    `unexplained` category exists for.

    **v1.5.** The query list is expanded along the move's edges (plan decision
    8) and the scoring is handed a context carrying the company's countries and
    the window's geopolitical counts, which is what the gate rule reads.
    `related_returns` is an optional frame of daily returns for the related
    tickers — one column per symbol — so the caller that already fetched them
    (`ingest_ticker`) is not made to fetch them again per move; leaving it None
    makes this function fetch the few names it needs, and only for the two
    sub-buckets that name names.
    """
    if move.id is None:
        raise ValueError("enrich_move needs a persisted move: move.id is None")

    settings = get_settings()
    limit = settings.news_limit if news_limit is None else news_limit
    articles_for_prompt = settings.top_k_articles if top_k is None else top_k

    existing = session.exec(select(Explanation).where(col(Explanation.move_id) == move.id)).first()
    if existing is not None and not refresh:
        return existing

    if refresh or not linked_articles(session, move.id):
        start, end = window_for(move.date)
        rival_names, chain_names = _edge_names(
            session, move, related_returns, settings.default_period
        )
        items: list[NewsItem] = []
        for bucket, query in queries_for_move(
            move.routing,
            company.name,
            company.ticker,
            company.peers,
            company.industry,
            sub_routing=move.sub_routing,
            rival_names=rival_names,
            chain_names=chain_names,
            country=_country_of(move.sub_routing),
        ):
            try:
                found = news_source.search(query, start, end, limit=limit)
            # A dead or rate-limited news source must not fail the ingest.
            except Exception:
                logger.warning(
                    "news search failed for %s %s (%s bucket)",
                    company.ticker,
                    move.date,
                    bucket,
                    exc_info=True,
                )
                continue
            # The source does not know which bucket asked, so tag it here.
            items.extend(replace(item, bucket=bucket) for item in found)

        if items:
            stored = upsert_articles(session, items)
            # `score_and_link` keeps its own default top_k (15): that is how
            # many headlines a model re-scores, not how many are explained.
            score_and_link(
                session,
                move,
                company,
                stored,
                provider,
                context=_move_context(session, move, company, start, end),
            )

    return get_or_create_explanation(
        session,
        move,
        company,
        provider,
        top_k=articles_for_prompt,
        refresh=refresh,
    )


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


def _fetch_sector_etf(etf: str | None, period: str) -> pd.DataFrame | None:
    """The sector ETF frame, or None when there is no ETF or it cannot be had.

    A missing sector ETF only costs the sector leg of the decomposition
    (`build_features` regresses on the market alone), so it is never fatal.
    """
    if not etf:
        return None
    try:
        return prices.fetch_ohlcv(etf, period)
    except PriceFetchError:
        logger.warning("no price data for sector ETF %s; continuing without it", etf)
        return None


def _write_prices(session: Session, ticker: str, features: pd.DataFrame) -> None:
    """Upsert one `prices` row per feature row, keyed on `(ticker, date)`."""
    stored = {
        row.date: row
        for row in session.exec(select(Price).where(col(Price.ticker) == ticker)).all()
    }

    for stamp, row in features.iterrows():
        day = pd.Timestamp(stamp).date()
        values = _price_columns(row)
        price = stored.get(day)
        if price is None:
            session.add(Price(ticker=ticker, date=day, **values))
        else:
            for name, value in values.items():
                setattr(price, name, value)
            session.add(price)
    session.commit()


def _write_moves(
    session: Session,
    ticker: str,
    detected: pd.DataFrame,
    comove: pd.Series,
    rival_comove: pd.Series,
    chain_comove: pd.Series,
) -> None:
    """Upsert one `moves` row per detected row, keyed on `(ticker, date)`.

    Moves that no longer qualify are left alone: the thresholds are query
    filters, so unstoring a move would hide it from a looser query.
    """
    stored = {
        row.date: row for row in session.exec(select(Move).where(col(Move.ticker) == ticker)).all()
    }

    for stamp, row in detected.iterrows():
        day = pd.Timestamp(stamp).date()
        values = _move_columns(
            row,
            comove.get(stamp),
            rival_comove.get(stamp),
            chain_comove.get(stamp),
        )
        move = stored.get(day)
        if move is None:
            session.add(Move(ticker=ticker, date=day, **values))
        else:
            for name, value in values.items():
                setattr(move, name, value)
            session.add(move)
    session.commit()


def _factor_proxies(session: Session, ticker: str) -> dict[str, str]:
    """The `{factor key: proxy symbol}` map to fetch returns for.

    The four macro proxies always, plus one country ETF per `country` edge that
    has one. A country outside `prices.COUNTRY_ETFS` keeps its edge and is
    simply skipped here — plan decision 4 names that limitation rather than
    inventing a proxy for it, and the consequence is that such a company can
    only ever be driven by `oil` or `dollar`.
    """
    proxies: dict[str, str] = dict(prices.FACTOR_ETFS)
    for code, _weight in country_codes(session, ticker):
        symbol = prices.COUNTRY_ETFS.get(code.strip().upper())
        if symbol:
            proxies[code.strip().upper()] = symbol
    return proxies


def _latest_betas(betas: pd.DataFrame) -> dict[str, float]:
    """The last finite beta of each factor column, as the stored edge weight.

    One number per factor is all an edge can carry, and the most recent fitted
    beta is the one that describes the company now. Columns that never fitted —
    a proxy with less than the full window of overlapping days — contribute no
    edge at all rather than a zero, because "no exposure measured" and "no
    exposure" are different claims.
    """
    latest: dict[str, float] = {}
    for name in betas.columns:
        for value in reversed(betas[name].to_list()):
            beta = _opt_float(value)
            if beta is not None:
                latest[str(name)] = beta
                break
    return latest


def _related_returns(rival_returns: pd.DataFrame, chain_returns: pd.DataFrame) -> pd.DataFrame:
    """The competitor and supplier/customer return frames as one, columns deduped.

    Handed down to `enrich_move` so the names behind a `share_shift` or a
    `supply_chain` query are read off prices this run already fetched. An empty
    frame is a real answer — "these names have no returns" — and is passed on
    as such rather than as None, which would send `enrich_move` back to the
    network for a fetch that has already failed.
    """
    frames = [frame for frame in (rival_returns, chain_returns) if not frame.empty]
    if not frames:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    combined = pd.concat(frames, axis=1)
    return combined.loc[:, ~combined.columns.duplicated()]


def _write_geo_events(
    session: Session,
    ticker: str,
    news_source: NewsSource,
    limit: int,
) -> None:
    """Store geopolitical headline counts for the `country:XX` moves (decision 5).

    Only for moves whose stored `sub_routing` names a country, only for
    countries the company has a `country` edge to, and only over each move's own
    ±1 day window — the three conditions that keep `geo_events` small. A company
    with no country edges does nothing at all here, which is every company when
    there is no model key (plan decision 3). `(country, window)` pairs are
    de-duplicated, so two moves a day apart cost one query rather than two.

    The moves are read back from the table rather than taken from this run's
    detection, so a country edge that only appears now also picks up the
    country-routed moves stored earlier. The upsert is idempotent on
    `(date, country, news_source)`, so a re-run rewrites nothing.

    A failing news source is swallowed exactly as it is in `enrich_move`: the
    evidence behind a gate is worth a network call, never an ingest.
    """
    codes = {code for code, _ in country_codes(session, ticker)}
    if not codes:
        return

    country_moves = session.exec(
        select(Move)
        .where(col(Move.ticker) == ticker)
        .where(col(Move.sub_routing).startswith(_COUNTRY_SUB_ROUTING_PREFIX))
    ).all()

    windows: dict[tuple[str, date, date], None] = {}
    for move in country_moves:
        code = _country_of(move.sub_routing)
        if code is None or code not in codes:
            continue
        start, end = window_for(move.date)
        windows[(code, start, end)] = None

    drafts: list[news_geo.GeoEventDraft] = []
    for code, start, end in windows:
        try:
            drafts.extend(news_geo.fetch_geo_events(news_source, code, start, end, limit))
        # A dead or rate-limited news source must not fail the ingest.
        except Exception:
            logger.warning(
                "geo search failed for %s %s (%s to %s)", ticker, code, start, end, exc_info=True
            )
            continue

    news_geo.upsert_geo_events(session, drafts)


def _count(session: Session, statement: Any) -> int:
    return int(session.exec(statement).one() or 0)


def _stored_provider_name(session: Session, ticker: str, configured: str) -> str:
    """The provider name actually written on this ticker's explanations.

    `explain.get_or_create_explanation` stamps a degraded row with the keyless
    provider's name, so the configured name would over-report a run in which
    every model call failed. Report the fallback only when *every* stored row
    is a fallback row; a single model-written row keeps the configured name.
    """
    stored = set(
        session.exec(
            select(col(Explanation.provider))
            .select_from(Explanation)
            .join(Move, col(Move.id) == col(Explanation.move_id))
            .where(col(Move.ticker) == ticker)
        ).all()
    )
    if stored and stored == {FALLBACK_PROVIDER_NAME}:
        return FALLBACK_PROVIDER_NAME
    return configured


def ingest_ticker(
    session: Session,
    ticker: str,
    *,
    period: str | None = None,
    top_n: int | None = None,
    z_threshold: float | None = None,
    pct_threshold: float | None = None,
    refresh: bool = False,
    provider: ModelProvider | None = None,
    news_source: NewsSource | None = None,
) -> IngestResult:
    """Run the whole pipeline for one ticker and return what is stored.

    Every `None` argument falls back to its `Settings` default. The run holds
    the ticker's lock from the first fetch to the last commit, so a second
    concurrent call waits and then finds the work already done.
    """
    settings = get_settings()
    key = ticker.strip().upper()
    run_period = settings.default_period if period is None else period
    run_top_n = settings.default_top_n if top_n is None else top_n
    run_z = settings.default_z_threshold if z_threshold is None else z_threshold
    run_pct = settings.default_pct_threshold if pct_threshold is None else pct_threshold

    model = get_provider(settings) if provider is None else provider
    news = (
        get_news_source(
            settings.news_source,
            timeout_s=settings.http_timeout_s,
            throttle_s=settings.gdelt_throttle_s,
        )
        if news_source is None
        else news_source
    )

    with _lock_for(key):
        # 1. Ontology: identity, sector ETF and peers, cached per company.
        company = get_or_build_company(session, key, model, refresh=refresh)

        # 1b. Edges, written in two passes and deliberately so (v1.5 decision
        #     4). The model is asked exactly once, here: the answer —
        #     competitor / supplier / customer / country — has to be in hand
        #     first because the country edges are what decide which country
        #     ETFs are worth fetching. Only once those returns are in can the
        #     betas be fitted, so the `factor` edges can only be written by a
        #     second pass. Both passes are handed the same `Relations`, so the
        #     second costs no model call; `build_edges` is replace-all per
        #     relation, so it rewrites the same four relations to the same rows
        #     and adds the fifth. Nothing accumulates and nothing is duplicated.
        relations = fetch_relations(company, model)
        build_edges(session, company, model, factor_betas=None, relations=relations)

        # 2. Inputs. The stock is required; the market and the sector ETF are
        #    the decomposition's regressors, and only the ETF is optional.
        stock = prices.fetch_ohlcv(key, run_period)
        market = prices.fetch_ohlcv(MARKET_ETF, run_period)
        etf = _fetch_sector_etf(company.sector_etf, run_period)
        earnings = prices.fetch_earnings_dates(key)
        peers = company.peers
        peer_returns = prices.fetch_peer_returns(peers, run_period) if peers else pd.DataFrame()

        # 2b. Factor proxies -> betas -> the `factor` edges. An empty frame is
        #     the honest "no proxy resolved" answer (every symbol failed, or
        #     `fetch_factor_returns` was handed nothing): the factor pass is
        #     then skipped entirely, `factor_betas=None` leaves any previously
        #     fitted edges alone, and `build_features` is called the v1 way.
        factor_returns = prices.fetch_factor_returns(_factor_proxies(session, key), run_period)
        has_factors = not factor_returns.empty
        if has_factors:
            betas = factor_betas(compute_returns(stock)["ret"], factor_returns)
            build_edges(
                session,
                company,
                model,
                factor_betas=_latest_betas(betas),
                relations=relations,
            )

        # 3. Features -> prices. With factors, every row also carries the
        #    `macro_driver` attribution (`moves.FACTOR_COLUMNS`).
        features = build_features(
            stock, market, etf, earnings, factors=factor_returns if has_factors else None
        )
        _write_prices(session, key, features)

        # 4. Detection -> moves, and the two v1.5 co-movements beside the v1
        #    one. `peer_comove` still averages `companies.peers_json`;
        #    `rival_comove` averages the `competitor` edges and `chain_comove`
        #    the `supplier` and `customer` edges, and those two are what
        #    `sub_route` judges (decision 6).
        detected = detect_moves(features, run_z, run_pct)
        dates = pd.DatetimeIndex(detected.index)
        comove = (
            peer_comove(peer_returns, dates)
            if peers and not peer_returns.empty
            else pd.Series(index=detected.index, dtype=float)
        )
        rivals = relation_tickers(session, key, "competitor")
        chain = _chain_tickers(session, key)
        rival_returns = prices.fetch_peer_returns(rivals, run_period) if rivals else pd.DataFrame()
        chain_returns = prices.fetch_peer_returns(chain, run_period) if chain else pd.DataFrame()
        _write_moves(
            session,
            key,
            detected,
            comove,
            signed_comove(rival_returns, dates),
            signed_comove(chain_returns, dates),
        )

        # 4b. Geopolitical counts for the moves that came out `country:XX`.
        _write_geo_events(session, key, news, settings.news_limit)

        # 5. Explain the biggest moves on record — not just this run's — so the
        #    top-N is stable as history accumulates. Everything else waits for
        #    `enrich_move` on demand. The related returns are handed down so a
        #    move that expands along its edges does not refetch them.
        related_returns = _related_returns(rival_returns, chain_returns)
        top = session.exec(
            select(Move)
            .where(col(Move.ticker) == key)
            .order_by(func.abs(col(Move.ret_z)).desc(), col(Move.date).desc())
            .limit(max(run_top_n, 0))
        ).all()
        for move in top:
            enrich_move(
                session,
                move,
                company,
                model,
                news,
                refresh=refresh,
                news_limit=settings.news_limit,
                top_k=settings.top_k_articles,
                related_returns=related_returns,
            )

        # 6. Counts read back from the tables, so they are totals and not a
        #    tally of what this particular run happened to insert.
        n_prices = _count(
            session,
            select(func.count(col(Price.id))).where(col(Price.ticker) == key),
        )
        n_moves = _count(
            session,
            select(func.count(col(Move.id))).where(col(Move.ticker) == key),
        )
        n_articles = _count(
            session,
            select(func.count(func.distinct(col(MoveArticle.article_id))))
            .select_from(MoveArticle)
            .join(Move, col(Move.id) == col(MoveArticle.move_id))
            .where(col(Move.ticker) == key),
        )
        n_explanations = _count(
            session,
            select(func.count(col(Explanation.id)))
            .select_from(Explanation)
            .join(Move, col(Move.id) == col(Explanation.move_id))
            .where(col(Move.ticker) == key),
        )
        n_edges = _count(
            session,
            select(func.count()).select_from(CompanyEdge).where(col(CompanyEdge.src) == key),
        )
        codes = [code for code, _ in country_codes(session, key)]
        n_geo_events = (
            _count(
                session,
                select(func.count()).select_from(GeoEvent).where(col(GeoEvent.country).in_(codes)),
            )
            if codes
            else 0
        )
        provider_name = _stored_provider_name(session, key, model.name)

    return IngestResult(
        ticker=key,
        period=run_period,
        top_n=run_top_n,
        n_prices=n_prices,
        n_moves=n_moves,
        n_articles=n_articles,
        n_explanations=n_explanations,
        provider=provider_name,
        news_source=getattr(news, "name", "unknown"),
        n_edges=n_edges,
        n_geo_events=n_geo_events,
    )
