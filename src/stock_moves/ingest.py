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

`prices` is imported as a module and called as `prices.fetch_ohlcv(...)` so a
test monkeypatches one module attribute and this layer never touches the
network.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import func
from sqlmodel import Session, col, select

from stock_moves import prices
from stock_moves.explain import FALLBACK_PROVIDER_NAME, get_or_create_explanation
from stock_moves.models import Company, Explanation, Move, MoveArticle, Price
from stock_moves.moves import build_features, detect_moves, peer_comove
from stock_moves.news import (
    NewsItem,
    NewsSource,
    get_news_source,
    queries_for_move,
    window_for,
)
from stock_moves.ontology import get_or_build_company
from stock_moves.prices import MARKET_ETF, PriceFetchError
from stock_moves.providers import ModelProvider, get_provider
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
)
_STR_FEATURES: tuple[str, ...] = ("routing", "regime_mkt", "regime_sector")
_BOOL_FEATURES: tuple[str, ...] = ("near_earnings", "near_fomc", "near_cpi")

# The three groups together are exactly `moves.FEATURE_COLUMNS`; `test_ingest`
# asserts that, so a column added there cannot be silently dropped here.

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


def _move_columns(row: pd.Series, peer_return: Any) -> dict[str, Any]:
    """The stored `moves` columns for one detected row.

    `ret_z` is NOT NULL but is NaN during the 20-day warm-up, where the
    percentage gate is what fired. It is stored as 0.0: SQLite turns a NaN into
    a NULL and the insert would fail, and 0.0 sorts such a day last in the
    `abs(ret_z)` ordering, which is exactly where a day with no dispersion
    estimate belongs.
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
) -> Explanation:
    """Attach news to one move and explain it, reusing whatever is cached.

    Three layers of cache, cheapest first: a stored explanation short-circuits
    everything; stored article links short-circuit the news fetch; and only
    then is the source queried. `refresh` skips all three.

    A failing news source is logged and swallowed per query — the move is still
    explained, from the decomposition alone, which is the case the
    `unexplained` category exists for.
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
        items: list[NewsItem] = []
        for bucket, query in queries_for_move(
            move.routing,
            company.name,
            company.ticker,
            company.peers,
            company.industry,
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
            score_and_link(session, move, company, stored, provider)

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


def _write_moves(session: Session, ticker: str, detected: pd.DataFrame, comove: pd.Series) -> None:
    """Upsert one `moves` row per detected row, keyed on `(ticker, date)`.

    Moves that no longer qualify are left alone: the thresholds are query
    filters, so unstoring a move would hide it from a looser query.
    """
    stored = {
        row.date: row for row in session.exec(select(Move).where(col(Move.ticker) == ticker)).all()
    }

    for stamp, row in detected.iterrows():
        day = pd.Timestamp(stamp).date()
        values = _move_columns(row, comove.get(stamp))
        move = stored.get(day)
        if move is None:
            session.add(Move(ticker=ticker, date=day, **values))
        else:
            for name, value in values.items():
                setattr(move, name, value)
            session.add(move)
    session.commit()


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

        # 2. Inputs. The stock is required; the market and the sector ETF are
        #    the decomposition's regressors, and only the ETF is optional.
        stock = prices.fetch_ohlcv(key, run_period)
        market = prices.fetch_ohlcv(MARKET_ETF, run_period)
        etf = _fetch_sector_etf(company.sector_etf, run_period)
        earnings = prices.fetch_earnings_dates(key)
        peers = company.peers
        peer_returns = prices.fetch_peer_returns(peers, run_period) if peers else pd.DataFrame()

        # 3. Features -> prices.
        features = build_features(stock, market, etf, earnings)
        _write_prices(session, key, features)

        # 4. Detection -> moves.
        detected = detect_moves(features, run_z, run_pct)
        comove = (
            peer_comove(peer_returns, pd.DatetimeIndex(detected.index))
            if peers and not peer_returns.empty
            else pd.Series(index=detected.index, dtype=float)
        )
        _write_moves(session, key, detected, comove)

        # 5. Explain the biggest moves on record — not just this run's — so the
        #    top-N is stable as history accumulates. Everything else waits for
        #    `enrich_move` on demand.
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
    )
