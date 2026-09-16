"""The v1.5 eval: three ticker-days whose `sub_routing` was decided before the build.

This is a script, not a test. It uses the network — prices from yfinance and
headlines from the configured `NewsSource` — so it can never run inside pytest,
and it takes minutes rather than seconds.

    uv run python scripts/eval_v15.py                  # a throwaway database
    uv run python scripts/eval_v15.py --db data/eval.db  # keep it to poke at

The cases mirror the table in `docs/v1.5-plan.md`, which is the source of truth
for v1.5. Each one accepts a *set* of answers rather than one, because two of
the three rows are genuinely ambiguous ahead of the data: an export-licence day
can route `company` (Nvidia's own charge) or `macro` (the country exposure), and
the plan says so. Routing and sub-routing are judged separately, so a run that
routes correctly but sub-routes wrongly says which half broke.

The plan's pass rule has a second half — "at least one cited article names the
event" — so each case also carries the words that would name its event, and the
citation is judged as a third verdict. A case whose move is outside the ingest's
explained top-N has no stored explanation to read, so this script explains it on
demand first, exactly the way `GET /tickers/{ticker}/moves/{date}` does: same
session, same provider, same news source. The first three cited titles are
printed for every case, because *which* headline the retrieval put first is as
much of the result as the verdict is.

Exit codes: 0 every case passed, 1 at least one failed (a missing move on the
date counts as a failure — the day is supposed to clear the move thresholds).

Without a working key the run still completes, but `suggest_relations` returns
competitors only, so there are no `country` edges: the geo gate never opens or
shuts, geopolitical headlines score like any other, and a `country:XX`
sub-routing is unreachable. The last line of the output says so.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO_ROOT_GUESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT_GUESS / "src"))

from dotenv import load_dotenv

from stock_moves.db import Session, configure_engine, init_db, session_scope
from stock_moves.explain import FALLBACK_PROVIDER_NAME
from stock_moves.ingest import IngestResult, enrich_move, ingest_ticker
from stock_moves.models import Article, Explanation, Move
from stock_moves.news import NewsSource, get_news_source
from stock_moves.providers import ModelProvider, get_provider
from stock_moves.queries import get_company, get_explanation, get_move
from stock_moves.settings import REPO_ROOT, get_settings

__all__ = ["EVAL_CASES", "EvalCase", "main"]

EXIT_OK = 0
EXIT_FAILED = 1

#: Two years, so the 200-day SMA behind `regime_*` has warmed up by the time
#: the 2025 dates arrive. The v1.5 plan makes this the default everywhere.
PERIOD = "2y"

#: How many cited titles are printed per case. Three is enough to see whether
#: the retrieval put the day's real story first or buried it.
CITED_SHOWN = 3


@dataclass(frozen=True)
class EvalCase:
    """One ticker-day and the answers that count as correct.

    `expected_routing` and `expected_sub_routing` are tuples of *accepted*
    values, not single answers; `(None,)` means the correct answer is no
    sub-routing at all (an `industry` day has no sub-bucket).

    `citation_keywords` is the other half of the plan's pass rule: words that a
    headline about *this* event would contain. One cited title containing one of
    them is enough — the rule is "names the event", not "is the best story on
    the day" — and the match is a case-insensitive substring of the title alone,
    not of the source.
    """

    ticker: str
    on: date
    what_happened: str
    expected_routing: tuple[str, ...]
    expected_sub_routing: tuple[str | None, ...]
    citation_keywords: tuple[str, ...]


#: The eval table from `docs/v1.5-plan.md`, decided before the build and
#: checked against yfinance on 2026-09-15: NVDA 2025-01-27 is -17.0% (z -5.8),
#: AAPL 2025-04-03 is -9.2% (z -4.7), and NVDA 2025-04-16 is -6.9% at z -1.2,
#: so that one enters through the 2% gate rather than the z rule.
EVAL_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        ticker="NVDA",
        on=date(2025, 4, 16),
        what_happened="export-licence requirement on H20, $5.5B charge; TSM down 3.6% the same day",
        expected_routing=("company", "macro"),
        expected_sub_routing=("supply_chain", "country:CN", "country:TW"),
        citation_keywords=("H20", "export", "China", "charge"),
    ),
    EvalCase(
        ticker="NVDA",
        on=date(2025, 1, 27),
        what_happened="DeepSeek day, chip names down together (AMD -6.4%, so rivals moved *with* it)",
        expected_routing=("industry",),
        expected_sub_routing=(None,),
        citation_keywords=("DeepSeek",),
    ),
    EvalCase(
        ticker="AAPL",
        on=date(2025, 4, 3),
        what_happened="tariff announcement, the day after 2025-04-02",
        expected_routing=("macro",),
        expected_sub_routing=("country:CN", "dollar"),
        citation_keywords=("tariff", "tariffs"),
    ),
)


def main(argv: list[str] | None = None) -> int:
    """Ingest each ticker once, judge every case, and return the exit code."""
    args = _parse_args(argv)

    # Explicit path rather than dotenv's walk-up search, so the script reads
    # the same file the app reads however it was invoked.
    load_dotenv(REPO_ROOT / ".env")
    get_settings.cache_clear()
    settings = get_settings()

    db_path = Path(args.db) if args.db else _temp_db_path()
    configure_engine(f"sqlite:///{db_path}")
    init_db()

    # The same two objects the API resolves per request, built once here and
    # handed to every on-demand explanation, so a case outside the ingest's
    # top-N is explained by exactly the path `read_move` would have taken.
    provider = get_provider(settings)
    news_source = get_news_source(
        settings.news_source,
        timeout_s=settings.http_timeout_s,
        throttle_s=settings.gdelt_throttle_s,
    )

    configured = provider.name
    print(f"db:       {db_path}")
    print(f"env:      {REPO_ROOT / '.env'}")
    print(f"period:   {PERIOD}")
    print(f"provider: {configured} (configured)")
    print()

    results, ingest_failures = _ingest_all(args.db is not None)
    print()

    failures: list[str] = list(ingest_failures.values())
    failed_cases = 0
    for case in EVAL_CASES:
        if case.ticker in ingest_failures:
            print(f"{case.ticker} {case.on.isoformat()}  skipped: the ingest above failed.")
            print()
            failed_cases += 1
            continue
        case_failures = _judge(case, provider, news_source)
        if case_failures:
            failed_cases += 1
            failures.extend(case_failures)
        print()

    used = _provider_used(results, configured)
    print(f"{len(EVAL_CASES) - failed_cases}/{len(EVAL_CASES)} cases passed")
    for failure in failures:
        print(f"  FAIL {failure}")
    print()
    print(f"provider used: {used}")
    if used == FALLBACK_PROVIDER_NAME:
        print(
            "  heuristic: no country edges are suggested without a working key, so the "
            "geo gate never opens or shuts, geopolitical headlines score like any "
            "other, and no `country:XX` sub-routing can fire."
        )
    return EXIT_FAILED if failures else EXIT_OK


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="eval_v15.py",
        description=(
            "Run the three v1.5 eval cases against live data. "
            "Exits 1 if any routing, sub-routing or citation does not match."
        ),
    )
    parser.add_argument(
        "--db",
        default=None,
        metavar="PATH",
        help=(
            "SQLite file to ingest into. Defaults to a fresh temporary file, "
            "so the eval never touches data/app.db."
        ),
    )
    return parser.parse_args(argv)


def _temp_db_path() -> Path:
    """A fresh file in a new temp directory, left behind for inspection."""
    return Path(tempfile.mkdtemp(prefix="stock-moves-eval-")) / "eval.db"


def _ingest_all(keep_db: bool) -> tuple[list[IngestResult], dict[str, str]]:
    """Ingest each distinct ticker once. Returns the results and the failures by ticker.

    A ticker is ingested once even though it carries two cases; with `--db`
    pointing at a database that already holds it, the run is still idempotent,
    it just pays for the fetch again.
    """
    results: list[IngestResult] = []
    failed: dict[str, str] = {}
    tickers = list(dict.fromkeys(case.ticker for case in EVAL_CASES))
    for ticker in tickers:
        print(f"ingesting {ticker} ({PERIOD})...", flush=True)
        try:
            with session_scope() as session:
                result = ingest_ticker(session, ticker, period=PERIOD)
        except Exception as error:  # noqa: BLE001 - one bad ticker must not stop the run
            failed[ticker] = f"{ticker} ingest raised {type(error).__name__}: {error}"
            print(f"  FAILED  {type(error).__name__}: {error}", flush=True)
            continue
        results.append(result)
        print(
            f"  prices={result.n_prices} moves={result.n_moves} articles={result.n_articles}"
            f" explained={result.n_explanations} edges={result.n_edges}"
            f" geo_events={result.n_geo_events} via {result.provider}",
            flush=True,
        )
    if keep_db:
        print("  (--db given: the ingested database is kept)", flush=True)
    return results, failed


def _judge(case: EvalCase, provider: ModelProvider, news_source: NewsSource) -> list[str]:
    """Print one case in full and return its failure lines (empty when clean)."""
    print(f"{case.ticker} {case.on.isoformat()}  {case.what_happened}")
    with session_scope() as session:
        move = get_move(session, case.ticker, case.on)
        if move is None:
            print("  no move stored on that date: it did not clear the move thresholds.")
            return [f"{case.ticker} {case.on.isoformat()}: no move on that date"]
        explanation = _explanation_for(session, move, provider, news_source)
        cited = _cited_articles(session, explanation)
        return _report(case, move, explanation, cited)


def _explanation_for(
    session: Session,
    move: Move,
    provider: ModelProvider,
    news_source: NewsSource,
) -> Explanation | None:
    """The stored explanation, written on demand if the ingest did not write it.

    This is `api.tickers.read_move`'s own fallback, and deliberately the same
    call with the same arguments: the ingest explains only the top-N moves by
    `abs(ret_z)`, and two of the three eval days sit outside it, so without this
    the citation half of the pass rule could never be judged at all. A failure
    here is reported and swallowed — the routing verdicts do not depend on it,
    and the citation verdict then fails on its own terms.
    """
    if move.id is None:
        return None
    explanation = get_explanation(session, move.id)
    if explanation is not None:
        return explanation
    company = get_company(session, move.ticker)
    if company is None:
        print("  no company row: the ingest never stored this ticker.")
        return None
    print("  explaining on demand (the move is outside the ingest's top-N)...", flush=True)
    try:
        return enrich_move(session, move, company, provider, news_source)
    except Exception as error:  # noqa: BLE001 - one bad explanation must not stop the run
        print(f"  on-demand explanation raised {type(error).__name__}: {error}", flush=True)
        return None


def _report(
    case: EvalCase,
    move: Move,
    explanation: Explanation | None,
    cited: list[tuple[str, str]],
) -> list[str]:
    """Print the numbers, the prose and the three verdicts; return the failures."""
    print(f"  ret          {_pct(move.ret)}   ret_z {_num(move.ret_z)}")
    print(f"  macro_driver {move.macro_driver} ({_num(move.macro_driver_component)})")
    print(f"  rival_comove {_num(move.rival_comove)}   chain_comove {_num(move.chain_comove)}")
    if explanation is None:
        print("  explanation  (none: not stored by the ingest and not written on demand)")
    else:
        print(f"  explanation  [{explanation.provider}] {explanation.summary}")
    if cited:
        # The first three only: the lead citation is the one worth reading,
        # and the order is the order the explanation cited them in.
        for _, display in cited[:CITED_SHOWN]:
            print(f"    cited: {display}")
        if len(cited) > CITED_SHOWN:
            print(f"    cited: (+{len(cited) - CITED_SHOWN} more)")
    else:
        print("    cited: (no articles cited)")

    failures: list[str] = []
    routing_ok = move.routing in case.expected_routing
    sub_ok = move.sub_routing in case.expected_sub_routing
    matched = _matched_keyword(case.citation_keywords, cited)
    print(
        f"  routing      {move.routing}"
        f"  expected {_accepted(case.expected_routing)}  {_verdict(routing_ok)}"
    )
    print(
        f"  sub_routing  {move.sub_routing}"
        f"  expected {_accepted(case.expected_sub_routing)}  {_verdict(sub_ok)}"
    )
    print(
        f"  citation     {matched if matched else 'none'}"
        f"  expected {_accepted(case.citation_keywords)}  {_verdict(matched is not None)}"
    )
    where = f"{case.ticker} {case.on.isoformat()}"
    if not routing_ok:
        failures.append(
            f"{where}: routing {move.routing}, expected {_accepted(case.expected_routing)}"
        )
    if not sub_ok:
        failures.append(
            f"{where}: sub_routing {move.sub_routing},"
            f" expected {_accepted(case.expected_sub_routing)}"
        )
    if matched is None:
        failures.append(
            f"{where}: no cited title names the event,"
            f" expected one of {_accepted(case.citation_keywords)}"
        )
    return failures


def _cited_articles(session: Session, explanation: Explanation | None) -> list[tuple[str, str]]:
    """The articles the explanation cited, in the order it cited them.

    Each entry is `(title, display)`: the bare title is what the keywords are
    matched against, and the display line adds the source for the reader. A
    citation whose row has gone has no title to match, only a line to print.
    """
    if explanation is None:
        return []
    cited: list[tuple[str, str]] = []
    for article_id in explanation.cited_article_ids:
        article = session.get(Article, article_id)
        if article is None:
            cited.append(("", f"[{article_id}] (article row missing)"))
            continue
        source = f" ({article.source})" if article.source else ""
        cited.append((article.title, f"{article.title}{source}"))
    return cited


def _matched_keyword(keywords: tuple[str, ...], cited: list[tuple[str, str]]) -> str | None:
    """The first keyword named by any cited title, or None if none of them is."""
    for title, _ in cited:
        lowered = title.lower()
        for keyword in keywords:
            if keyword.lower() in lowered:
                return keyword
    return None


def _provider_used(results: list[IngestResult], configured: str) -> str:
    """The provider the stored explanations actually came from, not the configured one."""
    observed = sorted({result.provider for result in results})
    if len(observed) == 1:
        return observed[0]
    if observed:
        return " and ".join(observed)
    return configured


def _accepted(values: tuple[str | None, ...]) -> str:
    return "|".join("None" if value is None else value for value in values)


def _verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _num(value: float | None) -> str:
    return "None" if value is None else f"{value:.3f}"


def _pct(value: float | None) -> str:
    return "None" if value is None else f"{value * 100:+.2f}%"


if __name__ == "__main__":
    raise SystemExit(main())
