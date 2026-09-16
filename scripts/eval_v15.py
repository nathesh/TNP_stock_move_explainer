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

Exit codes: 0 every case passed, 1 at least one failed (a missing move on the
date counts as a failure — the day is supposed to clear the move thresholds).

Without a working key the run still completes, but `suggest_relations` returns
competitors only, so there are no `country` edges, the geo gate never opens and
a `country:XX` sub-routing is unreachable. The last line of the output says so.
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
from stock_moves.ingest import IngestResult, ingest_ticker
from stock_moves.models import Article, Explanation, Move
from stock_moves.providers import get_provider
from stock_moves.queries import get_explanation, get_move
from stock_moves.settings import REPO_ROOT, get_settings

__all__ = ["EVAL_CASES", "EvalCase", "main"]

EXIT_OK = 0
EXIT_FAILED = 1

#: Two years, so the 200-day SMA behind `regime_*` has warmed up by the time
#: the 2025 dates arrive. The v1.5 plan makes this the default everywhere.
PERIOD = "2y"


@dataclass(frozen=True)
class EvalCase:
    """One ticker-day and the answers that count as correct.

    `expected_routing` and `expected_sub_routing` are tuples of *accepted*
    values, not single answers; `(None,)` means the correct answer is no
    sub-routing at all (an `industry` day has no sub-bucket).
    """

    ticker: str
    on: date
    what_happened: str
    expected_routing: tuple[str, ...]
    expected_sub_routing: tuple[str | None, ...]


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
    ),
    EvalCase(
        ticker="NVDA",
        on=date(2025, 1, 27),
        what_happened="DeepSeek day, chip names down together (AMD -6.4%, so rivals moved *with* it)",
        expected_routing=("industry",),
        expected_sub_routing=(None,),
    ),
    EvalCase(
        ticker="AAPL",
        on=date(2025, 4, 3),
        what_happened="tariff announcement, the day after 2025-04-02",
        expected_routing=("macro",),
        expected_sub_routing=("country:CN", "dollar"),
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

    configured = get_provider(settings).name
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
        case_failures = _judge(case)
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
            "  heuristic: no country edges are suggested without a working key, "
            "so the geo gate stays closed and no `country:XX` sub-routing can fire."
        )
    return EXIT_FAILED if failures else EXIT_OK


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="eval_v15.py",
        description=(
            "Run the three v1.5 eval cases against live data. "
            "Exits 1 if any routing or sub-routing does not match."
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


def _judge(case: EvalCase) -> list[str]:
    """Print one case in full and return its failure lines (empty when clean)."""
    print(f"{case.ticker} {case.on.isoformat()}  {case.what_happened}")
    with session_scope() as session:
        move = get_move(session, case.ticker, case.on)
        if move is None:
            print("  no move stored on that date: it did not clear the move thresholds.")
            return [f"{case.ticker} {case.on.isoformat()}: no move on that date"]
        explanation = get_explanation(session, move.id) if move.id is not None else None
        cited = _cited_titles(session, explanation)
        return _report(case, move, explanation, cited)


def _report(
    case: EvalCase,
    move: Move,
    explanation: Explanation | None,
    cited: list[str],
) -> list[str]:
    """Print the numbers, the prose and the two verdicts; return the failures."""
    print(f"  ret          {_pct(move.ret)}   ret_z {_num(move.ret_z)}")
    print(f"  macro_driver {move.macro_driver} ({_num(move.macro_driver_component)})")
    print(f"  rival_comove {_num(move.rival_comove)}   chain_comove {_num(move.chain_comove)}")
    if explanation is None:
        print("  explanation  (none stored: the move is outside the explained top-N)")
    else:
        print(f"  explanation  [{explanation.provider}] {explanation.summary}")
    if cited:
        for title in cited:
            print(f"    cited: {title}")
    else:
        print("    cited: (no articles cited)")

    failures: list[str] = []
    routing_ok = move.routing in case.expected_routing
    sub_ok = move.sub_routing in case.expected_sub_routing
    print(
        f"  routing      {move.routing}"
        f"  expected {_accepted(case.expected_routing)}  {_verdict(routing_ok)}"
    )
    print(
        f"  sub_routing  {move.sub_routing}"
        f"  expected {_accepted(case.expected_sub_routing)}  {_verdict(sub_ok)}"
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
    return failures


def _cited_titles(session: Session, explanation: Explanation | None) -> list[str]:
    """The titles of the articles the explanation cited, in the order it cited them."""
    if explanation is None:
        return []
    titles: list[str] = []
    for article_id in explanation.cited_article_ids:
        article = session.get(Article, article_id)
        if article is None:
            titles.append(f"[{article_id}] (article row missing)")
            continue
        source = f" ({article.source})" if article.source else ""
        titles.append(f"{article.title}{source}")
    return titles


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
