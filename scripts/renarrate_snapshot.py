"""Re-narrate the committed snapshot's keyless explanations: `data/snapshot.db.gz`.

The snapshot's stored `explanations` rows were written before `narrate.py`
existed, so they still speak in quant: "a 5.2-sigma day", "idiosyncratic
component dominated (market -0.9pp ...)". `GET /tickers/{ticker}` serves those
stored summaries verbatim, which is the one place the deployed app contradicts
the README's "written for a reader, not for a quant".

`build_snapshot.py` cannot fix that, and deliberately so: it skips any ticker
that already has prices, because being resumable after a rate limit matters
more than re-explaining. This script is the other half — it changes no prices,
no articles and no scores, it only re-renders the prose:

    uv run python scripts/renarrate_snapshot.py              # data/snapshot.db
    uv run python scripts/renarrate_snapshot.py --db copy.db # somewhere else

Every row whose `provider` is the keyless one is regenerated through
`explain.get_or_create_explanation(..., refresh=True)`, which re-reads the
*already stored* `move_articles` scores and hands them to the provider. Nothing
is fetched and nothing is re-scored, so the run is offline and deterministic.

`HeuristicProvider` is constructed here explicitly rather than taken from
`get_settings()`: this repo's `.env` holds real keys, and the committed
snapshot has to stay keyless, reproducible and free to rebuild. Model-written
rows (any other `provider`) are left exactly as they are — re-rendering one
from the template would silently downgrade it and lose its citations.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_snapshot import WORKING_DB, compress
from sqlmodel import col, select

from stock_moves.db import configure_engine, init_db, session_scope
from stock_moves.explain import (
    DEFAULT_TOP_K,
    FALLBACK_PROVIDER_NAME,
    get_or_create_explanation,
)
from stock_moves.models import Company, Explanation, Move
from stock_moves.providers.heuristic import HeuristicProvider

PROGRESS_EVERY = 50


def renarrate(
    db_path: Path,
    *,
    top_k: int = DEFAULT_TOP_K,
    progress_every: int = PROGRESS_EVERY,
) -> int:
    """Rewrite every keyless explanation in `db_path`. Returns the rows rewritten.

    The provider is `HeuristicProvider()`, built here and not looked up from
    the settings, so no key in the environment can change what gets written.
    """
    configure_engine(f"sqlite:///{db_path}")
    init_db()

    provider = HeuristicProvider()
    started = time.time()
    rewritten = 0
    skipped: list[int] = []

    with session_scope() as session:
        rows = session.exec(
            select(Explanation)
            .where(col(Explanation.provider) == FALLBACK_PROVIDER_NAME)
            .order_by(col(Explanation.id).asc())
        ).all()
        total = len(rows)
        print(f"{total} keyless explanation(s) to re-narrate in {db_path}", flush=True)

        for index, row in enumerate(rows, start=1):
            move = session.get(Move, row.move_id)
            company = session.get(Company, move.ticker) if move is not None else None
            if move is None or company is None:
                # An explanation with no move or no company is unrenderable;
                # leave the stored text alone rather than blanking it.
                skipped.append(int(row.id) if row.id is not None else 0)
                continue
            get_or_create_explanation(session, move, company, provider, top_k=top_k, refresh=True)
            rewritten += 1
            if progress_every > 0 and (index % progress_every == 0 or index == total):
                print(
                    f"[{index:4d}/{total}] {move.ticker:<6} {move.date}  rewritten={rewritten}",
                    flush=True,
                )

    print(
        f"re-narrated {rewritten} row(s) in {time.time() - started:.1f}s"
        f"{f'; skipped {len(skipped)} orphaned row(s)' if skipped else ''}",
        flush=True,
    )
    return rewritten


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, re-narrate, and gzip the default snapshot back into place."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--db",
        type=Path,
        default=WORKING_DB,
        help=f"SQLite file to re-narrate (default: {WORKING_DB})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"headlines handed to the provider per move (default: {DEFAULT_TOP_K})",
    )
    arguments = parser.parse_args(argv)

    db_path = Path(arguments.db).resolve()
    if not db_path.exists():
        parser.error(f"no such database: {db_path}")
    rewritten = renarrate(db_path, top_k=arguments.top_k)

    if db_path == WORKING_DB.resolve():
        compress()
    else:
        print(f"not the committed working copy ({WORKING_DB}); skipping gzip", flush=True)
    return rewritten


if __name__ == "__main__":
    main()
