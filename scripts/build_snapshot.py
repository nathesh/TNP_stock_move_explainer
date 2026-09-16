"""Build the committed demo snapshot: `data/snapshot.db.gz`.

Ingests a broad, liquid slice of the US large-cap market — every sector the
routing logic can choose between — and gzips the resulting SQLite file so a
deployed instance can seed itself in one decompression (`stock_moves.seed`).

Run it from the repository root:

    uv run python scripts/build_snapshot.py                 # all tickers
    uv run python scripts/build_snapshot.py AAPL NVDA       # just these

It is **resumable and idempotent**. A ticker that already has prices in the
working database is skipped, so an interrupted run (a Yahoo rate limit, a
dropped connection) is restarted by running the same command again. The
working file is `data/snapshot.db`, which `.gitignore` already excludes; only
the compressed copy is committed.

Explanations follow whatever provider the environment gives: with a valid
`ANTHROPIC_API_KEY` they are model-written and cited, without one they are
templated from the decomposition. The snapshot records which, per row, so a
rebuild with a key is visible in the data rather than a matter of trust.

Because a ticker with prices is skipped, this script will not re-explain what
is already there. When only the *phrasing* has changed, run
`scripts/renarrate_snapshot.py` instead: it rewrites the keyless explanations
in place from the stored scores and re-gzips, without refetching anything.
"""

from __future__ import annotations

import gzip
import logging
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stock_moves.db import configure_engine, init_db, session_scope
from stock_moves.ingest import ingest_ticker
from stock_moves.queries import has_prices

WORKING_DB = REPO_ROOT / "data" / "snapshot.db"
SNAPSHOT_GZ = REPO_ROOT / "data" / "snapshot.db.gz"

PERIOD = "1y"
TOP_N = 5

#: ~100 of the most-traded US names, spread across every sector in the ETF
#: map, so the demo shows company-, industry- and macro-routed days rather
#: than a hundred variations on one mega-cap tech story.
TICKERS: tuple[str, ...] = (
    # Technology & semis
    "AAPL",
    "MSFT",
    "NVDA",
    "AVGO",
    "AMD",
    "INTC",
    "MU",
    "QCOM",
    "TXN",
    "ADBE",
    "CRM",
    "ORCL",
    "CSCO",
    "IBM",
    "NOW",
    "INTU",
    "AMAT",
    "LRCX",
    "KLAC",
    "PANW",
    "SNOW",
    "PLTR",
    "DELL",
    "SMCI",
    "ARM",
    # Communication services & internet
    "GOOGL",
    "META",
    "NFLX",
    "DIS",
    "CMCSA",
    "T",
    "VZ",
    "TMUS",
    "SPOT",
    "UBER",
    # Consumer discretionary
    "AMZN",
    "TSLA",
    "HD",
    "MCD",
    "NKE",
    "SBUX",
    "LOW",
    "TJX",
    "BKNG",
    "GM",
    "F",
    "RIVN",
    "ABNB",
    "CMG",
    "LULU",
    # Consumer staples
    "WMT",
    "COST",
    "PG",
    "KO",
    "PEP",
    "PM",
    "MO",
    "MDLZ",
    "CL",
    "TGT",
    # Financials
    "BRK-B",
    "JPM",
    "BAC",
    "WFC",
    "GS",
    "MS",
    "C",
    "SCHW",
    "BLK",
    "AXP",
    "V",
    "MA",
    "PYPL",
    "COIN",
    "SPGI",
    # Health care
    "UNH",
    "JNJ",
    "LLY",
    "PFE",
    "MRK",
    "ABBV",
    "TMO",
    "ABT",
    "AMGN",
    "GILD",
    "CVS",
    "ISRG",
    "VRTX",
    "MRNA",
    "BMY",
    # Industrials & transport
    "CAT",
    "BA",
    "GE",
    "HON",
    "UNP",
    "UPS",
    "LMT",
    "RTX",
    "DE",
    "MMM",
    # Energy, materials, utilities, real estate
    "XOM",
    "CVX",
    "COP",
    "SLB",
    "OXY",
    "LIN",
    "FCX",
    "NEE",
    "DUK",
    "AMT",
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_snapshot")


def build(tickers: tuple[str, ...]) -> int:
    """Ingest every ticker into the working database. Returns the failure count."""
    WORKING_DB.parent.mkdir(parents=True, exist_ok=True)
    configure_engine(f"sqlite:///{WORKING_DB}")
    init_db()

    failures: list[tuple[str, str]] = []
    started = time.time()

    for index, ticker in enumerate(tickers, start=1):
        with session_scope() as session:
            if has_prices(session, ticker):
                print(
                    f"[{index:3d}/{len(tickers)}] {ticker:<6} skipped (already ingested)",
                    flush=True,
                )
                continue
        tick = time.time()
        try:
            with session_scope() as session:
                result = ingest_ticker(session, ticker, period=PERIOD, top_n=TOP_N)
        except Exception as error:  # noqa: BLE001 - one bad ticker must not stop the run
            failures.append((ticker, f"{type(error).__name__}: {error}"))
            print(
                f"[{index:3d}/{len(tickers)}] {ticker:<6} FAILED  {type(error).__name__}",
                flush=True,
            )
            continue
        print(
            f"[{index:3d}/{len(tickers)}] {ticker:<6} {time.time() - tick:5.1f}s"
            f"  moves={result.n_moves:<4} articles={result.n_articles:<5}"
            f"  explained={result.n_explanations}  via {result.provider}",
            flush=True,
        )

    print(f"\ningested in {(time.time() - started) / 60:.1f} min", flush=True)
    if failures:
        print(f"{len(failures)} failed:", flush=True)
        for ticker, message in failures:
            print(f"  {ticker}: {message}", flush=True)
    return len(failures)


def compress() -> None:
    """Gzip the working database into the committed snapshot."""
    with open(WORKING_DB, "rb") as plain, gzip.open(SNAPSHOT_GZ, "wb", compresslevel=9) as out:
        shutil.copyfileobj(plain, out)
    raw_mb = WORKING_DB.stat().st_size / 1_000_000
    gz_mb = SNAPSHOT_GZ.stat().st_size / 1_000_000
    print(f"{WORKING_DB.name}: {raw_mb:.1f} MB -> {SNAPSHOT_GZ.name}: {gz_mb:.1f} MB", flush=True)


if __name__ == "__main__":
    selected = tuple(argument.strip().upper() for argument in sys.argv[1:]) or TICKERS
    build(selected)
    compress()
