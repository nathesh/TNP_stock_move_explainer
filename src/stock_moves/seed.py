"""Seed the database from the committed snapshot (serverless deployments).

A serverless function gets a read-only filesystem with one writable directory,
`/tmp`, and that directory is per-instance and wiped between cold starts. A
deployment therefore starts every instance with an empty database, and the
first visitor would see nothing until they ran an ingest — a minute of prices,
news and model calls before the page has anything on it.

So the repository carries `data/snapshot.db.gz`, a pre-ingested database built
by `scripts/build_snapshot.py`, and `ensure_seeded()` expands it to the
configured `db_path` when no database is there yet. A cold start pays one
decompression instead of an ingest, and every instance shows the same data.

The file is written to a temporary name in the destination directory and then
renamed, because `os.replace` is atomic: two instances racing on the same
`/tmp` can never leave a half-written database behind for a reader.

Local runs are unaffected: `data/app.db` already exists after the first
ingest, so seeding is skipped.
"""

from __future__ import annotations

import gzip
import logging
import os
import shutil
import tempfile
from pathlib import Path

from stock_moves.settings import REPO_ROOT, get_settings

__all__ = ["SNAPSHOT_PATH", "ensure_seeded"]

logger = logging.getLogger(__name__)

#: The committed, pre-ingested database. Built by `scripts/build_snapshot.py`.
SNAPSHOT_PATH = REPO_ROOT / "data" / "snapshot.db.gz"


def ensure_seeded(db_path: Path | None = None, snapshot: Path | None = None) -> bool:
    """Expand the snapshot to `db_path` if no database is there yet.

    Returns True when a database was written, False when one already existed
    or no snapshot is available. Never raises: a deployment without a snapshot,
    or with an unreadable one, must still start and serve an empty database
    rather than fail its cold start.
    """
    target = Path(get_settings().db_path if db_path is None else db_path)
    source = SNAPSHOT_PATH if snapshot is None else snapshot

    if target.exists() and target.stat().st_size > 0:
        return False
    if not source.exists():
        logger.info("no snapshot at %s; starting with an empty database", source)
        return False

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Same directory as the target, so the rename below stays on one
        # filesystem and is therefore atomic.
        handle, staging = tempfile.mkstemp(dir=target.parent, suffix=".seed")
        os.close(handle)
        try:
            with gzip.open(source, "rb") as compressed, open(staging, "wb") as plain:
                shutil.copyfileobj(compressed, plain)
            os.replace(staging, target)
        except BaseException:
            Path(staging).unlink(missing_ok=True)
            raise
    except OSError as error:
        logger.warning("could not seed %s from %s: %s", target, source, error)
        return False

    logger.info("seeded %s from %s", target, source)
    return True
