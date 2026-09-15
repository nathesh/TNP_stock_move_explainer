"""Seeding the runtime database from the committed snapshot."""

from __future__ import annotations

import gzip
from pathlib import Path

from stock_moves.seed import ensure_seeded


def _snapshot(tmp_path: Path, payload: bytes = b"a database") -> Path:
    """A gzipped stand-in for `data/snapshot.db.gz`."""
    source = tmp_path / "snapshot.db.gz"
    with gzip.open(source, "wb") as handle:
        handle.write(payload)
    return source


def test_expands_the_snapshot_when_no_database_exists(tmp_path: Path) -> None:
    target = tmp_path / "runtime" / "app.db"

    assert ensure_seeded(db_path=target, snapshot=_snapshot(tmp_path)) is True
    assert target.read_bytes() == b"a database"


def test_leaves_an_existing_database_alone(tmp_path: Path) -> None:
    target = tmp_path / "app.db"
    target.write_bytes(b"the real one")

    assert ensure_seeded(db_path=target, snapshot=_snapshot(tmp_path)) is False
    assert target.read_bytes() == b"the real one"


def test_replaces_a_zero_byte_database(tmp_path: Path) -> None:
    """SQLite leaves an empty file behind when a previous start failed."""
    target = tmp_path / "app.db"
    target.touch()

    assert ensure_seeded(db_path=target, snapshot=_snapshot(tmp_path)) is True
    assert target.read_bytes() == b"a database"


def test_missing_snapshot_is_not_an_error(tmp_path: Path) -> None:
    """A deployment without a snapshot still starts, on an empty database."""
    target = tmp_path / "app.db"

    assert ensure_seeded(db_path=target, snapshot=tmp_path / "absent.db.gz") is False
    assert not target.exists()


def test_a_corrupt_snapshot_does_not_break_startup(tmp_path: Path) -> None:
    """Decompression failing must not take the cold start down with it."""
    source = tmp_path / "snapshot.db.gz"
    source.write_bytes(b"not gzip at all")
    target = tmp_path / "app.db"

    assert ensure_seeded(db_path=target, snapshot=source) is False
    # The staging file is cleaned up rather than left in the directory.
    assert list(tmp_path.glob("*.seed")) == []


def test_unreadable_destination_is_not_an_error(tmp_path: Path) -> None:
    """The read-only filesystem case: log it and serve an empty database."""
    read_only = tmp_path / "read-only"
    read_only.mkdir(mode=0o500)
    try:
        assert ensure_seeded(db_path=read_only / "app.db", snapshot=_snapshot(tmp_path)) is False
    finally:
        read_only.chmod(0o700)
