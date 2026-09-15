"""SQLite engine and session helpers.

A single module-level engine is cached so every caller shares one connection
pool. Tests call `configure_engine("sqlite://")` for an in-memory database;
`StaticPool` keeps that one connection alive across sessions, otherwise each
new session would get a fresh, empty database.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from stock_moves.settings import get_settings

__all__ = [
    "Session",
    "configure_engine",
    "get_engine",
    "get_session",
    "init_db",
    "session_scope",
]

# URLs that mean "in-memory", i.e. there is no file to create a directory for.
_MEMORY_URLS = frozenset({"sqlite://", "sqlite:///", "sqlite:///:memory:", "sqlite://:memory:"})

_engine: Engine | None = None


def _build_engine(url: str) -> Engine:
    connect_args = {"check_same_thread": False}
    if url in _MEMORY_URLS:
        # One shared connection, or each Session sees its own empty database.
        return create_engine(url, connect_args=connect_args, poolclass=StaticPool)
    path = Path(url.removeprefix("sqlite:///"))
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, connect_args=connect_args)


def configure_engine(url: str) -> Engine:
    """Replace the cached engine with one for `url` and return it."""
    global _engine
    old = _engine
    _engine = _build_engine(url)
    if old is not None:
        old.dispose()
    return _engine


def get_engine() -> Engine:
    """Return the cached engine, building one from the settings `db_path` if needed."""
    global _engine
    if _engine is None:
        _engine = _build_engine(f"sqlite:///{get_settings().db_path}")
    return _engine


def init_db(engine: Engine | None = None) -> None:
    """Create every table that does not exist yet."""
    SQLModel.metadata.create_all(engine if engine is not None else get_engine())


def get_session() -> Iterator[Session]:
    """Yield a session; usable directly as a FastAPI dependency."""
    with Session(get_engine()) as session:
        yield session


@contextmanager
def session_scope() -> Iterator[Session]:
    """Yield a session, committing on success and rolling back on any exception."""
    with Session(get_engine()) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
