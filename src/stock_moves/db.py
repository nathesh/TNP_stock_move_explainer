"""SQLite engine, session helpers and the one-way schema migration.

A single module-level engine is cached so every caller shares one connection
pool. Tests call `configure_engine("sqlite://")` for an in-memory database;
`StaticPool` keeps that one connection alive across sessions, otherwise each
new session would get a fresh, empty database.

**Schema.** `SQLModel.metadata.create_all` creates missing *tables* but never
adds a column to a table that already exists, so a database written before
v1.5 kept the v1 columns and the first v1.5 read of it died on
`no such column: prices.macro_driver`. `migrate_schema` closes exactly that
gap: for every table that is already there, it compares the model's columns
with `PRAGMA table_info` and adds what is missing with `ALTER TABLE ... ADD
COLUMN`. `init_db` runs it straight after `create_all`, so every entry point —
the app lifespan, the scripts, a seeded cold start — upgrades the file it
opens instead of failing on it.

It is deliberately the smallest migration that is safe on SQLite: additive
only. It never drops, renames or retypes a column, and a missing `NOT NULL`
column with no default raises `SchemaMigrationError` rather than being skipped
in silence, because SQLite cannot add one to a table that already has rows.
Anything beyond that is a real migration tool's job.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Column, Connection, Engine, inspect
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from stock_moves.settings import get_settings

__all__ = [
    "SchemaMigrationError",
    "Session",
    "configure_engine",
    "get_engine",
    "get_session",
    "init_db",
    "migrate_schema",
    "session_scope",
]

logger = logging.getLogger(__name__)

# URLs that mean "in-memory", i.e. there is no file to create a directory for.
_MEMORY_URLS = frozenset({"sqlite://", "sqlite:///", "sqlite:///:memory:", "sqlite://:memory:"})

_engine: Engine | None = None


class SchemaMigrationError(RuntimeError):
    """A column is missing that `ALTER TABLE ... ADD COLUMN` cannot add safely."""


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


def _stored_columns(connection: Connection, table: str) -> set[str]:
    """Column names SQLite reports for `table`, straight from `PRAGMA table_info`."""
    rows = connection.exec_driver_sql(f'PRAGMA table_info("{table}")').all()
    return {str(row[1]) for row in rows}


def _default_literal(column: Column[object]) -> str | None:
    """A SQL literal for `column`'s default, or None when there is no usable one.

    Only a fixed scalar is usable: a server default is already SQL, a Python
    scalar default can be rendered, but a `default_factory`-style callable is
    evaluated per row and has no single literal to back-fill the existing rows
    with.
    """
    server_default = column.server_default
    text = getattr(server_default, "arg", None)
    if text is not None:
        return str(getattr(text, "text", text))

    default = column.default
    if default is None or getattr(default, "is_callable", False):
        return None
    value = getattr(default, "arg", None)
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    return None


def _add_column_sql(table: str, column: Column[object], dialect: Dialect) -> str:
    """The `ALTER TABLE ... ADD COLUMN` statement for one missing column.

    The type is the model's own SQLAlchemy type compiled for this dialect, so
    the added column matches what `create_all` would have made. A `NOT NULL`
    column is only addable with a default to back-fill the existing rows;
    without one there is nothing safe to do, so say so loudly.
    """
    sql_type = column.type.compile(dialect=dialect)
    statement = f'ALTER TABLE "{table}" ADD COLUMN "{column.name}" {sql_type}'
    if column.nullable:
        return statement

    literal = _default_literal(column)
    if literal is None:
        raise SchemaMigrationError(
            f"cannot add {table}.{column.name}: it is NOT NULL with no default, "
            "and SQLite cannot add such a column to a table that already has rows. "
            "Rebuild the database (delete it and re-ingest, or reseed from a "
            "snapshot built by this version)."
        )
    return f"{statement} NOT NULL DEFAULT {literal}"


def migrate_schema(engine: Engine | None = None) -> list[str]:
    """Add every model column missing from a table that already exists.

    Returns the `"table.column"` names added, in the order they were added —
    empty for a database that is already current, which is the normal case and
    costs one `PRAGMA table_info` per table. Tables that do not exist at all
    are left to `create_all`; nothing is ever dropped or altered in place.
    """
    target = engine if engine is not None else get_engine()
    present_tables = set(inspect(target).get_table_names())
    added: list[str] = []

    with target.begin() as connection:
        for table in SQLModel.metadata.sorted_tables:
            if table.name not in present_tables:
                continue
            stored = _stored_columns(connection, table.name)
            for column in table.columns:
                if column.name in stored:
                    continue
                connection.exec_driver_sql(_add_column_sql(table.name, column, target.dialect))
                logger.info("schema: added column %s.%s", table.name, column.name)
                added.append(f"{table.name}.{column.name}")
    return added


def init_db(engine: Engine | None = None) -> None:
    """Create every table that does not exist yet, then add any missing columns."""
    target = engine if engine is not None else get_engine()
    SQLModel.metadata.create_all(target)
    migrate_schema(target)


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
