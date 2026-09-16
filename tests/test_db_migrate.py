"""Schema migration — opening a pre-v1.5 database without dying on it.

`SQLModel.metadata.create_all` adds missing *tables* and never a missing
*column*, so a `data/app.db` written by v1 kept the v1 columns and the first
v1.5 read of it raised `no such column: prices.macro_driver`. `migrate_schema`
is the additive `ALTER TABLE ... ADD COLUMN` pass that closes that gap, and
`init_db` runs it after `create_all`.

The v1-shaped database here is built the honest way: create the current schema,
then take the v1.5 columns and tables back out again, so the fixture cannot
drift from the models the way a hand-written `CREATE TABLE` would.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, select

from stock_moves.db import SchemaMigrationError, configure_engine, init_db, migrate_schema
from stock_moves.models import Article, Move, MoveArticle, Price

#: Every column v1.5 added to a table that already existed in v1 — all of them
#: nullable with a default, which is what makes them addable in place.
V15_COLUMNS: dict[str, tuple[str, ...]] = {
    "prices": ("macro_driver", "macro_driver_component"),
    "moves": (
        "macro_driver",
        "macro_driver_component",
        "sub_routing",
        "rival_comove",
        "chain_comove",
    ),
    "move_articles": ("geo_gate",),
}

#: The two tables v1.5 added outright; `create_all` is responsible for these.
V15_TABLES: tuple[str, ...] = ("company_edges", "geo_events")

TICKER = "OLD"
OLD_DAY = date(2025, 6, 2)


def columns_of(engine: Engine, table: str) -> set[str]:
    """What SQLite itself reports for a table, via `PRAGMA table_info`."""
    with engine.begin() as connection:
        rows = connection.exec_driver_sql(f'PRAGMA table_info("{table}")').all()
    return {str(row[1]) for row in rows}


def tables_of(engine: Engine) -> set[str]:
    with engine.begin() as connection:
        rows = connection.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).all()
    return {str(row[0]) for row in rows}


def drop_columns(engine: Engine, columns: dict[str, tuple[str, ...]]) -> None:
    with engine.begin() as connection:
        for table, names in columns.items():
            for name in names:
                connection.exec_driver_sql(f'ALTER TABLE "{table}" DROP COLUMN "{name}"')


@pytest.fixture
def v1_engine(tmp_path: Path) -> Iterator[Engine]:
    """A database with the v1 schema: the v1.5 columns and tables taken back out."""
    engine = configure_engine(f"sqlite:///{tmp_path / 'v1.db'}")
    SQLModel.metadata.create_all(engine)
    drop_columns(engine, V15_COLUMNS)
    with engine.begin() as connection:
        for table in V15_TABLES:
            connection.exec_driver_sql(f'DROP TABLE "{table}"')
    yield engine
    engine.dispose()


@pytest.fixture
def fresh_engine(tmp_path: Path) -> Iterator[Engine]:
    """An empty file, i.e. the database a first run creates."""
    engine = configure_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    yield engine
    engine.dispose()


def store_v1_row(engine: Engine) -> None:
    """One price row written through the v1 columns, before the migration."""
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "INSERT INTO prices "
            "(ticker, date, open, high, low, close, volume, ret, "
            "near_earnings, near_fomc, near_cpi) "
            f"VALUES ('{TICKER}', '{OLD_DAY.isoformat()}', "
            "10.0, 11.0, 9.0, 10.5, 1000.0, 0.01, 0, 0, 0)"
        )


def test_v1_fixture_really_is_missing_the_columns(v1_engine: Engine) -> None:
    """Guard on the fixture: if this fails, the rest proves nothing."""
    for table, names in V15_COLUMNS.items():
        assert columns_of(v1_engine, table).isdisjoint(names)
    assert tables_of(v1_engine).isdisjoint(V15_TABLES)


def test_init_db_adds_the_v15_columns_and_tables(v1_engine: Engine) -> None:
    store_v1_row(v1_engine)

    init_db(v1_engine)

    for table, names in V15_COLUMNS.items():
        assert set(names) <= columns_of(v1_engine, table)
    assert set(V15_TABLES) <= tables_of(v1_engine)

    # The pre-existing row survives and reads back, with the new column null.
    with Session(v1_engine) as session:
        stored = session.exec(select(Price)).one()
        assert stored.ticker == TICKER
        assert stored.macro_driver is None


def test_the_new_columns_are_writable_after_the_migration(v1_engine: Engine) -> None:
    """The failing path in production: an ingest writing the v1.5 fields."""
    init_db(v1_engine)

    with Session(v1_engine) as session:
        session.add(
            Price(
                ticker=TICKER,
                date=OLD_DAY,
                open=10.0,
                high=11.0,
                low=9.0,
                close=10.5,
                volume=1000.0,
                macro_driver="country:TW",
                macro_driver_component=-0.012,
            )
        )
        move = Move(
            ticker=TICKER,
            date=OLD_DAY,
            ret=-0.08,
            ret_z=-4.0,
            routing="macro",
            direction="down",
            sub_routing="country:TW",
            macro_driver="country:TW",
            macro_driver_component=-0.012,
            rival_comove=-0.01,
            chain_comove=0.02,
        )
        article = Article(
            url="https://news.example/tw",
            title="Taiwan export controls",
            news_source="google_rss",
        )
        session.add(move)
        session.add(article)
        session.commit()
        session.refresh(move)
        session.refresh(article)
        assert move.id is not None
        assert article.id is not None

        session.add(
            MoveArticle(
                move_id=move.id,
                article_id=article.id,
                relevance=0.3,
                category="macro",
                provider="heuristic",
                geo_gate=0.0,
            )
        )
        session.commit()

    with Session(v1_engine) as session:
        assert session.exec(select(Price)).one().macro_driver == "country:TW"
        assert session.exec(select(Move)).one().sub_routing == "country:TW"
        assert session.exec(select(MoveArticle)).one().geo_gate == 0.0


def test_second_init_db_is_a_no_op(v1_engine: Engine) -> None:
    init_db(v1_engine)
    assert migrate_schema(v1_engine) == []

    before = {table: columns_of(v1_engine, table) for table in V15_COLUMNS}
    init_db(v1_engine)
    assert {table: columns_of(v1_engine, table) for table in V15_COLUMNS} == before


def test_migrate_schema_reports_what_it_added(v1_engine: Engine) -> None:
    """`create_all` first, so the two new tables are made and not migrated."""
    SQLModel.metadata.create_all(v1_engine)

    added = migrate_schema(v1_engine)

    expected = {f"{table}.{name}" for table, names in V15_COLUMNS.items() for name in names}
    assert set(added) == expected


def test_a_fresh_database_is_unaffected(fresh_engine: Engine) -> None:
    init_db(fresh_engine)

    assert migrate_schema(fresh_engine) == []
    for table, names in V15_COLUMNS.items():
        assert set(names) <= columns_of(fresh_engine, table)
    assert set(V15_TABLES) <= tables_of(fresh_engine)


def test_a_not_null_column_with_a_default_is_back_filled(v1_engine: Engine) -> None:
    """`moves.near_earnings` is NOT NULL with a default, so it can be added."""
    drop_columns(v1_engine, {"moves": ("near_earnings",)})
    with v1_engine.begin() as connection:
        connection.exec_driver_sql(
            "INSERT INTO moves "
            "(ticker, date, ret, ret_z, routing, direction, "
            "near_fomc, near_cpi, created_at) "
            f"VALUES ('{TICKER}', '{OLD_DAY.isoformat()}', -0.08, -4.0, 'macro', 'down', "
            "0, 0, '2025-06-02 20:00:00')"
        )

    assert "moves.near_earnings" in migrate_schema(v1_engine)

    with Session(v1_engine) as session:
        assert session.exec(select(Move)).one().near_earnings is False


def test_a_not_null_column_without_a_default_is_refused(v1_engine: Engine) -> None:
    """`moves.routing` cannot be back-filled, so say so instead of half-migrating."""
    drop_columns(v1_engine, {"moves": ("routing",)})

    with pytest.raises(SchemaMigrationError, match="moves.routing"):
        migrate_schema(v1_engine)
