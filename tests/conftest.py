"""Fixtures shared by every test module."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from stock_moves import models  # noqa: F401  (registers tables on SQLModel.metadata)
from stock_moves.db import Session, configure_engine, init_db
from stock_moves.settings import get_settings


@pytest.fixture
def session() -> Iterator[Session]:
    """A session on a fresh in-memory database with every table created."""
    engine = configure_engine("sqlite://")
    init_db(engine)
    with Session(engine) as db_session:
        yield db_session
    engine.dispose()


@pytest.fixture
def no_api_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run with no `ANTHROPIC_API_KEY` in the environment, cache cleared both ways."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
