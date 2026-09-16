"""Tests for `scripts/renarrate_snapshot.py`.

The script exists to fix stored prose in place, so the two things worth
asserting are the two ways it could go wrong: a keyless row that keeps its
quant phrasing (the bug it was written for), and a model-written row that gets
quietly overwritten by the template (the damage it must not do). Everything
else -- prices, articles, scores -- it never touches.

No network, no key: the provider is `HeuristicProvider`, constructed by the
script itself rather than chosen from the settings.
"""

from __future__ import annotations

import importlib.util
import re
from datetime import date
from pathlib import Path

import pytest
from sqlmodel import Session, select

from stock_moves.db import configure_engine, init_db
from stock_moves.models import Company, Explanation, Move

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "renarrate_snapshot.py"

#: The phrasing that must not survive a run -- the same scan used to check the
#: committed snapshot.
JARGON = re.compile(r"sigma|idiosyncratic|\d\s?pp\b|z-score", re.IGNORECASE)

STALE_HEURISTIC = "a 5.2-sigma day; idiosyncratic component dominated (market -0.9pp)"
MODEL_TEXT = "model text"


def load_script():
    """Import `scripts/renarrate_snapshot.py`, which is a script and not a package."""
    spec = importlib.util.spec_from_file_location("renarrate_snapshot", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    """A one-company database with one stale keyless row and one model-written row."""
    db_path = tmp_path / "snapshot.db"
    engine = configure_engine(f"sqlite:///{db_path}")
    init_db(engine)
    with Session(engine) as session:
        session.add(
            Company(
                ticker="TEST",
                name="Testco Industries Inc",
                sector="Technology",
                industry="Semiconductors",
                sector_etf="XLK",
            )
        )
        keyless = Move(
            ticker="TEST",
            date=date(2026, 3, 4),
            ret=-0.061,
            ret_z=-2.6,
            mkt_component=-0.002,
            sector_component=-0.004,
            idio_component=-0.055,
            routing="company",
            direction="down",
        )
        keyed = Move(
            ticker="TEST",
            date=date(2026, 3, 5),
            ret=0.048,
            ret_z=2.1,
            mkt_component=0.010,
            sector_component=0.008,
            idio_component=0.030,
            routing="macro",
            direction="up",
        )
        session.add(keyless)
        session.add(keyed)
        session.commit()
        session.refresh(keyless)
        session.refresh(keyed)
        session.add(
            Explanation(
                move_id=int(keyless.id or 0),
                summary=STALE_HEURISTIC,
                primary_category="company",
                confidence=0.5,
                provider="heuristic",
            )
        )
        session.add(
            Explanation(
                move_id=int(keyed.id or 0),
                summary=MODEL_TEXT,
                primary_category="macro",
                confidence=0.77,
                cited_article_ids_json="[7]",
                provider="openai",
            )
        )
        session.commit()
    engine.dispose()
    return db_path


def rows(db_path: Path) -> dict[str, Explanation]:
    """Both explanations, keyed by the provider that is stored on them now."""
    engine = configure_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        stored = session.exec(select(Explanation)).all()
        for row in stored:
            session.expunge(row)
    return {row.provider: row for row in stored}


def test_rewrites_the_keyless_row_and_leaves_the_model_row_alone(snapshot: Path) -> None:
    module = load_script()
    before = rows(snapshot)

    rewritten = module.renarrate(snapshot)

    assert rewritten == 1
    after = rows(snapshot)
    assert set(after) == {"heuristic", "openai"}

    keyless = after["heuristic"]
    assert keyless.summary != STALE_HEURISTIC
    assert not JARGON.search(keyless.summary)
    assert "TEST" in keyless.summary or "Testco" in keyless.summary

    # The model-written row is untouched, field for field.
    keyed, keyed_before = after["openai"], before["openai"]
    assert keyed.summary == MODEL_TEXT
    assert keyed.primary_category == keyed_before.primary_category
    assert keyed.confidence == keyed_before.confidence
    assert keyed.cited_article_ids_json == keyed_before.cited_article_ids_json
    assert keyed.unexplained == keyed_before.unexplained
    assert keyed.created_at == keyed_before.created_at


def test_is_idempotent(snapshot: Path) -> None:
    """A second run rewrites the same row to the same text, and adds no rows."""
    module = load_script()
    module.renarrate(snapshot)
    first = rows(snapshot)["heuristic"].summary

    assert module.renarrate(snapshot) == 1
    after = rows(snapshot)
    assert len(after) == 2
    assert after["heuristic"].summary == first
