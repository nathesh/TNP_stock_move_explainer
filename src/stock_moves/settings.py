"""Application settings.

One `Settings` model, read once from the environment (after loading a `.env`
file from the repo root if present) and cached. Every default lives here so
thresholds and limits are configuration, not code constants.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

__all__ = ["Settings", "get_settings"]

# src/stock_moves/settings.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

# Loaded once at import time, so a later `get_settings.cache_clear()` after a
# test monkeypatches the environment does not resurrect values from the file.
load_dotenv(REPO_ROOT / ".env")


class Settings(BaseModel):
    """Runtime configuration; every field is overridable by its upper-cased name."""

    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"
    news_source: str = "google_rss"
    db_path: Path = Path("data/app.db")
    default_period: str = "1y"
    default_top_n: int = 10
    default_z_threshold: float = 2.0
    default_pct_threshold: float = 0.02
    top_k_articles: int = 8
    news_limit: int = 30
    gdelt_throttle_s: float = 5.0
    http_timeout_s: float = 20.0


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings, built from `os.environ`.

    A field is overridden by the environment variable named after it in upper
    case (`anthropic_model` <- `ANTHROPIC_MODEL`). An unset *or empty* variable
    leaves the default in place. Call `get_settings.cache_clear()` after
    changing the environment.
    """
    overrides: dict[str, str] = {}
    for name in Settings.model_fields:
        raw = os.environ.get(name.upper())
        if raw is not None and raw.strip() != "":
            overrides[name] = raw
    return Settings(**overrides)
