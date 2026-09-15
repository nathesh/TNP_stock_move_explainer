"""FastAPI dependencies: the database session, the model provider, the settings.

Three one-line indirections, and each buys something. `get_db` exists so a
test can override the session without touching the engine. `get_provider_dep`
is cached for the life of the process because building the Anthropic provider
constructs an SDK client, and a request must not pay for that; the cache is
exposed through `reset_provider_cache` so a test that changes
`ANTHROPIC_API_KEY` can force the next request to re-decide (DESIGN section 4:
the key decides which provider answers). `get_settings_dep` wraps the cached
`get_settings` so routes declare configuration as a dependency rather than
importing it.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator

from stock_moves.db import Session, get_session
from stock_moves.providers import ModelProvider, get_provider
from stock_moves.settings import Settings, get_settings

__all__ = [
    "get_db",
    "get_provider_dep",
    "get_settings_dep",
    "reset_provider_cache",
]


def get_db() -> Iterator[Session]:
    """Yield a session bound to the configured engine, closed after the request."""
    yield from get_session()


@functools.lru_cache(maxsize=1)
def get_provider_dep() -> ModelProvider:
    """The process-wide model provider: Anthropic with a key, heuristic without."""
    return get_provider()


def reset_provider_cache() -> None:
    """Drop the cached provider so the next request re-reads the settings."""
    get_provider_dep.cache_clear()


def get_settings_dep() -> Settings:
    """The cached settings."""
    return get_settings()
