"""The FastAPI application factory (DESIGN section 6).

`create_app()` rather than a bare module-level app, so a test can build a
fresh application against an in-memory engine. The routers are imported as
*modules* and their `.router` read at include time: that keeps this file
independent of what the route modules export, and lets the ticker and chat
routes be written in parallel with this one.

Tables are created in the lifespan, not at import, because importing the app
must not touch the filesystem.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from stock_moves.api import chat, tickers
from stock_moves.db import init_db
from stock_moves.settings import Settings, get_settings

__all__ = ["STATIC_DIR", "app", "create_app"]

STATIC_DIR = Path(__file__).parent / "static"
_INDEX_HTML = STATIC_DIR / "index.html"

_DESCRIPTION = (
    "Explains major daily stock moves. Each move is first decomposed into "
    "market, sector and idiosyncratic components, which routes the news "
    "search; a model then writes a cited explanation, or says the move is "
    "unexplained. Runs with no API keys."
)


def _package_version() -> str:
    """The installed distribution version, or the pinned fallback when the
    package is being run from a source tree that was never installed."""
    try:
        return version("stock-move-explainer")
    except PackageNotFoundError:
        return "0.1.0"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. `settings` overrides the cached settings for
    this app instance and is exposed as `app.state.settings`."""
    resolved = get_settings() if settings is None else settings

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Create any missing tables before the first request is served."""
        init_db()
        yield

    app = FastAPI(
        title="stock-move-explainer",
        version=_package_version(),
        description=_DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.settings = resolved

    app.include_router(tickers.router)
    app.include_router(chat.router)

    @app.get("/", response_class=FileResponse, tags=["ui"])
    def index() -> FileResponse:
        """The chat page: one static HTML file, no framework."""
        return FileResponse(_INDEX_HTML, media_type="text/html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


app = create_app()
