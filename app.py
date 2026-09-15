"""Deployment entrypoint.

Vercel loads a FastAPI instance named `app` from one of a fixed set of
filenames at the repo root (`app.py`, `index.py`, `main.py`, `server.py`,
`asgi.py`, `wsgi.py`); the real application lives in a package, so this file is
the two-line bridge between the two. `src` is put on the path explicitly so the
import works whether or not the project itself was installed into the runtime's
environment.

Nothing else imports this module — `uv run stock-moves` still starts
`stock_moves.api.app:app` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from stock_moves.api.app import app

__all__ = ["app"]
