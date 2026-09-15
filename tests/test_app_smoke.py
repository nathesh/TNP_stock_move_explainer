"""Smoke tests for the FastAPI scaffolding (DESIGN section 6).

The app is built against an in-memory engine, so the lifespan's `init_db()`
creates its tables there and nothing touches `data/app.db`. Nothing here hits
the network: no route is exercised beyond the static page and the schema.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from stock_moves import models  # noqa: F401  (registers tables on SQLModel.metadata)
from stock_moves.api.app import create_app
from stock_moves.db import configure_engine


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A client on an app whose engine is in-memory; `with` runs the lifespan."""
    configure_engine("sqlite://")
    with TestClient(create_app()) as test_client:
        yield test_client


def test_index_serves_the_chat_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    # The page must actually talk to the chat route, not just render.
    assert "/chat" in response.text


def test_docs_are_served(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200


def test_openapi_schema_is_valid(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "paths" in schema
    assert schema["info"]["title"] == "stock-move-explainer"


def test_static_dir_is_mounted(client: TestClient) -> None:
    response = client.get("/static/index.html")
    assert response.status_code == 200
    assert "/chat" in response.text
