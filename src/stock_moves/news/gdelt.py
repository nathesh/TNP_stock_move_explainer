"""GDELT DOC 2.0 — the second news source (DESIGN §3).

Also keyless and historical, and it adds a language field. The cost: one
request per five seconds, enforced client-side here, so a one-year ingest is
minutes on GDELT versus seconds on RSS. Used as a fallback, or as the primary
when `NEWS_SOURCE=gdelt`.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime
from typing import Any

import httpx

from .base import NewsItem

__all__ = ["GDELTSource", "parse_gdelt"]

logger = logging.getLogger(__name__)

GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
NEWS_SOURCE = "gdelt"
SEEN_DATE_FORMAT = "%Y%m%dT%H%M%SZ"

# Shared by every GDELTSource instance: the throttle is a property of the
# remote API, not of one client object. `time.monotonic` and `time.sleep` are
# looked up on the module so a test can monkeypatch them.
_last_call_monotonic: float | None = None


def _throttle(throttle_s: float) -> None:
    """Block until at least `throttle_s` has passed since the previous call."""
    global _last_call_monotonic

    now = time.monotonic()
    if _last_call_monotonic is not None:
        wait = throttle_s - (now - _last_call_monotonic)
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
    _last_call_monotonic = now


def _parse_seen_date(raw: Any) -> datetime | None:
    """Parse GDELT's `YYYYMMDDTHHMMSSZ` into a tz-aware UTC datetime, or None."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return datetime.strptime(raw.strip(), SEEN_DATE_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        logger.warning("gdelt: unparseable seendate %r", raw)
        return None


def _text_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def parse_gdelt(payload: dict, bucket: str = "") -> list[NewsItem]:
    """Parse a GDELT artlist payload into items, deduped on url, order kept."""
    if not isinstance(payload, dict):
        return []
    articles = payload.get("articles")
    if not isinstance(articles, list):
        return []

    items: list[NewsItem] = []
    seen: set[str] = set()
    for article in articles:
        if not isinstance(article, dict):
            continue
        url = _text_or_none(article.get("url"))
        if url is None or url in seen:
            continue
        seen.add(url)
        items.append(
            NewsItem(
                url=url,
                title=(_text_or_none(article.get("title")) or ""),
                source=_text_or_none(article.get("domain")),
                published_at=_parse_seen_date(article.get("seendate")),
                language=_text_or_none(article.get("language")),
                news_source=NEWS_SOURCE,
                bucket=bucket,
            )
        )
    return items


class GDELTSource:
    """`NewsSource` over the GDELT DOC 2.0 API, behind a client-side throttle."""

    name = NEWS_SOURCE

    def __init__(
        self,
        timeout_s: float = 20.0,
        throttle_s: float = 5.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.timeout_s = timeout_s
        self.throttle_s = throttle_s
        self._client = client

    def _params(self, query: str, start: date, end: date, limit: int) -> dict[str, str]:
        return {
            "query": f"{query} sourcelang:english",
            "mode": "artlist",
            "format": "json",
            "sort": "hybridrel",
            "maxrecords": str(limit),
            "startdatetime": f"{start:%Y%m%d}000000",
            "enddatetime": f"{end:%Y%m%d}235959",
        }

    def search(self, query: str, start: date, end: date, limit: int = 30) -> list[NewsItem]:
        """Search the window. Any transport, status or decode failure returns []."""
        _throttle(self.throttle_s)

        client = self._client or httpx.Client(timeout=self.timeout_s)
        try:
            response = client.get(GDELT_ENDPOINT, params=self._params(query, start, end, limit))
            if response.status_code != 200:
                logger.warning("gdelt: HTTP %s for query %r", response.status_code, query)
                return []
            payload = response.json()
        except httpx.HTTPError as exc:
            logger.warning("gdelt: request failed for query %r: %s", query, exc)
            return []
        except ValueError as exc:
            # GDELT answers a malformed query with plain text, not JSON.
            logger.warning("gdelt: non-JSON response for query %r: %s", query, exc)
            return []
        finally:
            if self._client is None:
                client.close()

        return parse_gdelt(payload)[:limit]
