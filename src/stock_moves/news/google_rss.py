"""Google News RSS — the v1 primary news source (DESIGN §3).

Keyless, historical via the `after:` / `before:` search operators, mainstream
outlets, no throttle. The cost: title, source, url and date only, no body, and
the url is a Google redirect, so the dedupe key is that redirect url.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import urlencode

import httpx

from .base import NewsItem

__all__ = ["GoogleNewsRSS", "parse_rss"]

logger = logging.getLogger(__name__)

RSS_ENDPOINT = "https://news.google.com/rss/search"
NEWS_SOURCE = "google_rss"

# Google News serves an empty feed to obvious bots.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


def _parse_pub_date(raw: str | None) -> datetime | None:
    """Parse an RFC 2822 `pubDate` into a tz-aware UTC datetime, or None."""
    if not raw or not raw.strip():
        return None
    try:
        parsed = parsedate_to_datetime(raw.strip())
    except (TypeError, ValueError):
        logger.warning("google_rss: unparseable pubDate %r", raw)
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _strip_source_suffix(title: str, source: str | None) -> str:
    """Drop the " - <outlet>" suffix Google appends to every headline."""
    if not source:
        return title
    suffix = f" - {source}"
    if title.endswith(suffix):
        return title[: -len(suffix)].strip()
    return title


def parse_rss(xml_text: str, bucket: str = "") -> list[NewsItem]:
    """Parse a Google News RSS document into items, deduped on link, order kept."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("google_rss: could not parse feed: %s", exc)
        return []

    items: list[NewsItem] = []
    seen: set[str] = set()
    for element in root.iterfind("./channel/item"):
        link = (element.findtext("link") or "").strip()
        if not link or link in seen:
            continue
        seen.add(link)

        source_element = element.find("source")
        source: str | None = None
        if source_element is not None and source_element.text:
            source = source_element.text.strip() or None

        title = (element.findtext("title") or "").strip()
        items.append(
            NewsItem(
                url=link,
                title=_strip_source_suffix(title, source),
                source=source,
                published_at=_parse_pub_date(element.findtext("pubDate")),
                language=None,
                news_source=NEWS_SOURCE,
                bucket=bucket,
            )
        )
    return items


class GoogleNewsRSS:
    """`NewsSource` over Google News RSS."""

    name = NEWS_SOURCE

    def __init__(self, timeout_s: float = 20.0, client: httpx.Client | None = None) -> None:
        self.timeout_s = timeout_s
        self._client = client

    def _url(self, query: str, start: date, end: date) -> str:
        # Google's `before:` is exclusive, so the window's last day is end + 1.
        windowed = f"{query} after:{start:%Y-%m-%d} before:{end + timedelta(days=1):%Y-%m-%d}"
        params = {
            "q": windowed,
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en",
        }
        return f"{RSS_ENDPOINT}?{urlencode(params)}"

    def search(self, query: str, start: date, end: date, limit: int = 30) -> list[NewsItem]:
        """Search the window. Any transport, status or parse failure returns []."""
        url = self._url(query, start, end)
        client = self._client or httpx.Client(timeout=self.timeout_s)
        try:
            response = client.get(url, headers={"User-Agent": USER_AGENT})
            if response.status_code != 200:
                logger.warning("google_rss: HTTP %s for query %r", response.status_code, query)
                return []
            xml_text = response.text
        except httpx.HTTPError as exc:
            logger.warning("google_rss: request failed for query %r: %s", query, exc)
            return []
        finally:
            if self._client is None:
                client.close()

        return parse_rss(xml_text)[:limit]
