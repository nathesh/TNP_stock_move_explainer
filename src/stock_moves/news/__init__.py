"""News retrieval: one protocol, two keyless sources, and the query builders."""

from __future__ import annotations

from .base import (
    MACRO_QUERY_TERMS,
    NewsItem,
    NewsSource,
    build_query,
    clean_company_name,
    get_news_source,
    queries_for_move,
    window_for,
)
from .gdelt import GDELTSource, parse_gdelt
from .google_rss import GoogleNewsRSS, parse_rss

__all__ = [
    "MACRO_QUERY_TERMS",
    "GDELTSource",
    "GoogleNewsRSS",
    "NewsItem",
    "NewsSource",
    "build_query",
    "clean_company_name",
    "get_news_source",
    "parse_gdelt",
    "parse_rss",
    "queries_for_move",
    "window_for",
]
