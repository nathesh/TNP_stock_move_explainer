"""The model layer: one interface, two implementations, one selector.

`get_provider()` is the only thing the rest of the app calls. It returns the
Anthropic provider when a key is configured and the heuristic provider
otherwise, so the app runs with zero keys (DESIGN section 4).
"""

from __future__ import annotations

from typing import Any

from stock_moves.providers.base import (
    MACRO_TERMS,
    TOOL_SPECS,
    ArticleInput,
    ArticleScore,
    ChatReply,
    ChatTurn,
    ExplanationResult,
    ModelProvider,
    MoveContext,
    ToolCallRecord,
    ToolFn,
)
from stock_moves.providers.heuristic import HeuristicProvider

__all__ = [
    "MACRO_TERMS",
    "TOOL_SPECS",
    "ArticleInput",
    "ArticleScore",
    "ChatReply",
    "ChatTurn",
    "ExplanationResult",
    "HeuristicProvider",
    "ModelProvider",
    "MoveContext",
    "ToolCallRecord",
    "ToolFn",
    "get_provider",
]


def get_provider(settings: Any | None = None) -> ModelProvider:
    """Pick the provider for this process.

    The settings and Anthropic imports are deliberately lazy: this package
    must import (and be testable) without a config file, and without the
    `anthropic` SDK installed.
    """
    if settings is None:
        try:
            from stock_moves.settings import get_settings
        except ImportError:  # settings module not present yet
            settings = None
        else:
            settings = get_settings()

    api_key = getattr(settings, "anthropic_api_key", None)
    if api_key:
        try:
            from stock_moves.providers.anthropic import AnthropicProvider
        except ImportError:  # SDK or module missing: fall back, never crash
            pass
        else:
            kwargs: dict[str, Any] = {"api_key": api_key}
            model = getattr(settings, "anthropic_model", None)
            if model:
                kwargs["model"] = model
            return AnthropicProvider(**kwargs)

    return HeuristicProvider()
