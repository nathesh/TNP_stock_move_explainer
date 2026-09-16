"""The model layer: one interface, three implementations, one selector.

`get_provider()` is the only thing the rest of the app calls. It returns a
keyed provider when a key is configured -- OpenAI first, then Anthropic --
and the heuristic provider otherwise, so the app runs with zero keys
(DESIGN section 4).

OpenAI is preferred when both keys are set, because that is the deployment
this app is configured for; setting only `ANTHROPIC_API_KEY` still selects
Anthropic, and the swap costs one environment variable.
"""

from __future__ import annotations

from importlib import import_module
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
    Relations,
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
    "Relations",
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

    provider = _keyed_provider(settings, "openai") or _keyed_provider(settings, "anthropic")
    return provider if provider is not None else HeuristicProvider()


def _keyed_provider(settings: Any, vendor: str) -> ModelProvider | None:
    """Build `vendor`'s provider, or None when it has no key or no SDK.

    The import is lazy and its failure is not: a missing SDK is the same
    situation as a missing key, and both mean "try the next one".
    """
    api_key = getattr(settings, f"{vendor}_api_key", None)
    if not api_key:
        return None
    try:
        module = import_module(f"stock_moves.providers.{vendor}")
    except ImportError:  # SDK or module missing: fall back, never crash
        return None
    cls = getattr(module, "OpenAIProvider" if vendor == "openai" else "AnthropicProvider")
    kwargs: dict[str, Any] = {"api_key": api_key}
    model = getattr(settings, f"{vendor}_model", None)
    if model:
        kwargs["model"] = model
    return cls(**kwargs)
