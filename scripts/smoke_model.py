"""Prove the live model path, because v1 shipped with that path unproven.
This is the first thing to run on any machine: one keyed-provider call and one raw SDK call.

Both calls matter. The keyed providers degrade to `HeuristicProvider` on *any*
exception by design (`providers/openai.py`), so a dead key looks like a quiet
empty answer through the app's own path. The raw SDK call is the one that
cannot swallow anything, so a 401 or a missing credit shows up as a 401.

    uv run python scripts/smoke_model.py

Exit codes: 0 both calls proved the path, 1 the SDK call failed or the
provider call degraded, 2 no key is configured at all.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT_GUESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT_GUESS / "src"))

from dotenv import load_dotenv

from stock_moves.providers import get_provider
from stock_moves.settings import REPO_ROOT, Settings, get_settings

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_NO_KEY = 2

# One cheap, well-known identity, so a wrong answer is obvious to a human
# reader and the prompt has nothing to do with whatever is in the database.
AAPL_TICKER = "AAPL"
AAPL_NAME = "Apple Inc."
AAPL_SECTOR = "Technology"
AAPL_INDUSTRY = "Consumer Electronics"

OK_PROMPT = "Reply with the single word OK."
OK_MAX_TOKENS = 8


def main() -> int:
    """Run the smoke check and return the process exit code."""
    # Explicit path rather than dotenv's walk-up search: the script must read
    # the same file the app reads, wherever it is invoked from.
    load_dotenv(REPO_ROOT / ".env")
    get_settings.cache_clear()
    settings = get_settings()

    vendor = _vendor(settings)
    if vendor is None:
        print("no key")
        print(f"  looked in: {REPO_ROOT / '.env'} (OPENAI_API_KEY, ANTHROPIC_API_KEY)")
        return EXIT_NO_KEY

    provider = get_provider(settings)
    if provider.name != vendor:
        # A key is set but `get_provider` still fell back: the SDK is missing
        # or the provider module failed to import.
        print(f"provider: {provider.name} (expected {vendor})")
        print(f"  {vendor.upper()}_API_KEY is set but the keyed provider could not be built;")
        print(f"  check that the `{vendor}` SDK is installed in this environment.")
        return EXIT_FAILED

    model = str(getattr(provider, "model", "?"))
    print(f"provider: {provider.name}")
    print(f"model:    {model}")
    print(f"env:      {REPO_ROOT / '.env'}")
    print()

    peers_ok = _run_provider_call(provider)
    print()
    sdk_ok = _run_sdk_call(provider, vendor, model)
    print()

    if sdk_ok and peers_ok:
        print("RESULT: the live model path works.")
        return EXIT_OK
    print("RESULT: the live model path is NOT proven. Do not trust keyed output.")
    return EXIT_FAILED


def _vendor(settings: Settings) -> str | None:
    """The vendor the app would pick: OpenAI first, then Anthropic, else None."""
    if settings.openai_api_key:
        return "openai"
    if settings.anthropic_api_key:
        return "anthropic"
    return None


def _run_provider_call(provider: Any) -> bool:
    """One `suggest_peers` call through the app's own path.

    True only when peers came back: an empty list is what a dead key looks
    like here, because the provider catches every exception.
    """
    started = time.monotonic()
    try:
        peers = provider.suggest_peers(AAPL_TICKER, AAPL_NAME, AAPL_SECTOR, AAPL_INDUSTRY)
    except Exception as exc:  # noqa: BLE001 - a provider is not supposed to raise at all
        elapsed_ms = _elapsed_ms(started)
        print(f"provider.suggest_peers(AAPL): RAISED  ({elapsed_ms} ms)")
        _print_exception(exc)
        return False
    elapsed_ms = _elapsed_ms(started)
    if peers:
        print(f"provider.suggest_peers(AAPL): {list(peers)}  ({elapsed_ms} ms)")
        return True
    print(f"provider.suggest_peers(AAPL): []  ({elapsed_ms} ms)")
    print("  empty: the provider degraded to the heuristic and swallowed the reason.")
    print("  the raw SDK call below is the one that will name it.")
    return False


def _run_sdk_call(provider: Any, vendor: str, model: str) -> bool:
    """One raw SDK call, whose failure is visible rather than swallowed.

    The provider's own client is reused deliberately: the point is to exercise
    the exact transport the app uses, key and all, not a lookalike.
    """
    client = getattr(provider, "_client", None)
    if client is None:
        print("raw SDK call: no client on the provider; nothing to test.")
        return False

    started = time.monotonic()
    try:
        text = _say_ok(client, vendor, model)
    except Exception as exc:  # noqa: BLE001 - this is the call whose failure we want
        elapsed_ms = _elapsed_ms(started)
        print(f"raw SDK call: FAILED  ({elapsed_ms} ms)")
        _print_exception(exc)
        return False
    elapsed_ms = _elapsed_ms(started)
    print(f"raw SDK call: {text!r}  ({elapsed_ms} ms)")
    return True


def _say_ok(client: Any, vendor: str, model: str) -> str:
    """Ask the model for the word OK over the vendor's native call shape."""
    if vendor == "openai":
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=OK_MAX_TOKENS,
            messages=[{"role": "user", "content": OK_PROMPT}],
        )
        return (response.choices[0].message.content or "").strip()
    response = client.messages.create(
        model=model,
        max_tokens=OK_MAX_TOKENS,
        messages=[{"role": "user", "content": OK_PROMPT}],
    )
    return "".join(
        str(getattr(block, "text", "")) for block in getattr(response, "content", [])
    ).strip()


def _print_exception(exc: BaseException) -> None:
    """Class, message and HTTP status, when the SDK exposes one."""
    print(f"  exception: {type(exc).__module__}.{type(exc).__name__}")
    print(f"  message:   {exc}")
    status = _http_status(exc)
    if status is not None:
        print(f"  http status: {status}")


def _http_status(exc: BaseException) -> int | None:
    """The HTTP status, from whichever attribute the SDK put it on."""
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


if __name__ == "__main__":
    raise SystemExit(main())
