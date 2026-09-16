"""The `/chat` route (DESIGN section 6).

The route is thin on purpose. It owns three things and nothing else:

1. **The tool bindings.** `make_tools` closes the request's `Session` and the
   session ticker over the read functions in `stock_moves.queries`, so a
   provider calls `list_moves(ticker="NVDA", limit=5)` and gets exactly the
   dicts the HTTP routes return — the model and the API client never see two
   different shapes. The three names are the ones promised by
   `providers.base.TOOL_SPECS`.
2. **The session.** A turn with no `session_id` mints one; the history is read
   back from `chat_messages`, so the tool-calling loop is stateless and the
   transcript is the state.
3. **Nothing about the model.** Which provider answers is decided by
   `get_provider_dep` (a key or no key), and the tool-calling loop lives in the
   provider. Plain JSON, no SSE.

Tool arguments arrive from a language model, so every accessor is defensive:
each tool takes `**kwargs`, ignores what it does not know, coerces the types it
does, and lets a genuinely bad argument raise `ValueError` — the provider turns
that into a message to the user rather than a 500.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends

from stock_moves import narrate
from stock_moves.api.deps import get_db, get_provider_dep
from stock_moves.api.schemas import ChatRequest, ChatResponse, ToolCallOut
from stock_moves.db import Session
from stock_moves.providers import ChatTurn, ModelProvider, ToolFn
from stock_moves.queries import (
    MoveFilters,
    article_to_dict,
    articles_for_move,
    chat_history,
    get_company,
    get_explanation,
    get_move,
    list_moves,
    move_to_dict,
    save_chat_message,
    search_news,
)

__all__ = ["make_tools", "router"]

router = APIRouter(tags=["chat"])

SessionDep = Annotated[Session, Depends(get_db)]
ProviderDep = Annotated[ModelProvider, Depends(get_provider_dep)]

#: Roles that may be replayed to a provider. `chat_messages` also stores the
#: assistant's tool calls, but a provider is handed the conversation, not the
#: previous turns' tool plumbing.
_CHAT_ROLES = frozenset({"user", "assistant"})

#: Tool defaults, matching the `default` values advertised in `TOOL_SPECS`.
_MOVES_LIMIT = 10
_ARTICLES_LIMIT = 10

#: Headlines attached to each move in a *list*. Three is what fits in a
#: readable block; the full set is one `get_move` away.
_LIST_ARTICLES = 3
_NEWS_LIMIT = 20


# --------------------------------------------------------------------------- #
# Argument coercion
# --------------------------------------------------------------------------- #


def _resolve_ticker(kwargs: dict[str, Any], default_ticker: str | None) -> str:
    """The tool's `ticker`, else the session ticker; upper-cased.

    Raises `ValueError` when neither is present: the provider reports that to
    the user, which is the right outcome for "what were the biggest moves?"
    asked with no ticker anywhere.
    """
    raw = kwargs.get("ticker") or default_ticker
    if raw is None or not str(raw).strip():
        raise ValueError("ticker required")
    return str(raw).strip().upper()


def _as_date(value: Any) -> date | None:
    """A `YYYY-MM-DD` string (or a real date) as a `date`; blank means unset.

    `date.fromisoformat` raises `ValueError` on anything else, which is the
    message the user should see.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip())


def _required_date(value: Any) -> date:
    parsed = _as_date(value)
    if parsed is None:
        raise ValueError("date required")
    return parsed


def _as_limit(value: Any, default: int) -> int:
    """A positive row cap. Unset, unparsable or non-positive falls back."""
    if value is None:
        return default
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return default
    return limit if limit > 0 else default


def _as_direction(value: Any) -> str | None:
    """`up` / `down`, or `None` for "either".

    `HeuristicProvider.chat` always sends a `direction` key and it is often
    `None`, so the "not filtering" case has to be the normal one.
    """
    if value is None:
        return None
    direction = str(value).strip().lower()
    return direction if direction in {"up", "down"} else None


def _as_text(value: Any) -> str | None:
    """A non-empty search string, or `None`."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


def make_tools(session: Session, default_ticker: str | None) -> dict[str, ToolFn]:
    """The `TOOL_SPECS` tools, bound to this request's session and ticker.

    Every value returned is JSON-serialisable, because it is the same dict the
    HTTP routes return and it is also echoed back to the client in
    `tool_calls[].output`.
    """

    def tool_list_moves(**kwargs: Any) -> list[dict[str, Any]]:
        """Biggest moves for the ticker, with each cached explanation.

        Both thresholds are zeroed: every stored move is already a move, so
        the ordering by `abs(ret_z)` and `limit` do the selecting. Re-applying
        the detection thresholds here would hide moves the ingest flagged.
        """
        ticker = _resolve_ticker(kwargs, default_ticker)
        filters = MoveFilters(
            start=_as_date(kwargs.get("start")),
            end=_as_date(kwargs.get("end")),
            z_threshold=0.0,
            pct_threshold=0.0,
            direction=_as_direction(kwargs.get("direction")),
            limit=_as_limit(kwargs.get("limit"), _MOVES_LIMIT),
        )
        company = get_company(session, ticker)
        return [
            _enriched(
                move_to_dict(
                    move,
                    None if move.id is None else get_explanation(session, move.id),
                    (
                        []
                        if move.id is None
                        else articles_for_move(session, move.id, limit=_LIST_ARTICLES)
                    ),
                ),
                ticker,
                company,
            )
            for move in list_moves(session, ticker, filters)
        ]

    def tool_get_move(**kwargs: Any) -> dict[str, Any]:
        """One move with its linked articles and explanation.

        A move with no explanation is returned as it stands: computing one on
        demand is the ticker route's job (DESIGN section 4), and doing it here
        would let a chat turn silently spend a model call.
        """
        ticker = _resolve_ticker(kwargs, default_ticker)
        on = _required_date(kwargs.get("date"))
        move = get_move(session, ticker, on)
        if move is None or move.id is None:
            return {"error": f"no move for {ticker} on {on.isoformat()}"}
        return _enriched(
            move_to_dict(
                move,
                get_explanation(session, move.id),
                articles_for_move(session, move.id, limit=_ARTICLES_LIMIT),
            ),
            ticker,
            get_company(session, ticker),
        )

    def tool_search_news(**kwargs: Any) -> list[dict[str, Any]]:
        """Headlines linked to the ticker's moves, most relevant first."""
        ticker = _resolve_ticker(kwargs, default_ticker)
        rows = search_news(
            session,
            ticker,
            query=_as_text(kwargs.get("query")),
            start=_as_date(kwargs.get("start")),
            end=_as_date(kwargs.get("end")),
            limit=_as_limit(kwargs.get("limit"), _NEWS_LIMIT),
        )
        return [article_to_dict(article, link) for article, link in rows]

    return {
        "list_moves": tool_list_moves,
        "get_move": tool_get_move,
        "search_news": tool_search_news,
    }


def _enriched(move: dict[str, Any], ticker: str, company: Any | None) -> dict[str, Any]:
    """Add the two things a *reader* of this move needs and the columns lack.

    `company` so an answer can say "Tesla" rather than "TSLA", and `narrative`
    -- the decomposition already turned into English by `narrate`. A model
    handed the narration is synthesising across moves and weighing headlines
    rather than converting factor loadings into prose itself, which is the step
    that used to leak "idiosyncratic -8.6pp" through to the reader.
    """
    name = "" if company is None else str(getattr(company, "name", "") or "")
    move["company"] = narrate.short_company_name(name or ticker, ticker)
    move["narrative"] = " ".join(
        narrate.summary_sentences(narrate.MoveFacts.from_mapping(move, ticker, name))
    )
    return move


# --------------------------------------------------------------------------- #
# Route
# --------------------------------------------------------------------------- #


@router.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    session: SessionDep,
    provider: ProviderDep,
) -> ChatResponse:
    """Answer one chat turn against the stored moves, news and explanations.

    The provider sees the trimmed transcript plus this turn, and the tools
    bound to `req.ticker`; both turns are persisted, the assistant's with the
    tool calls it made so the UI can show its work.
    """
    session_id = req.session_id or uuid4().hex

    history = [
        ChatTurn(role=message.role, content=message.content)
        for message in chat_history(session, session_id)
        if message.role in _CHAT_ROLES
    ]
    history.append(ChatTurn(role="user", content=req.message))
    save_chat_message(session, session_id, "user", req.message)

    reply = provider.chat(history, make_tools(session, req.ticker), req.ticker)
    tool_calls = [
        {"name": call.name, "input": call.input, "output": call.output} for call in reply.tool_calls
    ]
    save_chat_message(session, session_id, "assistant", reply.reply, tool_calls)

    return ChatResponse(
        reply=reply.reply,
        session_id=session_id,
        tool_calls=[ToolCallOut(**call) for call in tool_calls],
    )
