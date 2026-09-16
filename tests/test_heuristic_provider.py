"""Tests for the keyless provider. No DB, no network: every context is built
by hand and every tool is a fake that records its calls."""

from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any

from stock_moves.providers import get_provider
from stock_moves.providers.base import (
    TOOL_SPECS,
    ArticleInput,
    ArticleScore,
    ChatTurn,
    MoveContext,
    ToolFn,
)
from stock_moves.providers.heuristic import HeuristicProvider

PROVIDER = HeuristicProvider()

#: Terms that are meaningful inside the decomposition and meaningless to the
#: reader of an explanation. Every one of these used to reach the user
#: verbatim; the readability contract in `prompts.PLAIN_ENGLISH` bans the same
#: list from the keyed providers, so this is the keyless half of one rule.
JARGON = ("sigma", "z=", "z-score", "idiosyncratic", "pp)", "pp,", "pp.", "beta")


def assert_no_jargon(text: str) -> None:
    lowered = text.lower()
    leaked = [term for term in JARGON if term in lowered]
    assert not leaked, f"jargon reached the reader: {leaked}\n{text}"


def ctx(**overrides: Any) -> MoveContext:
    """A move context with sane defaults; override only what a test needs."""
    base: dict[str, Any] = {
        "ticker": "TEST",
        "company_name": "Testco Industries Inc",
        "date": date(2025, 6, 3),
        "ret": -0.05,
        "ret_z": -2.5,
        "gap_ret": -0.03,
        "intraday_ret": -0.02,
        "vol_z": 2.1,
        "mkt_component": -0.001,
        "sector_component": -0.002,
        "idio_component": -0.047,
        "routing": "company",
        "direction": "down",
        "regime_mkt": "bull",
        "regime_sector": "bull",
        "near_earnings": False,
        "sector": "Technology",
        "industry": None,
        "peers": (),
        "peer_comove": None,
    }
    base.update(overrides)
    return MoveContext(**base)


def art(article_id: int, title: str, source: str | None = "Reuters") -> ArticleInput:
    return ArticleInput(
        id=article_id,
        title=title,
        source=source,
        url=f"https://example.test/{article_id}",
        published_at=datetime(2025, 6, 3, 12, 0, tzinfo=UTC),
    )


def recording_tool(result: Any) -> tuple[ToolFn, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def fn(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return result

    return fn, calls


# --------------------------------------------------------------------------- #
# score_articles
# --------------------------------------------------------------------------- #


def test_company_name_and_ticker_score_top() -> None:
    scores = PROVIDER.score_articles(
        ctx(),
        [
            art(1, "Testco Industries warns on demand"),
            art(2, "Analysts cut TEST to hold"),
            art(3, "TESTCO shares slide after guidance"),
        ],
    )
    assert [s.relevance for s in scores] == [1.0, 1.0, 1.0]
    assert {s.category for s in scores} == {"company"}
    assert [s.article_id for s in scores] == [1, 2, 3]


def test_first_word_of_a_multiword_name_counts() -> None:
    move = ctx(ticker="AMD", company_name="Advanced Micro Devices, Inc.")
    (score,) = PROVIDER.score_articles(move, [art(1, "Advanced chips ship early")])
    assert (score.relevance, score.category) == (1.0, "company")


def test_short_ticker_and_substring_do_not_match_the_company() -> None:
    # "GE" is under the 3-letter rule; "latest" merely contains "test".
    short = ctx(ticker="GE", company_name="Zeta Works", routing="macro")
    (score,) = PROVIDER.score_articles(short, [art(1, "GE lifts full-year guidance")])
    assert (score.relevance, score.category) == (0.1, "macro")

    (score2,) = PROVIDER.score_articles(
        ctx(routing="industry"), [art(2, "Latest wrap: quiet session for equities")]
    )
    assert (score2.relevance, score2.category) == (0.1, "industry")


def test_peer_ticker_and_industry_score_as_industry() -> None:
    move = ctx(
        ticker="ZTA",
        company_name="Zeta Works",
        peers=("AMD", "NVDA"),
        industry="Semiconductors",
    )
    scores = PROVIDER.score_articles(
        move,
        [
            art(1, "AMD slips after a downgrade"),
            art(2, "Semiconductors face a rough week"),
        ],
    )
    assert [(s.relevance, s.category) for s in scores] == [
        (0.7, "industry"),
        (0.7, "industry"),
    ]


def test_macro_terms_score_as_macro() -> None:
    move = ctx(ticker="ZTA", company_name="Zeta Works")
    scores = PROVIDER.score_articles(
        move,
        [
            art(1, "Federal Reserve holds rates steady"),
            art(2, "Inflation cooled again in May"),
        ],
    )
    assert [(s.relevance, s.category) for s in scores] == [
        (0.5, "macro"),
        (0.5, "macro"),
    ]


def test_company_beats_peer_beats_macro() -> None:
    move = ctx(peers=("AMD",), industry="Semiconductors")
    (score,) = PROVIDER.score_articles(
        move, [art(1, "Testco and AMD slide as the Federal Reserve holds rates")]
    )
    assert (score.relevance, score.category) == (1.0, "company")


# --------------------------------------------------------------------------- #
# explain
# --------------------------------------------------------------------------- #


def test_explain_returns_unexplained_when_nothing_matched() -> None:
    scored = [(art(1, "Quiet session for equities"), ArticleScore(1, 0.1, "company"))]
    result = PROVIDER.explain(ctx(ret_z=-2.5), scored)
    assert result.unexplained is True
    assert result.primary_category == "unexplained"
    assert result.confidence == 0.2
    assert result.cited_article_ids == ()
    assert "no headline" in result.summary
    assert "down 5.0%" in result.summary


def test_an_extreme_move_is_explained_even_with_no_headlines() -> None:
    result = PROVIDER.explain(ctx(ret_z=-3.6), [])
    assert result.unexplained is False
    assert result.primary_category == "company"
    assert result.cited_article_ids == ()


def test_explain_routes_cites_at_most_three_and_caps_confidence() -> None:
    # Each headline names an event ("cuts guidance"), so the confidence is the
    # ungrounded-headline check's pass case and reaches the 0.9 cap.
    articles = [art(i, f"Testco Industries cuts guidance, headline {i}") for i in range(1, 5)]
    scored = [(a, ArticleScore(a.id, 1.0, "company")) for a in articles]
    result = PROVIDER.explain(ctx(near_earnings=True), scored)

    assert result.unexplained is False
    assert result.primary_category == "company"
    assert result.cited_article_ids == (1, 2, 3)
    assert result.confidence == 0.9
    assert "was down 5.0%" in result.summary
    # The size as a multiple of an ordinary day, and the decomposition as a
    # share of the move -- not a z-score and not percentage points.
    assert "twice the size of a typical day" in result.summary
    assert "4.7 of the 5.0 points" in result.summary
    assert "within a day of the company's own earnings" in result.summary
    # One headline, named as coverage. Three pasted into a sentence is what
    # made the old summaries unreadable.
    assert "the day's coverage led with" in result.summary
    assert "(Reuters)" in result.summary
    assert "headline 2" not in result.summary
    assert_no_jargon(result.summary)


def test_explain_mentions_macro_windows_and_peer_comovement() -> None:
    scored = [(art(1, "Inflation cooled again"), ArticleScore(1, 0.5, "macro"))]
    result = PROVIDER.explain(
        ctx(routing="macro", near_fomc=True, near_cpi=True, peer_comove=-0.031),
        scored,
    )
    assert result.primary_category == "macro"
    assert "Federal Reserve rate decision" in result.summary
    assert "inflation release" in result.summary
    assert "Comparable companies were down 3.1%" in result.summary
    assert_no_jargon(result.summary)


def test_explain_without_a_decomposition() -> None:
    scored = [(art(1, "Testco Industries warns on revenue"), ArticleScore(1, 1.0, "company"))]
    result = PROVIDER.explain(
        ctx(mkt_component=None, sector_component=None, idio_component=None), scored
    )
    assert "no market-versus-company breakdown" in result.summary
    # 0.3 + 0.15*1, no earnings bonus, no dominance bonus.
    assert result.confidence == 0.45


def test_an_event_headline_keeps_the_full_confidence() -> None:
    """One relevant headline naming an event: 0.3 + 0.15*1 + 0.1 dominance."""
    scored = [
        (art(1, "Testco Industries cuts its revenue forecast"), ArticleScore(1, 0.9, "company"))
    ]
    result = PROVIDER.explain(ctx(), scored)

    assert result.confidence == 0.55
    assert "attribution is weak" not in result.summary


def test_generic_commentary_headlines_lower_the_confidence() -> None:
    """Headlines that match the company but name no event are weak evidence."""
    titles = ("Can TEST stock reach $350?", "How to play TEST shares")
    scored = [(art(i, title), ArticleScore(i, 0.9, "company")) for i, title in enumerate(titles, 1)]
    result = PROVIDER.explain(ctx(), scored)

    # 0.3 + 0.15*2 + 0.1 dominance = 0.7, less the 0.25 ungrounded penalty.
    assert result.confidence == 0.45
    assert (
        "The matched headlines mention the company but do not name a specific "
        "event, so this attribution is weak."
    ) in result.summary


def test_suggest_peers_is_empty_without_a_key() -> None:
    assert PROVIDER.suggest_peers("TEST", "Testco Industries Inc", "Technology", None) == []


# --------------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------------- #


def _chat_tools(
    provider: str = "heuristic",
    summary: str = "Testco fell on a guidance cut.",
) -> tuple[dict[str, ToolFn], dict[str, list[dict[str, Any]]]]:
    move_payload = {
        "date": "2025-06-03",
        "ret": -0.05,
        "ret_z": -2.5,
        "routing": "company",
        "explanation": {"summary": summary, "provider": provider},
        "articles": [
            {
                "id": i,
                "title": f"Headline {i}",
                "source": "Reuters",
                "published_at": "2025-06-03T12:00:00",
            }
            for i in range(1, 5)
        ],
    }
    get_move, get_move_calls = recording_tool(move_payload)
    list_moves, list_moves_calls = recording_tool([move_payload])
    search_news, search_news_calls = recording_tool(
        [{"id": 9, "title": "Testco in the news", "source": "WSJ"}]
    )
    tools: dict[str, ToolFn] = {
        "get_move": get_move,
        "list_moves": list_moves,
        "search_news": search_news,
    }
    calls = {
        "get_move": get_move_calls,
        "list_moves": list_moves_calls,
        "search_news": search_news_calls,
    }
    return tools, calls


def test_chat_with_a_date_calls_get_move_once() -> None:
    tools, calls = _chat_tools()
    reply = PROVIDER.chat([ChatTurn("user", "what happened to TEST on 2025-06-03")], tools, None)
    assert calls["get_move"] == [{"ticker": "TEST", "date": "2025-06-03"}]
    assert calls["list_moves"] == [] and calls["search_news"] == []
    assert [tc.name for tc in reply.tool_calls] == ["get_move"]
    # The stored summary was written by the heuristic, so it is regenerated
    # from the move's own numbers rather than replayed -- rows written before
    # `narrate` existed still read the way the current code speaks.
    assert "Tuesday, 3 June 2025 -- down 5.0%" in reply.reply
    assert reply.reply.count("Headline") == 3  # capped at three
    assert_no_jargon(reply.reply)


def test_chat_replays_a_summary_a_model_wrote() -> None:
    """A keyed provider read the headlines, so its prose says more than the
    decomposition can. Only the heuristic's own summaries are regenerated."""
    tools, _ = _chat_tools(provider="openai", summary="Guidance cut sank the stock.")
    reply = PROVIDER.chat([ChatTurn("user", "what happened to TEST on 2025-06-03")], tools, None)
    assert "Guidance cut sank the stock." in reply.reply


def test_chat_surfaces_the_get_move_error_instead_of_a_blank_explanation() -> None:
    """get_move returns {"error": ...} for an unknown date; the reply must say so
    rather than pretending a move exists with no explanation yet."""
    get_move, _ = recording_tool({"error": "no move for TEST on 2025-06-04"})
    tools: dict[str, ToolFn] = {"get_move": get_move}
    reply = PROVIDER.chat([ChatTurn("user", "what happened to TEST on 2025-06-04")], tools, None)
    # Said as a sentence rather than replayed as the raw developer string.
    assert reply.reply == "TEST did not have a major move on Wednesday, 4 June 2025."
    assert "No explanation yet" not in reply.reply


def test_chat_asking_for_news_calls_search_news() -> None:
    tools, calls = _chat_tools()
    reply = PROVIDER.chat([ChatTurn("user", "any news on TEST")], tools, None)
    assert calls["search_news"] == [{"ticker": "TEST", "limit": 10}]
    assert calls["get_move"] == [] and calls["list_moves"] == []
    assert "Testco in the news" in reply.reply


def test_chat_asking_why_it_dropped_lists_down_moves() -> None:
    tools, calls = _chat_tools()
    reply = PROVIDER.chat([ChatTurn("user", "why did TEST drop")], tools, None)
    # No superlative in the question, so the survey ranking and row count
    # stand: five moves, ranked by how unusual each day was.
    assert calls["list_moves"] == [
        {"ticker": "TEST", "direction": "down", "order": "z", "limit": 5}
    ]
    assert calls["get_move"] == [] and calls["search_news"] == []
    # The synthesis: what the set is, then each move as a dated block. No
    # z-score reaches the reader. The header names the ranking it got, so a
    # z-ordered list is not called "the biggest fall".
    assert "the most unusual fall in the data" in reply.reply
    assert "Tuesday, 3 June 2025 -- down 5.0%" in reply.reply
    assert_no_jargon(reply.reply)


def test_chat_up_words_flip_the_direction() -> None:
    tools, calls = _chat_tools()
    PROVIDER.chat([ChatTurn("user", "biggest TEST rally")], tools, None)
    assert calls["list_moves"][0]["direction"] == "up"


def test_chat_without_a_ticker_calls_nothing() -> None:
    tools, calls = _chat_tools()
    reply = PROVIDER.chat([ChatTurn("user", "why did it move so much?")], tools, None)
    assert reply.tool_calls == []
    assert all(c == [] for c in calls.values())
    assert "Tell me a ticker" in reply.reply


def test_chat_prefers_the_session_ticker_and_the_newest_user_turn() -> None:
    tools, calls = _chat_tools()
    history = [
        ChatTurn("user", "hello"),
        ChatTurn("assistant", "hi, ask me about a move"),
        ChatTurn("user", "any headlines?"),
    ]
    PROVIDER.chat(history, tools, "NVDA")
    assert calls["search_news"] == [{"ticker": "NVDA", "limit": 10}]


def test_chat_reports_a_tool_failure_instead_of_raising() -> None:
    def boom(**kwargs: Any) -> Any:
        raise RuntimeError("db is locked")

    reply = PROVIDER.chat([ChatTurn("user", "why did TEST drop")], {"list_moves": boom}, None)
    assert "db is locked" in reply.reply


# --------------------------------------------------------------------------- #
# base.py plumbing
# --------------------------------------------------------------------------- #


def test_move_context_from_objects_reads_attributes_by_name() -> None:
    move = SimpleNamespace(
        ticker="NVDA",
        date=date(2025, 8, 28),
        ret=-0.061,
        ret_z=-2.8,
        gap_ret=-0.04,
        intraday_ret=-0.021,
        vol_z=3.2,
        mkt_component=-0.005,
        sector_component=-0.012,
        idio_component=-0.044,
        routing="company",
        direction="down",
        regime_mkt="bull",
        regime_sector="bear",
        near_earnings=True,
        peer_comove=-0.018,
    )
    company = SimpleNamespace(
        ticker="NVDA",
        name="NVIDIA Corporation",
        sector="Technology",
        industry="Semiconductors",
        peers=["AMD", "AVGO"],
    )
    built = MoveContext.from_objects(move, company)

    assert built.ticker == "NVDA"
    assert built.company_name == "NVIDIA Corporation"
    assert built.peers == ("AMD", "AVGO")
    assert built.near_earnings is True
    # Columns the row does not carry fall back rather than raising.
    assert built.near_fomc is False and built.near_cpi is False
    assert built.industry == "Semiconductors"


def test_move_context_from_objects_tolerates_a_sparse_row() -> None:
    built = MoveContext.from_objects(
        SimpleNamespace(ticker="ZTA", date=date(2025, 1, 2), ret=-0.03),
        SimpleNamespace(),
    )
    assert built.company_name == "ZTA"
    assert built.direction == "down"
    assert built.routing == "company"
    assert built.peers == ()
    assert built.ret_z == 0.0
    assert built.mkt_component is None


def test_article_input_from_object() -> None:
    built = ArticleInput.from_object(
        SimpleNamespace(
            id=7,
            title="Testco Industries warns",
            source="Reuters",
            url="https://example.test/7",
            published_at=datetime(2025, 6, 3, 12, 0, tzinfo=UTC),
            language="en",
        )
    )
    assert built.id == 7 and built.source == "Reuters"
    assert built.title == "Testco Industries warns"


def test_tool_specs_cover_the_three_read_functions() -> None:
    names = [spec["name"] for spec in TOOL_SPECS]
    assert names == ["list_moves", "get_move", "search_news"]
    for spec in TOOL_SPECS:
        assert spec["input_schema"]["type"] == "object"
        assert spec["description"]
    get_move = next(s for s in TOOL_SPECS if s["name"] == "get_move")
    assert get_move["input_schema"]["required"] == ["date"]


def test_get_provider_falls_back_to_the_heuristic_without_a_key() -> None:
    provider = get_provider(SimpleNamespace(anthropic_api_key=None))
    assert isinstance(provider, HeuristicProvider)
    assert provider.name == "heuristic"


# --------------------------------------------------------------------------- #
# Superlative questions, and two polish items in the same router
# --------------------------------------------------------------------------- #


def test_a_superlative_question_asks_for_one_move_ranked_by_size() -> None:
    """ "Drop the most" is a question about size and about one day. Ranking it
    by z and answering with five rows is the bug this fixes."""
    tools, calls = _chat_tools()
    PROVIDER.chat([ChatTurn("user", "why did TEST drop the most this year?")], tools, None)
    assert calls["list_moves"] == [
        {"ticker": "TEST", "direction": "down", "order": "pct", "limit": 1}
    ]


def test_single_biggest_one_day_drop_is_one_move() -> None:
    """ "one-day" hides the word "one"; it is not a row count."""
    tools, calls = _chat_tools()
    PROVIDER.chat(
        [ChatTurn("user", "what was TEST's single biggest one-day drop, and by how much?")],
        tools,
        None,
    )
    assert calls["list_moves"][0]["limit"] == 1
    assert calls["list_moves"][0]["order"] == "pct"


def test_a_plural_superlative_keeps_the_list() -> None:
    tools, calls = _chat_tools()
    PROVIDER.chat([ChatTurn("user", "what were TEST's biggest falls?")], tools, None)
    assert calls["list_moves"][0]["limit"] == 5
    assert calls["list_moves"][0]["order"] == "pct"


def test_a_named_count_keeps_the_list() -> None:
    tools, calls = _chat_tools()
    PROVIDER.chat([ChatTurn("user", "TEST's worst 3 sessions")], tools, None)
    assert calls["list_moves"][0]["limit"] == 5

    tools, calls = _chat_tools()
    PROVIDER.chat([ChatTurn("user", "the three worst TEST sessions")], tools, None)
    assert calls["list_moves"][0]["limit"] == 5


def test_a_plain_question_keeps_the_survey_ranking() -> None:
    tools, calls = _chat_tools()
    PROVIDER.chat([ChatTurn("user", "what moved TEST around?")], tools, None)
    assert calls["list_moves"][0]["order"] == "z"
    assert calls["list_moves"][0]["limit"] == 5


def test_a_superlative_header_says_biggest() -> None:
    tools, _ = _chat_tools()
    reply = PROVIDER.chat([ChatTurn("user", "TEST's biggest drop ever")], tools, None)
    assert "the biggest fall in the data" in reply.reply
    assert_no_jargon(reply.reply)


def test_the_no_ticker_example_date_is_inside_the_data_window() -> None:
    """2025-08-28 was outside it, so the first thing a new user copied was a
    question with no answer. It matches the chat box's placeholder."""
    tools, calls = _chat_tools()
    reply = PROVIDER.chat([ChatTurn("user", "so what happened?")], tools, None)
    assert "2026-01-20" in reply.reply
    assert "2025-08-28" not in reply.reply
    assert all(c == [] for c in calls.values())


def test_an_unrecognised_get_move_error_is_passed_through() -> None:
    """Only the documented shape is rewritten; anything else reaches the user
    as it stands rather than being swallowed."""
    get_move, _ = recording_tool({"error": "database is on fire"})
    reply = PROVIDER.chat(
        [ChatTurn("user", "what happened to TEST on 2026-03-02")], {"get_move": get_move}, None
    )
    assert reply.reply == "database is on fire"


def test_the_get_move_error_names_the_day_in_english() -> None:
    get_move, _ = recording_tool({"error": "no move for AAPL on 2026-03-02"})
    reply = PROVIDER.chat(
        [ChatTurn("user", "what happened to AAPL on 2026-03-02")], {"get_move": get_move}, None
    )
    assert reply.reply == "AAPL did not have a major move on Monday, 2 March 2026."


def test_the_list_moves_spec_documents_the_order_argument() -> None:
    spec = next(s for s in TOOL_SPECS if s["name"] == "list_moves")
    order = spec["input_schema"]["properties"]["order"]
    assert order["enum"] == ["z", "pct"]
    assert order["default"] == "z"
    assert "biggest" in order["description"]
    # The description no longer promises one ordering unconditionally.
    assert "sorted by absolute z-score" not in spec["description"]
