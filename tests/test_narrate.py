"""Tests for the phrasing layer.

`narrate` is the one place that decides how a quantity is *said*, so these are
mostly assertions about words. Two things are being pinned down: that the
jargon never reaches the reader, and that each sentence is still *true* of the
numbers it came from -- a readable explanation that misstates the
decomposition would be worse than the unreadable one it replaced.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from stock_moves.narrate import (
    ATTRIBUTION_CAVEAT,
    MoveFacts,
    attribution_sentence,
    count_word,
    narrate_moves,
    narrate_news,
    peers_sentence,
    short_company_name,
    size_phrase,
    summary_sentences,
)

JARGON = ("sigma", "z=", "z-score", "idiosyncratic", "beta", "pp")


def facts(**overrides: Any) -> MoveFacts:
    base: dict[str, Any] = {
        "ticker": "TSLA",
        "company": "Tesla",
        "day": date(2026, 7, 23),
        "ret": -0.145,
        "ret_z": -4.0,
        "routing": "company",
        "near_earnings": True,
        "mkt_component": -0.022,
        "sector_component": -0.036,
        "idio_component": -0.086,
        "peer_comove": -0.018,
    }
    base.update(overrides)
    return MoveFacts(**base)


#: A real move that used to make the app contradict itself two lines apart:
#: the sector carried 5.3 of the 4.4 points, and the peers sentence then said
#: NVDA had moved largely on its own.
NVDA: dict[str, Any] = {
    "ticker": "NVDA",
    "company": "NVIDIA",
    "day": date(2026, 1, 20),
    "ret": -0.0438,
    "ret_z": -3.0,
    "routing": "macro",
    "near_earnings": False,
    "mkt_component": 0.028,
    "sector_component": -0.053,
    "idio_component": -0.019,
    "peer_comove": -0.012,
}


def move_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "date": "2026-07-23",
        "ret": -0.145,
        "ret_z": -4.0,
        "routing": "company",
        "near_earnings": True,
        "mkt_component": -0.022,
        "sector_component": -0.036,
        "idio_component": -0.086,
        "peer_comove": -0.018,
        "articles": [
            {"title": "Tesla stock stumbles after latest earnings report", "source": "CNBC"}
        ],
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# Company names
# --------------------------------------------------------------------------- #


def test_legal_suffixes_are_dropped() -> None:
    assert short_company_name("Tesla, Inc.") == "Tesla"
    assert short_company_name("Advanced Micro Devices Inc") == "Advanced Micro Devices"
    assert short_company_name("Alphabet Inc. Class A") == "Alphabet"


def test_a_meaningful_trailing_word_is_kept() -> None:
    """ "Motor" is part of the name; "Corp" is not."""
    assert short_company_name("Ford Motor Company") == "Ford Motor"


def test_a_name_that_is_all_suffix_falls_back_to_the_ticker() -> None:
    assert short_company_name("Holdings Ltd", "HLD") == "HLD"


# --------------------------------------------------------------------------- #
# Size
# --------------------------------------------------------------------------- #


def test_the_z_score_becomes_a_multiple_of_an_ordinary_day() -> None:
    said = size_phrase(-4.0, "TSLA")
    assert "four times the size of a typical day" in said
    assert not any(term in said.lower() for term in JARGON)


def test_a_move_inside_ordinary_noise_is_not_given_a_multiple() -> None:
    """A z near 1 times "a typical day" is a distinction without a difference."""
    assert "ordinary day" in size_phrase(-1.1, "TSLA")
    assert "times" not in size_phrase(-1.1, "TSLA")


def test_a_large_multiple_falls_back_to_digits() -> None:
    assert "about 9 times" in size_phrase(9.2, "TSLA")


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #


def test_the_components_are_said_as_a_share_of_the_move() -> None:
    """The components sum to the return, so they can be stated as points of it
    and the reader can add them up."""
    said = attribution_sentence(facts())
    assert said is not None
    assert "Most of it was Tesla itself" in said
    assert "8.6 of the 14.5 points" in said
    assert "3.6 from the rest of the sector" in said
    assert "2.2 from the wider market" in said


def test_a_component_opposing_the_move_is_called_out() -> None:
    """Reporting its size silently would read as if it had contributed."""
    said = attribution_sentence(facts(mkt_component=0.01))
    assert said is not None
    assert "the wider market pushed 1.0 the other way" in said
    assert "3.6 from the rest of the sector in the same direction" in said


def test_two_components_opposing_the_move_are_called_out_together() -> None:
    """ "Pushing the other way" is said once for both, not once each."""
    said = attribution_sentence(facts(mkt_component=0.01, sector_component=0.004))
    assert said is not None
    assert "1.0 from the wider market and 0.4 from the rest of the sector" in said
    assert "both pushing the other way" in said


def test_a_split_remainder_names_each_direction() -> None:
    """NVDA, 2026-01-20: the sector led, the company's own part went with the
    move and the market against it. "with 1.9 from NVIDIA itself and 2.8 from
    the wider market pushing the other way" lets the participle attach to
    both, so the reader hears that the 1.9 pushed the stock up."""
    said = attribution_sentence(facts(**NVDA))
    assert said is not None
    assert "1.9 from NVIDIA itself in the same direction" in said
    assert "the wider market pushed 2.8 the other way" in said
    assert "1.9 from NVIDIA itself and 2.8 from the wider market pushing the other way" not in said
    assert not any(term in said.lower() for term in JARGON)


def test_a_component_that_rounds_to_nothing_is_left_out() -> None:
    """ "and 0.0 from the wider market" is noise, not a third of the story."""
    said = attribution_sentence(facts(mkt_component=-0.0002))
    assert said is not None
    assert "the wider market" not in said
    assert "3.6 from the rest of the sector" in said


def test_a_dominant_component_that_rounds_to_nothing_is_still_named() -> None:
    """Dropping it would leave the sentence with nothing to be about."""
    said = attribution_sentence(
        facts(
            ret=-0.0006,
            mkt_component=-0.0001,
            sector_component=-0.0002,
            idio_component=-0.0003,
        )
    )
    assert said is not None
    assert "Tesla itself" in said
    assert "the wider market" not in said


def test_no_dominant_component_is_not_called_the_reason() -> None:
    said = attribution_sentence(
        facts(ret=-0.09, mkt_component=-0.03, sector_component=-0.03, idio_component=-0.03)
    )
    assert said is not None
    assert "The largest single piece" in said
    assert "Most of it" not in said


def test_a_missing_decomposition_is_stated_rather_than_skipped() -> None:
    said = attribution_sentence(
        facts(mkt_component=None, sector_component=None, idio_component=None)
    )
    assert said == "There is no market-versus-company breakdown for this day."


# --------------------------------------------------------------------------- #
# Peers
# --------------------------------------------------------------------------- #


def test_peers_moving_far_less_means_the_stock_moved_alone() -> None:
    said = peers_sentence(facts())
    assert said is not None and "moved largely on its own" in said


def test_peers_moving_with_it_means_the_group_moved() -> None:
    """The decomposition has to agree, so the sector carries this one: peers
    moving with a company-led move is the disagreement case below."""
    said = peers_sentence(facts(peer_comove=-0.13, sector_component=-0.086, idio_component=-0.036))
    assert said is not None and "whole group moved together" in said


def test_peers_moving_the_other_way_is_not_called_far_less() -> None:
    said = peers_sentence(facts(ret=0.085, ret_z=2.8, peer_comove=-0.002))
    assert said is not None and "opposite direction" in said


def test_a_flat_peer_average_is_not_printed_as_a_signed_zero() -> None:
    """ "up 0.0%" invites the reader to wonder what it means."""
    said = peers_sentence(facts(peer_comove=0.0001))
    assert said is not None and "barely moved" in said and "0.0%" not in said


def test_a_group_led_split_is_not_followed_by_moved_on_its_own() -> None:
    """NVDA, 2026-01-20: the sector carried 5.3 of the 4.4 points, and the
    next line used to say the stock had moved largely on its own."""
    said = peers_sentence(facts(**NVDA))
    assert said is not None
    assert "on its own" not in said
    assert "sits oddly with the sector-led split above" in said
    assert "far less than NVDA" in said
    assert not any(term in said.lower() for term in JARGON)


def test_a_market_led_split_disagreeing_with_the_peers_names_the_market() -> None:
    said = peers_sentence(
        facts(
            ticker="FCX",
            company="Freeport",
            ret=-0.091,
            mkt_component=-0.040,
            sector_component=-0.032,
            idio_component=-0.019,
            peer_comove=-0.005,
        )
    )
    assert said is not None
    assert "on its own" not in said
    assert "sits oddly with the market-led split above" in said


def test_peers_the_other_way_against_a_group_led_split_is_a_disagreement() -> None:
    """Opposite-direction peers are still "its own move" when the company's
    own part led; against a sector-led split they are a contradiction."""
    said = peers_sentence(facts(sector_component=-0.086, idio_component=-0.036, peer_comove=0.01))
    assert said is not None
    assert "own move" not in said
    assert "the opposite direction, which sits oddly with the sector-led split above" in said


def test_a_company_led_split_is_not_followed_by_the_whole_group_moving() -> None:
    said = peers_sentence(
        facts(
            ticker="AAPL",
            company="Apple",
            ret=-0.05,
            mkt_component=-0.005,
            sector_component=-0.010,
            idio_component=-0.035,
            peer_comove=-0.035,
        )
    )
    assert said is not None
    assert "whole group moved together" not in said
    assert "so the group moved with it even though the split above" in said
    assert "on Apple itself" in said
    # The company's own name carries the "pp" the jargon list bans, which is
    # why the provider-side list spells it "pp," / "pp." / "pp)".
    assert not any(term in said.lower().replace("apple", "") for term in JARGON)


def test_a_middling_peer_average_still_says_the_move_was_partly_shared() -> None:
    """Halfway peers claim nothing strong enough to disagree with."""
    for extra in ({}, {"sector_component": -0.086, "idio_component": -0.036}):
        said = peers_sentence(facts(peer_comove=-0.072, **extra))
        assert said is not None and "shared across the group" in said


def test_peers_without_a_decomposition_keep_their_own_verdict() -> None:
    said = peers_sentence(
        facts(mkt_component=None, sector_component=None, idio_component=None, peer_comove=-0.13)
    )
    assert said is not None and "whole group moved together" in said


def test_the_two_readings_of_one_move_never_contradict_each_other() -> None:
    """The pair of sentences is what the reader actually sees, two lines
    apart. Whichever way the split came out, the peers sentence may not assert
    the opposite verdict."""
    for extra in (
        NVDA,
        {"peer_comove": -0.13},
        {"sector_component": -0.086, "idio_component": -0.036, "peer_comove": -0.002},
    ):
        made = facts(**extra)
        attribution = attribution_sentence(made)
        peers = peers_sentence(made)
        assert attribution is not None and peers is not None
        # The verdict clause, before the breakdown: the group labels are the
        # only ones that begin with "the".
        verdict = attribution.split(":")[0].split(",")[0]
        if "the wider market" in verdict or "the rest of the sector" in verdict:
            assert "moved largely on its own" not in peers and "own move" not in peers
        else:
            assert "whole group moved together" not in peers
        assert not any(term in f"{attribution} {peers}".lower() for term in JARGON)


# --------------------------------------------------------------------------- #
# A whole move
# --------------------------------------------------------------------------- #


def test_a_move_reads_as_sentences_with_no_jargon() -> None:
    said = " ".join(summary_sentences(facts()))
    assert said.startswith("Tesla was down 14.5% on Thursday, 23 July 2026.")
    assert "within a day of the company's own earnings" in said
    assert not any(term in said.lower() for term in JARGON)


def test_scheduled_events_are_named_in_ordinary_words() -> None:
    said = " ".join(summary_sentences(facts(near_earnings=False, near_fomc=True, near_cpi=True)))
    assert "Federal Reserve rate decision" in said
    assert "inflation release" in said
    assert "FOMC" not in said and "CPI" not in said


# --------------------------------------------------------------------------- #
# A set of moves -- the synthesis
# --------------------------------------------------------------------------- #


def test_a_set_of_moves_gets_a_sentence_about_the_set() -> None:
    """The part a list of rows cannot do: whether these days have anything in
    common."""
    said = narrate_moves("TSLA", [move_dict(), move_dict(date="2026-06-05")], company="Tesla, Inc.")
    # Default ordering is by z, so the header says "most unusual", not
    # "biggest" -- the two rankings pick different days.
    assert said.startswith("Tesla (TSLA) -- the two most unusual moves in the data")
    assert "Taken together," in said
    assert "every one of them was driven by the company" in said


def test_headlines_are_bullets_rather_than_a_semicolon_run_on() -> None:
    said = narrate_moves("TSLA", [move_dict()], company="Tesla")
    assert "  Reported that day:" in said
    assert "    - Tesla stock stumbles after latest earnings report (CNBC)" in said


def test_one_move_is_singular() -> None:
    said = narrate_moves("TSLA", [move_dict()], company="Tesla", direction="down")
    assert "the most unusual fall in the data" in said


def test_a_mixed_set_makes_no_claim_about_the_set() -> None:
    """A hedge about a mixed bag is worse than saying nothing."""
    said = narrate_moves(
        "TSLA",
        [
            move_dict(routing="company", near_earnings=True),
            move_dict(date="2026-06-05", routing="macro", ret=0.05, near_earnings=False),
        ],
        company="Tesla",
    )
    assert "Taken together" not in said


def test_the_caveat_is_said_once_per_answer() -> None:
    said = narrate_moves("TSLA", [move_dict(), move_dict(date="2026-06-05")], company="Tesla")
    assert said.count(ATTRIBUTION_CAVEAT) == 1


def test_an_empty_set_says_so() -> None:
    assert narrate_moves("TSLA", []) == "No stored moves for TSLA."
    assert narrate_news("TSLA", []) == "No stored headlines for TSLA."


def test_news_is_grouped_under_the_day_it_was_published() -> None:
    said = narrate_news(
        "TSLA",
        [
            {"title": "A", "source": "CNBC", "published_at": "2026-07-23T12:00:00"},
            {"title": "B", "source": "WSJ", "published_at": "2026-07-23T15:00:00"},
        ],
    )
    assert said.count("Thursday, 23 July 2026") == 1
    assert "    - A (CNBC)" in said and "    - B (WSJ)" in said


# --------------------------------------------------------------------------- #
# Tolerating the query layer
# --------------------------------------------------------------------------- #


def test_a_move_with_nulls_still_narrates() -> None:
    """`move_to_dict` emits nulls for an unfitted day; none of them may raise."""
    said = narrate_moves("TSLA", [{"date": None, "ret": None, "ret_z": None}])
    assert "unknown date" in said


def test_small_counts_are_words_and_large_ones_digits() -> None:
    assert count_word(0) == "no"
    assert count_word(5) == "five"
    assert count_word(25) == "25"


def test_a_dominant_part_larger_than_the_move_is_not_said_as_a_share() -> None:
    """A component pushing the other way can leave the dominant part larger
    than the move it sits inside. "8.4 of the 7.4 points" is true and
    unreadable."""
    said = attribution_sentence(
        facts(
            ret=-0.074,
            mkt_component=0.010,
            sector_component=0.001,
            idio_component=-0.084,
        )
    )
    assert said is not None
    assert "8.4 of the 7.4" not in said
    assert "more than the 7.4-point move" in said
    assert "pushing the other way" in said


# --------------------------------------------------------------------------- #
# The header names the ranking the rows arrived in
# --------------------------------------------------------------------------- #


def test_a_z_ordered_list_is_not_called_the_biggest() -> None:
    """The bug this guards: a z-ordered list headed "the biggest falls" whose
    lead row is a 3.5% day sitting above a 7.4% one."""
    said = narrate_moves(
        "AAPL",
        [move_dict(), move_dict(date="2026-06-05")],
        company="Apple",
        direction="down",
        order="z",
    )
    assert "the two most unusual falls in the data" in said
    assert "biggest" not in said


def test_a_pct_ordered_list_is_called_the_biggest() -> None:
    said = narrate_moves(
        "AAPL",
        [move_dict(), move_dict(date="2026-06-05")],
        company="Apple",
        direction="down",
        order="pct",
    )
    assert "the two biggest falls in the data" in said
    assert "most unusual" not in said


def test_one_move_says_the_ranking_in_the_singular() -> None:
    for direction, kind in (("down", "fall"), ("up", "gain"), (None, "move")):
        unusual = narrate_moves(
            "AAPL", [move_dict()], company="Apple", direction=direction, order="z"
        )
        assert f"the most unusual {kind} in the data" in unusual

        biggest = narrate_moves(
            "AAPL", [move_dict()], company="Apple", direction=direction, order="pct"
        )
        assert f"the biggest {kind} in the data" in biggest


def test_the_plural_header_covers_every_direction() -> None:
    pair = [move_dict(), move_dict(date="2026-06-05")]
    for direction, kind in (("down", "falls"), ("up", "gains"), (None, "moves")):
        unusual = narrate_moves("AAPL", pair, company="Apple", direction=direction, order="z")
        assert f"the two most unusual {kind} in the data" in unusual

        biggest = narrate_moves("AAPL", pair, company="Apple", direction=direction, order="pct")
        assert f"the two biggest {kind} in the data" in biggest


def test_the_default_order_is_the_unusual_one() -> None:
    """`order` defaults to "z" because `MoveFilters` does."""
    assert "most unusual move" in narrate_moves("AAPL", [move_dict()], company="Apple")
