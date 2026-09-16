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
    assert "pushing the other way" in said


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
    said = peers_sentence(facts(peer_comove=-0.13))
    assert said is not None and "whole group moved together" in said


def test_peers_moving_the_other_way_is_not_called_far_less() -> None:
    said = peers_sentence(facts(ret=0.085, ret_z=2.8, peer_comove=-0.002))
    assert said is not None and "opposite direction" in said


def test_a_flat_peer_average_is_not_printed_as_a_signed_zero() -> None:
    """ "up 0.0%" invites the reader to wonder what it means."""
    said = peers_sentence(facts(peer_comove=0.0001))
    assert said is not None and "barely moved" in said and "0.0%" not in said


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
    assert said.startswith("Tesla (TSLA) -- the two biggest moves in the data")
    assert "Taken together," in said
    assert "every one of them was driven by the company" in said


def test_headlines_are_bullets_rather_than_a_semicolon_run_on() -> None:
    said = narrate_moves("TSLA", [move_dict()], company="Tesla")
    assert "  Reported that day:" in said
    assert "    - Tesla stock stumbles after latest earnings report (CNBC)" in said


def test_one_move_is_singular() -> None:
    said = narrate_moves("TSLA", [move_dict()], company="Tesla", direction="down")
    assert "the biggest fall in the data" in said


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
