"""Tests for the phrasing layer.

`narrate` is the one place that decides how a quantity is *said*, so these are
mostly assertions about words. Two things are being pinned down: that the
jargon never reaches the reader, and that each sentence is still *true* of the
numbers it came from -- a readable explanation that misstates the
decomposition would be worse than the unreadable one it replaced.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any

from stock_moves.narrate import (
    ATTRIBUTION_CAVEAT,
    MoveFacts,
    attribution_sentence,
    count_word,
    geo_sentence,
    narrate_moves,
    narrate_news,
    peers_sentence,
    share_shift_sentence,
    short_company_name,
    size_phrase,
    summary_sentences,
    supply_chain_sentence,
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
# The v1.5 sentences -- share shift, supply chain, macro driver
# --------------------------------------------------------------------------- #


def shift(**overrides: Any) -> MoveFacts:
    """A company-routed move that `moves.sub_route` labelled `share_shift`."""
    base: dict[str, Any] = {
        "ticker": "NVDA",
        "company": "NVIDIA",
        "routing": "company",
        "sub_routing": "share_shift",
        "rival_comove": 0.031,
        "competitors": ("AMD",),
    }
    base.update(overrides)
    return facts(**base)


def geo(**overrides: Any) -> MoveFacts:
    """A macro-routed move whose driver is a country exposure."""
    base: dict[str, Any] = {
        "ticker": "AAPL",
        "company": "Apple",
        "day": date(2025, 4, 3),
        "ret": -0.092,
        "ret_z": -4.7,
        "routing": "macro",
        "sub_routing": "country:TW",
        "macro_driver": "country:TW",
        "macro_driver_component": -0.023,
        "countries": (("TW", 0.4),),
        "geo_events": (
            "2025-04-02 TW 3 headlines: Taipei braces for tariff decision",
            "2025-04-03 TW 14 headlines: Taiwan hit by new tariffs",
        ),
    }
    base.update(overrides)
    return facts(**base)


def test_a_share_shift_names_the_rival_and_which_way_it_went() -> None:
    assert share_shift_sentence(shift()) == "Share shift: AMD rose 3.1% the same day."


def test_several_rivals_are_named_two_at_a_time_and_then_counted() -> None:
    """Three symbols in a row is a list, not a sentence -- and the number is a
    mean, so it says so once it stops being about one company."""
    said = share_shift_sentence(shift(competitors=("AMD", "INTC", "AVGO", "MU")))
    assert said == "Share shift: AMD, INTC and two others rose 3.1% on average the same day."
    assert share_shift_sentence(shift(competitors=("AMD", "INTC", "AVGO"))) == (
        "Share shift: AMD, INTC and one other rose 3.1% on average the same day."
    )


def test_a_share_shift_with_no_stored_rivals_still_says_the_direction() -> None:
    """The edges are facts about the company and may not have been loaded; the
    verdict is still the move's."""
    assert share_shift_sentence(shift(competitors=())) == (
        "Share shift: rivals rose 3.1% on average the same day."
    )


def test_the_share_shift_sentence_is_absent_without_the_signal() -> None:
    # No sub-bucket: `moves.sub_route` is the only thing that decides whether
    # rivals moving the other way was large enough to mean anything.
    assert share_shift_sentence(shift(sub_routing=None)) is None
    assert share_shift_sentence(shift(sub_routing="supply_chain")) is None
    assert share_shift_sentence(shift(rival_comove=None)) is None
    assert share_shift_sentence(facts()) is None


def test_the_supply_chain_sentence_names_the_chain() -> None:
    made = facts(
        ticker="NVDA",
        company="NVIDIA",
        routing="company",
        sub_routing="supply_chain",
        chain_comove=-0.036,
        suppliers=("TSM",),
    )
    assert supply_chain_sentence(made) == "Supply chain: TSM fell 3.6% the same day."


def test_the_supply_chain_sentence_covers_customers_as_well_as_suppliers() -> None:
    made = facts(
        routing="company",
        sub_routing="supply_chain",
        chain_comove=-0.036,
        suppliers=("TSM",),
        customers=("DELL",),
    )
    said = supply_chain_sentence(made)
    assert said == "Supply chain: TSM and DELL fell 3.6% on average the same day."


def test_the_supply_chain_sentence_is_absent_without_the_signal() -> None:
    assert supply_chain_sentence(facts()) is None
    assert supply_chain_sentence(facts(sub_routing="supply_chain")) is None
    assert supply_chain_sentence(shift()) is None


def test_a_country_driver_says_the_exposure_the_etf_and_the_headlines() -> None:
    assert geo_sentence(geo()) == (
        "Macro, geopolitical, via Taiwan exposure: the Taiwan ETF moved the stock "
        "2.3% and 14 geopolitical headlines named Taiwan."
    )


def test_the_headline_clause_is_dropped_when_no_event_matches_that_day() -> None:
    """A country-day with no stored row is absence of evidence: the clause goes
    rather than being printed as a zero."""
    said = geo_sentence(geo(geo_events=("2025-04-02 TW 3 headlines: Taipei braces",)))
    assert said == (
        "Macro, geopolitical, via Taiwan exposure: the Taiwan ETF moved the stock 2.3%."
    )
    # The same day, a different country: still not this move's evidence.
    other = geo_sentence(geo(geo_events=("2025-04-03 CN 14 headlines: China retaliates",)))
    assert other is not None and "headlines" not in other


def test_a_commodity_driver_gets_one_short_sentence() -> None:
    made = facts(
        routing="macro",
        sub_routing="oil",
        macro_driver="oil",
        macro_driver_component=0.012,
    )
    assert geo_sentence(made) == "Macro, via oil: the oil proxy moved the stock 1.2%."


def test_a_driver_with_no_contribution_stops_at_the_name() -> None:
    made = facts(routing="macro", sub_routing="rates", macro_driver="rates")
    assert geo_sentence(made) == "Macro, via rates."


def test_a_macro_driver_on_a_company_day_is_not_narrated_as_macro() -> None:
    """`macro_driver` is computed for every day whatever the routing, so a
    company move with a jumpy oil price must not read as an oil day."""
    made = facts(routing="company", macro_driver="oil", macro_driver_component=0.012)
    assert geo_sentence(made) is None
    assert geo_sentence(facts()) is None


def test_an_unmapped_country_code_is_its_own_name() -> None:
    made = facts(
        routing="macro",
        sub_routing="country:ZZ",
        macro_driver="country:ZZ",
        macro_driver_component=-0.01,
    )
    said = geo_sentence(made)
    assert said is not None and said.startswith("Macro, geopolitical, via ZZ exposure:")


def test_the_new_sentences_follow_the_split_and_carry_no_jargon() -> None:
    said = summary_sentences(shift(day=date(2026, 7, 23)))
    attribution = next(i for i, s in enumerate(said) if "points" in s or "breakdown" in s)
    assert said[attribution + 1].startswith("Share shift:")
    assert not any(term in " ".join(said).lower() for term in JARGON)
    assert "*" not in " ".join(said) and "#" not in " ".join(said)


def test_a_move_with_no_new_signals_reads_exactly_as_it_did_in_v1() -> None:
    """Every new sentence is absent unless its signal is stored, so the
    keyless, edgeless install is unchanged."""
    said = summary_sentences(facts())
    assert not any(
        sentence.startswith(("Share shift:", "Supply chain:", "Macro,")) for sentence in said
    )
    assert said == summary_sentences(
        facts(sub_routing=None, macro_driver=None, rival_comove=None, chain_comove=None)
    )


def test_from_context_carries_the_relationship_fields() -> None:
    """Duck-typed off the provider layer's `MoveContext`, the same way the v1
    fields are."""
    context = SimpleNamespace(
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        date=date(2025, 4, 16),
        ret=-0.069,
        ret_z=-1.2,
        routing="company",
        sub_routing="share_shift",
        macro_driver="country:TW",
        macro_driver_component=-0.023,
        rival_comove=0.031,
        chain_comove=-0.036,
        competitors=["amd", "intc"],
        suppliers=["TSM"],
        customers=[],
        countries=[("tw", 0.4)],
        geo_events=["2025-04-16 TW 9 headlines: export licence"],
    )

    made = MoveFacts.from_context(context)

    assert made.company == "NVIDIA"
    assert made.sub_routing == "share_shift"
    assert made.macro_driver_component == -0.023
    assert made.rival_comove == 0.031
    assert made.chain_comove == -0.036
    # Carried as given; the sentence is the layer that upper-cases a symbol.
    assert made.competitors == ("amd", "intc")
    assert share_shift_sentence(made) is not None
    assert "AMD and INTC" in str(share_shift_sentence(made))
    assert made.suppliers == ("TSM",)
    assert made.customers == ()
    assert made.countries == (("TW", 0.4),)
    assert made.geo_events == ("2025-04-16 TW 9 headlines: export licence",)


def test_a_v1_row_leaves_the_new_fields_at_their_defaults() -> None:
    made = MoveFacts.from_context(
        SimpleNamespace(ticker="TSLA", company_name="Tesla, Inc.", date=None, ret=-0.1, ret_z=-2.0)
    )
    assert made.sub_routing is None
    assert made.competitors == () and made.countries == () and made.geo_events == ()


def test_the_five_dated_columns_survive_a_move_dict() -> None:
    """`move_to_dict` carries the five per-day columns; the edge lists are
    facts about the company and are not in it, so the sentence falls back to
    the unnamed subject rather than dropping."""
    made = MoveFacts.from_mapping(
        move_dict(sub_routing="share_shift", rival_comove=0.031), "NVDA", "NVIDIA"
    )
    assert made.sub_routing == "share_shift"
    assert share_shift_sentence(made) == "Share shift: rivals rose 3.1% on average the same day."


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
