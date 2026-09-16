"""Tests for `suggest_relations` on all three providers (v1.5 decision 3).

No network and no API key: each keyed provider gets the same fake-client
pattern its own suite uses -- a queue of pre-canned results, an `Exception` in
the queue meaning "the call failed" -- so the two vendors are held to one
contract in one file. The cleaning itself (upper-casing, deduping, dropping the
company's own ticker, clamping and rescaling the country weights) lives in
`prompts.py` and is shared, and these tests assert both providers actually get
the shared behaviour rather than a copy of it.

The keyless provider is here too, because "empty in all four relations" is not
a stub: it is the decision that keeps the geopolitical gate shut without a key.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from stock_moves.providers.anthropic import AnthropicProvider
from stock_moves.providers.base import Relations
from stock_moves.providers.heuristic import HeuristicProvider
from stock_moves.providers.openai import MAX_TOKENS_SHORT, OpenAIProvider
from stock_moves.providers.prompts import (
    MAX_COUNTRIES,
    MAX_PEERS,
    MAX_SUPPLY_CHAIN,
    RELATIONS_SYSTEM,
    RelationsOut,
)

# --------------------------------------------------------------------------- #
# Fakes: one per vendor, each mirroring that vendor's own suite
# --------------------------------------------------------------------------- #


class FakeOpenAICompletions:
    def __init__(self) -> None:
        self.parse_queue: list[Any] = []
        self.parse_calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> Any:
        self.parse_calls.append(kwargs)
        item = self.parse_queue.pop(0)
        if isinstance(item, Exception):
            raise item
        message = (
            SimpleNamespace(parsed=None, refusal=item, content=None)
            if isinstance(item, str)
            else SimpleNamespace(parsed=item, refusal=None, content=None)
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeOpenAICompletions())

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.chat.completions.parse_calls

    def queue(self, item: Any) -> None:
        self.chat.completions.parse_queue.append(item)


class FakeAnthropicMessages:
    def __init__(self) -> None:
        self.parse_queue: list[Any] = []
        self.parse_calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> Any:
        self.parse_calls.append(kwargs)
        item = self.parse_queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(parsed_output=item)


class FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = FakeAnthropicMessages()

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.messages.parse_calls

    def queue(self, item: Any) -> None:
        self.messages.parse_queue.append(item)


#: `(vendor label, client factory, provider factory)`. Every behavioural test
#: below runs against both, which is the point of sharing `prompts.py`: the two
#: can only differ in transport.
VENDORS = [
    pytest.param(
        FakeOpenAIClient,
        lambda client: OpenAIProvider(api_key="test-key", client=client),
        id="openai",
    ),
    pytest.param(
        FakeAnthropicClient,
        lambda client: AnthropicProvider(api_key="test-key", client=client),
        id="anthropic",
    ),
]


def relations_out(**overrides: Any) -> RelationsOut:
    base: dict[str, Any] = {
        "competitors": ["nvda", "INTC"],
        "suppliers": ["TSM"],
        "customers": ["MSFT"],
        "countries": [{"country": "tw", "weight": 0.4}, {"country": "CN", "weight": 0.2}],
    }
    base.update(overrides)
    return RelationsOut.model_validate(base)


def ask(provider: Any) -> Relations:
    return provider.suggest_relations("AMD", "Advanced Micro Devices Inc", "Technology", "Semis")


# --------------------------------------------------------------------------- #
# The parse
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("make_client", "make_provider"), VENDORS)
def test_the_four_relations_come_back_cleaned(make_client: Any, make_provider: Any) -> None:
    client = make_client()
    client.queue(relations_out())

    result = ask(make_provider(client))

    assert result == Relations(
        competitors=("NVDA", "INTC"),
        suppliers=("TSM",),
        customers=("MSFT",),
        countries=(("TW", 0.4), ("CN", 0.2)),
    )
    assert len(client.calls) == 1


@pytest.mark.parametrize(("make_client", "make_provider"), VENDORS)
def test_tickers_are_upper_cased_deduped_and_stripped_of_self(
    make_client: Any, make_provider: Any
) -> None:
    client = make_client()
    client.queue(
        relations_out(
            competitors=["nvda", " NVDA ", "AMD", "", "intc"],
            suppliers=["amd", "tsm"],
            customers=["MU", "mu"],
        )
    )

    result = ask(make_provider(client))

    assert result.competitors == ("NVDA", "INTC")
    assert result.suppliers == ("TSM",)  # the company's own ticker is dropped here too
    assert result.customers == ("MU",)


@pytest.mark.parametrize(("make_client", "make_provider"), VENDORS)
def test_each_list_is_cut_to_its_own_cap(make_client: Any, make_provider: Any) -> None:
    """Competitors get a longer list than the supply chain: a company has many
    rivals and only a handful of public counterparties worth naming."""
    client = make_client()
    client.queue(
        relations_out(
            competitors=[f"C{i}" for i in range(MAX_PEERS + 3)],
            suppliers=[f"S{i}" for i in range(MAX_SUPPLY_CHAIN + 3)],
            customers=[f"K{i}" for i in range(MAX_SUPPLY_CHAIN + 3)],
            countries=[
                {"country": code, "weight": 0.01}
                for code in ("TW", "CN", "JP", "KR", "DE", "GB", "MX")
            ],
        )
    )

    result = ask(make_provider(client))

    assert len(result.competitors) == MAX_PEERS == 6
    assert len(result.suppliers) == MAX_SUPPLY_CHAIN == 4
    assert len(result.customers) == MAX_SUPPLY_CHAIN == 4
    assert len(result.countries) == MAX_COUNTRIES == 5
    assert [code for code, _ in result.countries] == ["TW", "CN", "JP", "KR", "DE"]


@pytest.mark.parametrize(("make_client", "make_provider"), VENDORS)
def test_a_code_that_is_not_alpha_2_is_dropped(make_client: Any, make_provider: Any) -> None:
    """The country-ETF lookup downstream is keyed on alpha-2, so "Taiwan" and
    "USA" are not near-misses to be repaired -- they are unusable."""
    client = make_client()
    client.queue(
        relations_out(
            countries=[
                {"country": "Taiwan", "weight": 0.3},
                {"country": "USA", "weight": 0.2},
                {"country": "de", "weight": 0.1},
                {"country": "", "weight": 0.1},
            ]
        )
    )

    assert ask(make_provider(client)).countries == (("DE", 0.1),)


@pytest.mark.parametrize(("make_client", "make_provider"), VENDORS)
def test_a_repeated_country_keeps_only_its_first_weight(
    make_client: Any, make_provider: Any
) -> None:
    client = make_client()
    client.queue(
        relations_out(
            countries=[
                {"country": "CN", "weight": 0.3},
                {"country": "cn", "weight": 0.9},
            ]
        )
    )

    assert ask(make_provider(client)).countries == (("CN", 0.3),)


@pytest.mark.parametrize(("make_client", "make_provider"), VENDORS)
def test_an_answer_with_nothing_in_it_is_a_valid_answer(
    make_client: Any, make_provider: Any
) -> None:
    """Empty is what the prompt asks for rather than padding, so it must not
    be mistaken for a failure anywhere in the cleaning path."""
    client = make_client()
    client.queue(RelationsOut.model_validate({}))

    assert ask(make_provider(client)) == Relations((), (), (), ())


# --------------------------------------------------------------------------- #
# The clamp on weights
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("make_client", "make_provider"), VENDORS)
def test_a_single_weight_is_clamped_into_the_unit_interval(
    make_client: Any, make_provider: Any
) -> None:
    client = make_client()
    client.queue(relations_out(countries=[{"country": "CN", "weight": 4.0}]))

    assert ask(make_provider(client)).countries == (("CN", 1.0),)


@pytest.mark.parametrize(("make_client", "make_provider"), VENDORS)
def test_a_negative_weight_is_clamped_to_zero(make_client: Any, make_provider: Any) -> None:
    client = make_client()
    client.queue(relations_out(countries=[{"country": "CN", "weight": -0.5}]))

    assert ask(make_provider(client)).countries == (("CN", 0.0),)


@pytest.mark.parametrize(("make_client", "make_provider"), VENDORS)
def test_weights_over_one_in_total_are_scaled_down_proportionally(
    make_client: Any, make_provider: Any
) -> None:
    """Scaled, not truncated: a model that says twice as much China as Taiwan
    still says twice as much afterwards, and neither country is silently lost
    -- losing one is exactly how the geo gate would stop firing."""
    client = make_client()
    client.queue(
        relations_out(
            countries=[
                {"country": "CN", "weight": 0.8},
                {"country": "TW", "weight": 0.4},
                {"country": "DE", "weight": 0.4},
            ]
        )
    )

    countries = ask(make_provider(client)).countries

    assert [code for code, _ in countries] == ["CN", "TW", "DE"]
    assert sum(weight for _, weight in countries) == pytest.approx(1.0)
    assert countries[0][1] == pytest.approx(0.5)
    assert countries[1][1] == pytest.approx(0.25)


@pytest.mark.parametrize(("make_client", "make_provider"), VENDORS)
def test_weights_that_already_sum_below_one_are_left_alone(
    make_client: Any, make_provider: Any
) -> None:
    """They are shares of the whole company, not of each other, so a mostly
    domestic name's weights are meant to be small."""
    client = make_client()
    client.queue(
        relations_out(
            countries=[{"country": "CN", "weight": 0.1}, {"country": "MX", "weight": 0.05}]
        )
    )

    assert ask(make_provider(client)).countries == (("CN", 0.1), ("MX", 0.05))


@pytest.mark.parametrize(("make_client", "make_provider"), VENDORS)
def test_the_clamp_runs_before_the_rescale(make_client: Any, make_provider: Any) -> None:
    """One absurd weight must not be allowed to shrink every honest one: 5.0 is
    clamped to 1.0 first, so the rescale divides by 1.2, not by 5.2."""
    client = make_client()
    client.queue(
        relations_out(
            countries=[{"country": "CN", "weight": 5.0}, {"country": "TW", "weight": 0.2}]
        )
    )

    countries = ask(make_provider(client)).countries

    assert countries[0][1] == pytest.approx(1 / 1.2)
    assert countries[1][1] == pytest.approx(0.2 / 1.2)


# --------------------------------------------------------------------------- #
# The call itself
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("make_client", "make_provider"), VENDORS)
def test_the_relations_system_prompt_and_the_short_budget_are_used(
    make_client: Any, make_provider: Any
) -> None:
    client = make_client()
    client.queue(relations_out())
    ask(make_provider(client))

    call = client.calls[0]
    sent = call.get("system") or call["messages"][0]["content"]
    assert sent == RELATIONS_SYSTEM
    budget = call.get("max_tokens") or call["max_completion_tokens"]
    assert budget == MAX_TOKENS_SHORT

    prompt = call["messages"][-1]["content"]
    assert "AMD" in prompt and "Advanced Micro Devices Inc" in prompt


def test_the_system_prompt_states_every_cap_and_the_public_only_rule() -> None:
    """The caps are asked for in words and enforced in code; if the two drift,
    the model is being told one thing and judged by another."""
    assert f"up to {MAX_PEERS} direct competitors" in RELATIONS_SYSTEM
    assert f"up to {MAX_SUPPLY_CHAIN} suppliers" in RELATIONS_SYSTEM
    assert f"up to {MAX_COUNTRIES} countries" in RELATIONS_SYSTEM
    assert "ISO-3166 alpha-2" in RELATIONS_SYSTEM
    assert "sum to at most 1" in RELATIONS_SYSTEM
    assert "Public companies only" in RELATIONS_SYSTEM
    assert "never the company's own ticker" in RELATIONS_SYSTEM


# --------------------------------------------------------------------------- #
# The degrade path
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("make_client", "make_provider"), VENDORS)
def test_a_failed_call_returns_the_empty_relations_and_logs(
    make_client: Any, make_provider: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """The same degrade as `suggest_peers`: edges are optional, and an ingest
    that loses this call stores no edges rather than failing."""
    client = make_client()
    client.queue(RuntimeError("401 invalid api key"))

    with caplog.at_level(logging.WARNING):
        result = ask(make_provider(client))

    assert result == Relations((), (), (), ())
    assert "suggest_relations failed" in caplog.text
    assert "AMD" in caplog.text


@pytest.mark.parametrize(("make_client", "make_provider"), VENDORS)
def test_no_parsed_output_degrades_like_a_transport_failure(
    make_client: Any, make_provider: Any
) -> None:
    client = make_client()
    client.queue(None)  # both fakes turn this into "no parsed output"

    assert ask(make_provider(client)) == Relations((), (), (), ())


def test_an_openai_refusal_degrades_rather_than_returning_half_an_answer() -> None:
    """Specific to the OpenAI transport: a refusal arrives as a field on the
    message, not as an exception, and must take the same path."""
    client = FakeOpenAIClient()
    client.queue("I cannot help with that")

    provider = OpenAIProvider(api_key="test-key", client=client)
    assert ask(provider) == Relations((), (), (), ())


@pytest.mark.parametrize(("make_client", "make_provider"), VENDORS)
def test_an_unusable_payload_degrades_rather_than_raising(
    make_client: Any, make_provider: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """A parsed object whose fields are not what the schema promised (a bare
    string where a list belongs) is a failed call, not an exception escaping
    into the ingest."""
    client = make_client()
    client.queue(SimpleNamespace(competitors=1, suppliers=[], customers=[], countries=[]))

    with caplog.at_level(logging.WARNING):
        result = ask(make_provider(client))

    assert result == Relations((), (), (), ())
    assert "unusable relations" in caplog.text


# --------------------------------------------------------------------------- #
# The keyless provider
# --------------------------------------------------------------------------- #


def test_the_heuristic_provider_returns_nothing_in_all_four_relations() -> None:
    """v1.5 decision 3. The ETF fallback for competitors is the ontology's job;
    suppliers, customers and countries have no keyless source at all, so
    without a key there are no country edges and the geo gate cannot open."""
    result = HeuristicProvider().suggest_relations("AMD", "Advanced Micro Devices", "Tech", None)

    assert result == Relations(competitors=(), suppliers=(), customers=(), countries=())


def test_the_heuristic_relations_are_immutable_and_typed() -> None:
    result = HeuristicProvider().suggest_relations("AMD", "Advanced Micro Devices", None, None)

    assert isinstance(result, Relations)
    with pytest.raises(AttributeError):
        result.competitors = ("NVDA",)  # type: ignore[misc]
