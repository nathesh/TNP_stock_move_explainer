"""Turning the decomposition into English (DESIGN section 4a).

Every number this app stores is meaningful to someone who already knows what a
z-score and a factor loading are. The reader of a stock-move explanation is
usually not that person, so this module is the one place that decides how a
quantity is *said*: a z-score becomes "about four times the size of a typical
day", a component in percentage points becomes "8.6 of those 14.5 points were
specific to Tesla".

It is deliberately dependency-free -- no database, no settings, no provider
imports -- for the same reason `providers/base.py` is: the phrasing is the part
most likely to be tested, tuned and argued about, and none of that should need
a config file. Both consumers feed it the same `MoveFacts`:

- `HeuristicProvider` renders it directly, which is what a keyless install and
  any failed model call actually show the user;
- the keyed providers put it in the prompt, so the model is handed English
  rather than being asked to translate percentage points itself.

The output is plain text with real newlines. The chat UI sets `white-space:
pre-wrap` and assigns to `textContent`, so line breaks and indentation survive
and markdown would show up as literal asterisks.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

__all__ = [
    "ATTRIBUTION_CAVEAT",
    "MoveFacts",
    "count_word",
    "headline_lines",
    "move_paragraph",
    "narrate_moves",
    "narrate_news",
    "narrate_one_move",
    "narrate_relations",
    "plain_day",
    "plain_move",
    "short_company_name",
    "size_phrase",
    "summary_sentences",
]

ATTRIBUTION_CAVEAT = (
    "These are attributions from same-day news, not proven causes: they say "
    "what was being reported when the stock moved."
)
"""The framing from DESIGN section 0, said once per answer rather than per move.
A reader who takes "why" literally will over-read any of this."""

#: Below this, a "move" is not really bigger than the stock's ordinary noise and
#: saying "N times a typical day" would overstate a multiple near 1.
_UNREMARKABLE_Z = 1.5

#: A component has to carry this share of the move before it is called the
#: reason for it, rather than merely the largest of three similar parts.
_DOMINANT_SHARE = 0.5

#: Peers moving less than this fraction of the stock's move means the stock
#: moved alone; more than the upper bound means the whole group moved.
_PEERS_ALONE = 0.4
_PEERS_TOGETHER = 0.6

#: How many related tickers a sentence names before it counts the rest. Three
#: symbols in a row is already a list rather than a sentence, and the point of
#: the share-shift line is *who* the money went to, not the full roster.
_MAX_NAMED = 2

#: The two company-side sub-buckets and the four factor proxies of v1.5
#: decision 4. Held as literals rather than imported from `moves` because this
#: module imports nothing from the rest of the app; the strings are the storage
#: contract either way.
_SUB_SHARE_SHIFT = "share_shift"
_SUB_SUPPLY_CHAIN = "supply_chain"
_COUNTRY_PREFIX = "country:"
_FACTOR_DRIVERS: frozenset[str] = frozenset({"oil", "dollar", "rates", "gold"})

#: A rendered `geo_events` line: "2025-04-03 CN 14 headlines: <title>". Parsed
#: rather than re-queried because `narrate` never touches the database — the
#: caller has already turned the rows into these lines.
_GEO_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2})\s+([A-Za-z]{2})\s+(\d+)\s+headlines")

#: The five relations an edge can carry (v1.5 decision 10), in the order a
#: reader wants them: who the company competes with first, because that is the
#: question people actually ask, and the fitted factor betas last, because they
#: are the least concrete thing on the list.
_RELATION_ORDER: tuple[str, ...] = (
    "competitor",
    "supplier",
    "customer",
    "country",
    "factor",
)

#: How a factor is said. Only the dollar takes an article in English, and
#: "rates" is already the plain word for the thing the proxy tracks.
_FACTOR_LABELS: dict[str, str] = {
    "oil": "oil",
    "dollar": "the dollar",
    "rates": "rates",
    "gold": "gold",
}

#: Below this absolute beta, a one-standard-deviation move in the proxy shifts
#: the stock by less than a fifth of one of its own: true, and not worth a
#: reader's attention. Said as "barely registers" rather than dropped, because
#: "we fitted it and it came out near zero" is itself an answer.
_FACTOR_NOTABLE = 0.2

#: Where an edge came from, said in English. The difference between a number
#: fitted from prices and a name a language model volunteered is the whole
#: reason `source` is stored, so it is said out loud rather than implied.
_SOURCE_PHRASES: dict[str, str] = {
    "etf_holdings": "from the sector ETF's largest holdings",
    "model": "as suggested by the model",
    "prices": "fitted from prices",
}

_NUMBER_WORDS: dict[int, str] = {
    2: "twice",
    3: "three times",
    4: "four times",
    5: "five times",
    6: "six times",
}

_COUNT_WORDS: tuple[str, ...] = (
    "no",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
)

#: Legal suffixes, and the punctuation that leads them. "Tesla, Inc. itself"
#: is how a filing refers to the company; "Tesla itself" is how a person does.
_LEGAL_SUFFIX_WORDS: frozenset[str] = frozenset(
    {
        "inc",
        "inc.",
        "incorporated",
        "corp",
        "corp.",
        "corporation",
        "co",
        "co.",
        "company",
        "ltd",
        "ltd.",
        "limited",
        "plc",
        "llc",
        "lp",
        "holdings",
        "holding",
        "group",
        "sa",
        "nv",
        "ag",
    }
)


def short_company_name(name: str, ticker: str = "") -> str:
    """`Tesla, Inc.` -> `Tesla`; `Alphabet Inc. Class A` -> `Alphabet`.

    Only trailing suffix words are dropped, so `Ford Motor` keeps `Motor` and a
    name that is *all* suffix (`Holdings Ltd`) falls back to the ticker rather
    than to an empty string.
    """
    words = name.replace(",", " ").split()
    suffixes = {suffix.strip(".") for suffix in _LEGAL_SUFFIX_WORDS}
    # "Class A" / "Series B" share-class tails carry no meaning in prose, and
    # they sit *outside* the legal suffix ("Alphabet Inc. Class A"), so they
    # have to come off before the suffix loop can reach what they hide.
    if len(words) >= 2 and words[-2].lower() in {"class", "series"} and len(words[-1]) <= 2:
        words = words[:-2]
    while words and words[-1].lower().strip(".") in suffixes:
        words.pop()
    return " ".join(words) or (ticker or name)


def count_word(n: int) -> str:
    """Small counts as words, larger ones as digits -- ordinary prose style."""
    return _COUNT_WORDS[n] if 0 <= n < len(_COUNT_WORDS) else str(n)


# --------------------------------------------------------------------------- #
# Facts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MoveFacts:
    """The subset of a move this module knows how to say out loud.

    A separate shape from `MoveContext` on purpose: the two callers hold
    different objects (a provider holds the dataclass, the chat renderer holds
    the query layer's dict) and neither should have to convert to the other's.
    """

    ticker: str
    company: str
    day: date | None
    ret: float
    ret_z: float
    routing: str
    near_earnings: bool = False
    near_fomc: bool = False
    near_cpi: bool = False
    mkt_component: float | None = None
    sector_component: float | None = None
    idio_component: float | None = None
    peer_comove: float | None = None

    # --- v1.5: the relationship layer -------------------------------------- #
    # The signals `sub_routing` is built from, plus the names needed to say
    # them out loud. Every one is defaulted, so a caller that knows none of
    # them -- a v1 row, or the chat renderer's dict -- keeps working and simply
    # renders the v1 sentences.

    #: "share_shift" | "supply_chain" | "oil" | "dollar" | "rates" | "gold" |
    #: "country:XX", or None when no rule fired (v1.5 decision 6).
    sub_routing: str | None = None
    macro_driver: str | None = None
    #: The driver's contribution to the day's return, as a return.
    macro_driver_component: float | None = None
    #: Same-day co-movement of the `competitor` edges, and of the
    #: `supplier`/`customer` edges.
    rival_comove: float | None = None
    chain_comove: float | None = None
    competitors: tuple[str, ...] = ()
    suppliers: tuple[str, ...] = ()
    customers: tuple[str, ...] = ()
    countries: tuple[tuple[str, float], ...] = ()
    #: Rendered geo-event lines for the move's window, as
    #: `news.geo.geo_event_lines` writes them.
    geo_events: tuple[str, ...] = ()

    @classmethod
    def from_context(cls, move: Any) -> MoveFacts:
        """Build from a provider's `MoveContext`.

        Duck-typed by `getattr`, the same idiom `base.py` uses, so this module
        keeps its promise not to import the provider layer.
        """
        return cls(
            ticker=str(getattr(move, "ticker", "") or ""),
            company=short_company_name(
                str(getattr(move, "company_name", "") or ""),
                str(getattr(move, "ticker", "") or ""),
            ),
            day=_as_date(getattr(move, "date", None)),
            ret=_as_float(getattr(move, "ret", None)) or 0.0,
            ret_z=_as_float(getattr(move, "ret_z", None)) or 0.0,
            routing=str(getattr(move, "routing", "") or "company"),
            near_earnings=bool(getattr(move, "near_earnings", False)),
            near_fomc=bool(getattr(move, "near_fomc", False)),
            near_cpi=bool(getattr(move, "near_cpi", False)),
            mkt_component=_as_float(getattr(move, "mkt_component", None)),
            sector_component=_as_float(getattr(move, "sector_component", None)),
            idio_component=_as_float(getattr(move, "idio_component", None)),
            peer_comove=_as_float(getattr(move, "peer_comove", None)),
            sub_routing=_as_str(getattr(move, "sub_routing", None)),
            macro_driver=_as_str(getattr(move, "macro_driver", None)),
            macro_driver_component=_as_float(getattr(move, "macro_driver_component", None)),
            rival_comove=_as_float(getattr(move, "rival_comove", None)),
            chain_comove=_as_float(getattr(move, "chain_comove", None)),
            competitors=_as_str_tuple(getattr(move, "competitors", None)),
            suppliers=_as_str_tuple(getattr(move, "suppliers", None)),
            customers=_as_str_tuple(getattr(move, "customers", None)),
            countries=_as_weighted_tuple(getattr(move, "countries", None)),
            geo_events=_as_str_tuple(getattr(move, "geo_events", None)),
        )

    @classmethod
    def from_mapping(cls, move: Mapping[str, Any], ticker: str, company: str = "") -> MoveFacts:
        """Build from the query layer's `move_to_dict` output."""
        return cls(
            ticker=ticker,
            company=short_company_name(company or ticker, ticker),
            day=_as_date(move.get("date")),
            ret=_as_float(move.get("ret")) or 0.0,
            ret_z=_as_float(move.get("ret_z")) or 0.0,
            routing=str(move.get("routing") or "company"),
            near_earnings=bool(move.get("near_earnings")),
            near_fomc=bool(move.get("near_fomc")),
            near_cpi=bool(move.get("near_cpi")),
            mkt_component=_as_float(move.get("mkt_component")),
            sector_component=_as_float(move.get("sector_component")),
            idio_component=_as_float(move.get("idio_component")),
            peer_comove=_as_float(move.get("peer_comove")),
            # The five v1.5 columns `move_to_dict` carries (decision 10). The
            # edge lists are not in that dict -- they are facts about the
            # company, not about the day -- so a move rendered from a mapping
            # says "rivals" where one rendered from a context names them.
            sub_routing=_as_str(move.get("sub_routing")),
            macro_driver=_as_str(move.get("macro_driver")),
            macro_driver_component=_as_float(move.get("macro_driver_component")),
            rival_comove=_as_float(move.get("rival_comove")),
            chain_comove=_as_float(move.get("chain_comove")),
        )


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str | None:
    """A non-empty string, or None -- so `""` and absent read the same."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    """A sequence of strings as a tuple; `()` for absent, and for a bare string
    (one string is not a list of tickers)."""
    if value is None or isinstance(value, (str, bytes)):
        return ()
    try:
        return tuple(str(item).strip() for item in value if str(item).strip())
    except TypeError:
        return ()


def _as_weighted_tuple(value: Any) -> tuple[tuple[str, float], ...]:
    """`(code, weight)` pairs, skipping anything that is not such a pair."""
    if value is None or isinstance(value, (str, bytes)):
        return ()
    try:
        items = list(value)
    except TypeError:
        return ()
    pairs: list[tuple[str, float]] = []
    for item in items:
        if isinstance(item, (str, bytes)):
            continue
        try:
            code, weight = item
        except (TypeError, ValueError):
            continue
        number = _as_float(weight)
        if number is None:
            continue
        pairs.append((str(code).strip().upper(), number))
    return tuple(pairs)


def _join(items: Sequence[str]) -> str:
    """`["a", "b", "c"]` -> `a, b and c`. Prose, so no serial comma."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


# --------------------------------------------------------------------------- #
# Saying one quantity
# --------------------------------------------------------------------------- #


def plain_day(day: date | None) -> str:
    """`2026-07-23` -> `Thursday, 23 July 2026`.

    The weekday is not decoration: "it was a Friday" is often the first thing a
    reader notices about a cluster of moves.
    """
    return "an unknown date" if day is None else day.strftime("%A, %-d %B %Y")


def plain_move(ret: float) -> str:
    """`-0.145` -> `down 14.5%`."""
    return f"{'down' if ret < 0 else 'up'} {abs(ret):.1%}"


def size_phrase(ret_z: float, ticker: str) -> str:
    """The z-score as a multiple of an ordinary day.

    `ret_z` is the return over a trailing 20-day volatility, so "N times a
    typical day" is what it literally measures -- the one piece of jargon here
    that translates without hedging.
    """
    magnitude = abs(ret_z)
    if magnitude < _UNREMARKABLE_Z:
        return f"That is close to an ordinary day's movement for {ticker}."
    rounded = round(magnitude)
    word = _NUMBER_WORDS.get(rounded)
    multiple = word if word else f"about {magnitude:.0f} times"
    lead = "roughly " if word else ""
    return f"That is {lead}{multiple} the size of a typical day for {ticker}."


def _components(facts: MoveFacts) -> list[tuple[str, str, float]]:
    """The components that exist, as `(kind, label, value)`, largest first.

    `kind` is the machine-readable half -- prose reads "the rest of the sector",
    code asks whether the company's own part won.
    """
    labelled = (
        ("market", "the wider market", facts.mkt_component),
        ("sector", "the rest of the sector", facts.sector_component),
        ("company", f"{facts.company} itself", facts.idio_component),
    )
    present = [(kind, label, value) for kind, label, value in labelled if value is not None]
    present.sort(key=lambda item: abs(item[2]), reverse=True)
    return present


def _dominant(facts: MoveFacts) -> tuple[str, str, float] | None:
    """The largest component by size, sign ignored, or `None` if there is no
    usable decomposition.

    One function, two callers, on purpose: the attribution sentence and the
    peers sentence sit two lines apart in the output, and they used to reach
    opposite conclusions about which part of the move was the big one because
    each worked it out its own way.
    """
    present = _components(facts)
    if not present or sum(abs(value) for _, _, value in present) <= 0:
        return None
    return present[0]


def attribution_sentence(facts: MoveFacts) -> str | None:
    """Split the move into market, sector and company in points of the move.

    The components are `beta * factor return`, so they sum to the day's return
    and can be said as "8.6 of those 14.5 points" -- a share of the move the
    reader can check by adding up. A component whose sign opposes the move is
    called out as pushing the other way, because silently reporting its size
    would read as if it had contributed.
    """
    # Said rather than skipped: a reader who is told how large the move was and
    # then hears nothing about what drove it should know the split is missing,
    # not be left to assume it was unremarkable.
    absent = "There is no market-versus-company breakdown for this day."
    top = _dominant(facts)
    if top is None:
        return absent

    present = _components(facts)
    total = sum(abs(value) for _, _, value in present)
    _, top_label, top_value = top
    points = abs(facts.ret) * 100

    # The verdict already names the dominant component, so the breakdown that
    # follows it must not name it again ("Most of it was Tesla itself: 8.6 came
    # from Tesla itself" is what naming it twice reads like).
    # The two directions are grouped so "pushing the other way" is said once
    # for however many components did it, not once each.
    with_move: list[tuple[str, float]] = []
    against_move: list[tuple[str, float]] = []
    for _kind, label, value in present[1:]:
        component_points = abs(value) * 100
        # "and 0.0 from the rest of the sector" is noise: a part that rounds to
        # nothing is only a component of the move in the arithmetic sense. The
        # dominant one is kept however small, because dropping it would leave
        # the sentence with nothing to be about.
        if component_points < 0.05:
            continue
        bucket = against_move if (value < 0) != (facts.ret < 0) else with_move
        bucket.append((label, component_points))

    if with_move and against_move:
        # "with 1.9 from NVIDIA itself and 2.8 from the wider market pushing
        # the other way" lets the trailing participle attach to both, so the
        # reader takes the 1.9 to have pushed the move along when it did the
        # opposite. When the remainder splits, each side says its own
        # direction, in its own clause.
        same = _join([f"{value:.1f} from {label}" for label, value in with_move])
        other = _join([f"{label} pushed {value:.1f}" for label, value in against_move])
        breakdown = f"{same} in the same direction, while {other} the other way"
    else:
        parts = [f"{value:.1f} from {label}" for label, value in with_move]
        if against_move:
            against_parts = [f"{value:.1f} from {label}" for label, value in against_move]
            tail = (
                "both pushing the other way" if len(against_parts) > 1 else "pushing the other way"
            )
            parts.append(f"{_join(against_parts)} {tail}")
        breakdown = _join(parts)

    share = abs(top_value) / total
    verdict = (
        f"Most of it was {top_label}"
        if share >= _DOMINANT_SHARE
        else f"The largest single piece was {top_label}"
    )
    # When something pushed the other way, the dominant part can be larger than
    # the move it sits inside -- a stock down 7.4 on 8.4 points of company news
    # and a point of help from the market. "8.4 of the 7.4 points" is true and
    # unreadable, so that case is said as what it is.
    top_points = abs(top_value) * 100
    if top_points > points:
        sentence = (
            f"{verdict}, worth {top_points:.1f} points on its own -- more than "
            f"the {points:.1f}-point move"
        )
    else:
        sentence = f"{verdict}: {top_points:.1f} of the {points:.1f} points"
    if breakdown:
        sentence += f", with {breakdown}"
    return f"{sentence}."


def event_sentence(facts: MoveFacts) -> str | None:
    """Name the scheduled events the day sat next to, if any."""
    events = []
    if facts.near_earnings:
        events.append("the company's own earnings")
    if facts.near_fomc:
        events.append("a Federal Reserve rate decision")
    if facts.near_cpi:
        events.append("an inflation release")
    if not events:
        return None
    if len(events) == 1:
        return f"The move landed within a day of {events[0]}."
    return f"The move landed within a day of {', '.join(events[:-1])} and {events[-1]}."


def peers_sentence(facts: MoveFacts) -> str | None:
    """Whether comparable companies moved with it, which is the cheapest test
    of "was this the company or the sector?" available without a model.

    It is a *second* reading of the same question the decomposition answers,
    and the two can come out differently -- peers are a handful of named
    companies, the sector component is a fitted factor. When they do, this
    sentence says so. It used to answer on its own and contradict the
    attribution sentence two lines above it ("Most of it was the rest of the
    sector ... NVDA moved largely on its own"), which reads as the app not
    believing itself.
    """
    peer = facts.peer_comove
    if peer is None:
        return None
    move = abs(facts.ret)
    # Rounding to the same 0.1% the text prints: a peer average of 0.0004 is
    # "barely moved", and printing "up 0.0%" invites the reader to wonder.
    if abs(peer) < 0.0005:
        peer_text = "Comparable companies barely moved that day"
    else:
        peer_text = f"Comparable companies were {plain_move(peer)} on average that day"
    if move <= 0:
        return f"{peer_text}."
    together = (peer < 0) == (facts.ret < 0)
    ratio = abs(peer) / move
    if together and ratio >= _PEERS_TOGETHER:
        peers_say = "together"
    elif not together or ratio <= _PEERS_ALONE:
        peers_say = "alone"
    else:
        peers_say = "mixed"

    # A middling peer average claims nothing strong enough to disagree with.
    if peers_say == "mixed":
        return f"{peer_text}, so some of this was shared across the group."

    top = _dominant(facts)
    company_led = top is not None and top[0] == "company"
    if top is None or company_led == (peers_say == "alone"):
        if peers_say == "together":
            return f"{peer_text}, so the whole group moved together."
        if not together:
            return f"{peer_text} -- the opposite direction, so this was {facts.ticker}'s own move."
        return f"{peer_text} -- far less, so {facts.ticker} moved largely on its own."

    # The two readings disagree. Neither verdict may be asserted as if the
    # other did not exist, and the tension is the honest thing to report.
    if peers_say == "together":
        return (
            f"{peer_text}, so the group moved with it even though the split above "
            f"puts the bigger part of the move on {facts.company} itself."
        )
    led = "sector" if top is not None and top[0] == "sector" else "market"
    gap = "the opposite direction" if not together else f"far less than {facts.ticker}"
    return (
        f"{peer_text} -- {gap}, which sits oddly with the {led}-led split above; "
        "read that split with some caution."
    )


# --------------------------------------------------------------------------- #
# Saying a relationship -- the v1.5 sentences (plan decision 9)
# --------------------------------------------------------------------------- #


def _clean_names(names: Sequence[str]) -> list[str]:
    """Upper-cased, de-duplicated, blank-free tickers, in the order given."""
    return [name for name in dict.fromkeys(str(n).strip().upper() for n in names) if name]


def _name_list(names: Sequence[str]) -> str:
    """`AMD`, `AMD and INTC`, `AMD, INTC and two others`.

    Past two symbols the sentence stops being a sentence, and the reader does
    not need the roster: the claim is that the money went to rivals, and two of
    them plus a count is as much of that as prose can carry.
    """
    kept = list(names)
    if len(kept) <= _MAX_NAMED:
        return _join(kept)
    rest = len(kept) - _MAX_NAMED
    others = f"{count_word(rest)} other" + ("s" if rest > 1 else "")
    return _join([*kept[:_MAX_NAMED], others])


def _moved_verb(ret: float) -> str:
    """`-0.036` -> `fell 3.6%`.

    A verb rather than `plain_move`'s "down 3.6%" because the subject here is a
    named company: "TSM down 3.6%" is a ticker tape, "TSM fell 3.6%" is English.
    """
    return f"{'fell' if ret < 0 else 'rose'} {abs(ret):.1%}"


def _comove_sentence(
    lead: str, comove: float | None, names: Sequence[str], fallback: str
) -> str | None:
    """`<lead>: AMD rose 3.1% the same day.`, or None with no number to say.

    `comove` is a *mean* across the related names, so it is labelled as one
    whenever the subject is not a single company.
    """
    if comove is None:
        return None
    kept = _clean_names(names)
    who = _name_list(kept) if kept else fallback
    average = "" if len(kept) == 1 else " on average"
    return f"{lead}: {who} {_moved_verb(comove)}{average} the same day."


def share_shift_sentence(facts: MoveFacts) -> str | None:
    """The rivals that moved the other way, when routing said `share_shift`.

    Said only when the sub-bucket was actually stored: the co-movement number
    on its own is not the claim -- `moves.sub_route` is the one place that
    decides whether rivals moving against the stock was large enough to mean
    anything, and this sentence repeats its verdict rather than re-deriving it.
    """
    if facts.sub_routing != _SUB_SHARE_SHIFT:
        return None
    return _comove_sentence("Share shift", facts.rival_comove, facts.competitors, "rivals")


def supply_chain_sentence(facts: MoveFacts) -> str | None:
    """The suppliers and customers that moved with it, when routing said
    `supply_chain`."""
    if facts.sub_routing != _SUB_SUPPLY_CHAIN:
        return None
    chain = (*facts.suppliers, *facts.customers)
    return _comove_sentence("Supply chain", facts.chain_comove, chain, "suppliers and customers")


def _driver_of(facts: MoveFacts) -> str | None:
    """The macro driver this move was actually labelled with, or None.

    `sub_routing` is the authority: `macro_driver` is computed for every day,
    whatever the routing bucket, so a company-driven day with a jumpy oil price
    still carries one and must not be narrated as a macro day. The fallback to
    `macro_driver` exists for facts built from a mapping that predates
    `sub_routing`, and is taken only when the move routed macro.
    """
    candidates = (facts.sub_routing, facts.macro_driver if facts.routing == "macro" else None)
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in _FACTOR_DRIVERS:
            return candidate
        if candidate.startswith(_COUNTRY_PREFIX) and candidate[len(_COUNTRY_PREFIX) :]:
            return candidate
    return None


def _country_name(code: str) -> str:
    """`TW` -> `Taiwan`; an unmapped code is its own name.

    Imported late, and from the one module that already owns the mapping, so
    this file keeps its promise to import nothing from the rest of the app at
    module level.
    """
    from stock_moves.news.geo import COUNTRY_NAMES

    return COUNTRY_NAMES.get(code, code)


def _geo_headline_count(facts: MoveFacts, code: str) -> int | None:
    """The stored headline count for this country on the day of the move.

    The lines are `news.geo.geo_event_lines` output and cover a ±1 day window,
    so the move's own date is picked out of them; a country-day with no row
    yields None and the clause that would have said it is dropped rather than
    printed as a zero.
    """
    if facts.day is None:
        return None
    wanted = facts.day.isoformat()
    for line in facts.geo_events:
        match = _GEO_LINE.match(str(line).strip())
        if match is None:
            continue
        day, country, count = match.groups()
        if day == wanted and country.upper() == code:
            return int(count)
    return None


def geo_sentence(facts: MoveFacts) -> str | None:
    """Name the macro driver the move was routed to (v1.5 decisions 4 and 6).

    Two shapes, because the two drivers are different claims. A country
    exposure is a *gated* one -- it is said only because the company has an
    edge to that country and the country's ETF moved -- so the sentence names
    the exposure, the ETF and, when there is one, how loud that country was in
    the news that day. A commodity or rates proxy needs none of that: there is
    one number and one clause.

    The percentage is the driver's contribution to this stock's return
    (`beta * proxy return`), not the proxy's own move, which is why it is said
    as what the ETF moved *the stock*.
    """
    driver = _driver_of(facts)
    if driver is None:
        return None

    component = facts.macro_driver_component
    moved = None if component is None else f"{abs(component):.1%}"

    if not driver.startswith(_COUNTRY_PREFIX):
        if moved is None:
            return f"Macro, via {driver}."
        return f"Macro, via {driver}: the {driver} proxy moved the stock {moved}."

    code = driver[len(_COUNTRY_PREFIX) :].upper()
    name = _country_name(code)
    clauses = []
    if moved is not None:
        clauses.append(f"the {name} ETF moved the stock {moved}")
    count = _geo_headline_count(facts, code)
    if count is not None:
        headline = "headline" if count == 1 else "headlines"
        clauses.append(f"{count_word(count)} geopolitical {headline} named {name}")
    head = f"Macro, geopolitical, via {name} exposure"
    return f"{head}: {_join(clauses)}." if clauses else f"{head}."


# --------------------------------------------------------------------------- #
# Saying one move
# --------------------------------------------------------------------------- #


def summary_sentences(facts: MoveFacts, headlines: Sequence[str] = ()) -> list[str]:
    """The sentences of a move's explanation, before any layout is applied.

    `HeuristicProvider.explain` joins these into the stored `summary`; the chat
    renderer lays them out over several lines instead. Same words either way,
    which is the point of having one function.
    """
    sentences = [
        f"{facts.company} was {plain_move(facts.ret)} on {plain_day(facts.day)}.",
        size_phrase(facts.ret_z, facts.ticker),
    ]
    sentences += [
        sentence
        for sentence in (
            attribution_sentence(facts),
            # The v1.5 sentences sit here, between the split and the calendar:
            # they are a finer reading of the same question the split answers
            # ("what was this move about?"), and they are absent whenever the
            # move carried no sub-bucket, which is most days.
            share_shift_sentence(facts),
            supply_chain_sentence(facts),
            geo_sentence(facts),
            event_sentence(facts),
            peers_sentence(facts),
        )
        if sentence
    ]
    if headlines:
        sentences.append(f"Reported that day: {_join(list(headlines))}.")
    return [sentence for sentence in sentences if sentence]


def headline_lines(articles: Sequence[Mapping[str, Any]], limit: int = 3) -> list[str]:
    """Headlines as indented bullets, one per line.

    The templated summary used to join these with semicolons, which is what
    turned a three-headline move into an unreadable 400-character line.
    """
    lines = []
    for article in articles[:limit]:
        title = str(article.get("title") or "").strip()
        if not title:
            continue
        source = str(article.get("source") or "").strip()
        lines.append(f"    - {title}" + (f" ({source})" if source else ""))
    return lines


def move_paragraph(
    facts: MoveFacts,
    articles: Sequence[Mapping[str, Any]] = (),
    note: str | None = None,
) -> str:
    """One move as a dated heading, an indented explanation and its headlines."""
    heading = f"{plain_day(facts.day)} -- {plain_move(facts.ret)}"
    body = [f"  {sentence}" for sentence in summary_sentences(facts)[1:]]
    if note:
        body.append(f"  {note}")
    lines = [heading, *body]
    bullets = headline_lines(articles)
    if bullets:
        lines.append("  Reported that day:")
        lines.extend(bullets)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Saying a set of moves -- the synthesis step
# --------------------------------------------------------------------------- #


def _set_summary(facts: Sequence[MoveFacts]) -> str | None:
    """One sentence about the set as a whole.

    This is the part a list of rows cannot do for the reader: whether these
    days have anything in common. Only claims that hold for a clear majority
    are stated, so the sentence is never a hedge about a mixed bag.
    """
    if len(facts) < 2:
        return None
    total = len(facts)
    company_led = sum(1 for f in facts if f.routing == "company")
    macro_led = sum(1 for f in facts if f.routing == "macro")
    earnings = sum(1 for f in facts if f.near_earnings)
    downs = sum(1 for f in facts if f.ret < 0)

    # A primary claim is one that characterises the set: what drove these days,
    # or that they all went the same way. Without one there is nothing to say,
    # and a lone detail ("one of them was an earnings day") is not a synthesis
    # -- it is a hedge dressed as one.
    primary = None
    if company_led == total:
        primary = "every one of them was driven by the company rather than the market"
    elif company_led > total / 2:
        primary = (
            f"{count_word(company_led)} of the {count_word(total)} were driven by "
            "the company itself"
        )
    elif macro_led > total / 2:
        primary = (
            f"{count_word(macro_led)} of the {count_word(total)} were market-wide "
            "rather than company news"
        )
    elif downs == total:
        primary = "all of them were falls"
    elif downs == 0:
        primary = "all of them were gains"

    if primary is None:
        return None

    parts = [primary]
    if downs == total and primary != "all of them were falls":
        parts.append("all of them were falls")
    elif downs == 0 and primary != "all of them were gains":
        parts.append("all of them were gains")
    if earnings == total:
        parts.append("every one landed around an earnings report")
    elif earnings:
        parts.append(f"{count_word(earnings)} landed around an earnings report")

    joined = parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])}, and {parts[-1]}"
    return f"Taken together, {joined}."


def _set_noun(direction: str | None, order: str) -> str:
    """What to call the set, in the singular, given how it was ranked.

    The header is the one line a reader takes the ranking from, so it has to
    name the ranking that actually produced the rows. A z-ordered list led by
    a 3.5% fall is not "the biggest fall" -- it is the most unusual one, and
    calling it the biggest is how the wrong day ends up quoted back.
    """
    kind = {"down": "fall", "up": "gain"}.get(direction or "", "move")
    adjective = "biggest" if order == "pct" else "most unusual"
    return f"{adjective} {kind}"


def narrate_moves(
    ticker: str,
    moves: Sequence[Mapping[str, Any]],
    company: str = "",
    direction: str | None = None,
    order: str = "z",
) -> str:
    """The synthesis handed to a reader in place of one row per move.

    Structure: what this set is, what it has in common, then each move as its
    own dated block, then the caveat once.

    `order` is the ranking the rows arrived in ("z" or "pct"), and it decides
    only what the header calls them -- the rows are already in whatever order
    the query layer put them in.
    """
    if not moves:
        return f"No stored moves for {ticker}."

    facts = [MoveFacts.from_mapping(move, ticker, company) for move in moves]
    singular = _set_noun(direction, order)
    subject = short_company_name(company or ticker, ticker)
    if len(moves) == 1:
        header = f"{subject} ({ticker}) -- the {singular} in the data"
    else:
        header = f"{subject} ({ticker}) -- the {count_word(len(moves))} {singular}s in the data"

    blocks = [
        move_paragraph(fact, _articles_of(move)) for fact, move in zip(facts, moves, strict=True)
    ]
    sections = [header]
    lead = _set_summary(facts)
    if lead:
        sections.append(lead)
    sections.extend(blocks)
    sections.append(ATTRIBUTION_CAVEAT)
    return "\n\n".join(sections)


def narrate_one_move(
    ticker: str,
    move: Mapping[str, Any],
    company: str = "",
    model_summary: str | None = None,
) -> str:
    """One move in full.

    `model_summary` is a stored explanation written by a keyed provider. It is
    shown *instead of* the templated sentences, because a model that read the
    headlines can say more than the decomposition alone. A summary written by
    the heuristic is not passed here: it is regenerated from the numbers, so
    rows stored before this module existed still read well.
    """
    facts = MoveFacts.from_mapping(move, ticker, company)
    articles = _articles_of(move)
    if model_summary:
        heading = f"{plain_day(facts.day)} -- {plain_move(facts.ret)}"
        lines = [heading, f"  {model_summary}"]
        bullets = headline_lines(articles)
        if bullets:
            lines.append("  Reported that day:")
            lines.extend(bullets)
        return "\n\n".join(["\n".join(lines), ATTRIBUTION_CAVEAT])
    return "\n\n".join([move_paragraph(facts, articles), ATTRIBUTION_CAVEAT])


def narrate_news(ticker: str, articles: Sequence[Mapping[str, Any]], limit: int = 10) -> str:
    """Headlines grouped under the day they were published."""
    if not articles:
        return f"No stored headlines for {ticker}."

    by_day: dict[str, list[Mapping[str, Any]]] = {}
    for article in articles[:limit]:
        day = _as_date(article.get("published_at"))
        by_day.setdefault(plain_day(day), []).append(article)

    sections = [f"Headlines linked to {ticker}'s moves:"]
    for day, rows in by_day.items():
        sections.append("\n".join([f"  {day}", *headline_lines(rows, limit=len(rows))]))
    return "\n\n".join(sections)


# --------------------------------------------------------------------------- #
# Saying the edges
# --------------------------------------------------------------------------- #


def _relation_group(
    edges: Sequence[Mapping[str, Any]], relation: str
) -> list[tuple[str, float, str]]:
    """One relation's edges as `(dst, weight, source)`, strongest first.

    Weight order rather than the alphabetical order the query layer returns:
    the answer to "who does it compete with" is the closest rival, and an ETF's
    holdings carry their size in the weight. `sorted` is stable, so edges of
    equal weight keep the order they arrived in. A row with no `dst` is
    dropped -- it names nothing, and printing an empty name would be worse
    than being one name short.
    """
    rows: list[tuple[str, float, str]] = []
    for edge in edges:
        if not isinstance(edge, Mapping) or _as_str(edge.get("relation")) != relation:
            continue
        dst = _as_str(edge.get("dst"))
        if dst is None:
            continue
        weight = _as_float(edge.get("weight"))
        rows.append((dst, 0.0 if weight is None else weight, _as_str(edge.get("source")) or ""))
    return sorted(rows, key=lambda row: -abs(row[1]))


def _source_clause(rows: Sequence[tuple[str, float, str]]) -> str:
    """` (from the sector ETF's largest holdings)`, or nothing at all.

    Sources are de-duplicated in the order they appear, so a group that came
    from one place reads as one phrase. An unrecognised source name is left
    out rather than printed raw: a storage string in the middle of a sentence
    tells the reader less than no clause does.
    """
    seen = dict.fromkeys(row[2] for row in rows)
    phrases = [_SOURCE_PHRASES[source] for source in seen if source in _SOURCE_PHRASES]
    return f" ({_join(phrases)})" if phrases else ""


def _competitor_sentence(subject: str, rows: Sequence[tuple[str, float, str]]) -> str:
    """The whole roster, not the two-name cap `_name_list` applies elsewhere.

    A share-shift clause inside a move is making a point about where the money
    went, so two names and a count carry it. This sentence *is* the answer to
    the question asked, so leaving rivals out of it would be answering a
    different one.
    """
    names = _join([row[0] for row in rows])
    clause = _source_clause(rows)
    if len(rows) == 1:
        return f"{subject}'s closest listed competitor on record is {names}{clause}."
    return f"{subject}'s closest listed competitors on record are {names}{clause}."


def _chain_sentence(rows: Sequence[tuple[str, float, str]], noun: str) -> str:
    """`Its suppliers on record are TSM and ASML (as suggested by the model).`"""
    names = _join([row[0] for row in rows])
    clause = _source_clause(rows)
    if len(rows) == 1:
        return f"Its {noun} on record is {names}{clause}."
    return f"Its {noun}s on record are {names}{clause}."


def _country_sentence(rows: Sequence[tuple[str, float, str]]) -> str:
    """Countries by name, with the weight as a percentage.

    The weight is that source's strength, not a share of revenue, so it is
    attached to the country rather than described -- a reader who sees `(30%)`
    next to a source clause can tell what kind of number it is, and a sentence
    that called it "30% of its business" would be inventing a fact.
    """
    named = [f"{_country_name(dst.upper())} ({weight:.0%})" for dst, weight, _ in rows]
    return f"It is exposed to {_join(named)}{_source_clause(rows)}."


def _factor_label(name: str) -> str:
    """`dollar` -> `the dollar`; an unmapped proxy is its own name."""
    return _FACTOR_LABELS.get(name.lower(), name)


def _factor_sentence(rows: Sequence[tuple[str, float, str]]) -> str:
    """The fitted exposures as one sentence, loudest first.

    The word "beta" appears once, on the first number, and the rest are bare:
    repeating the label four times is a table, not a sentence, and the reader
    only needs telling once what kind of number is in the brackets. No source
    clause -- a beta is fitted from prices by construction, so the clause would
    say what the word already said.
    """
    strong = [row for row in rows if abs(row[1]) >= _FACTOR_NOTABLE]
    weak = [_factor_label(row[0]) for row in rows if abs(row[1]) < _FACTOR_NOTABLE]
    if not strong:
        return f"It barely moves with {_join(weak)}."

    named = [
        f"{_factor_label(dst)} ({'beta ' if index == 0 else ''}{weight:.1f})"
        for index, (dst, weight, _) in enumerate(strong)
    ]
    sentence = f"It moves most with {_join(named)}"
    if not weak:
        return f"{sentence}."
    verb = "barely registers" if len(weak) == 1 else "barely register"
    return f"{sentence}; {_join(weak)} {verb}."


def narrate_relations(
    ticker: str,
    company_name: str,
    edges: Sequence[Mapping[str, Any]],
) -> str:
    """The stored edges of one company, read out loud (v1.5 decision 10).

    The keyless answer to "who does NVDA compete with?". `get_relations`
    returns `{ticker, edges: [{dst, relation, weight, source}]}`, which is a
    table; this turns it into the four or five sentences a reader wanted --
    rivals, then the supply chain, then country exposure, then the fitted
    factor betas -- each one naming where its claim came from.

    An empty edge list is a real answer rather than an error (v1.5 decision 3:
    without a key there are no model relations at all), so it is said as the
    next step it actually is: ingest the ticker.
    """
    rows = [edge for edge in edges if isinstance(edge, Mapping)]
    subject = short_company_name(company_name or ticker, ticker)
    grouped = {relation: _relation_group(rows, relation) for relation in _RELATION_ORDER}

    lines: list[str] = []
    if grouped["competitor"]:
        lines.append(_competitor_sentence(subject, grouped["competitor"]))
    if grouped["supplier"]:
        lines.append(_chain_sentence(grouped["supplier"], "supplier"))
    if grouped["customer"]:
        lines.append(_chain_sentence(grouped["customer"], "customer"))
    if grouped["country"]:
        lines.append(_country_sentence(grouped["country"]))
    if grouped["factor"]:
        lines.append(_factor_sentence(grouped["factor"]))

    # A relation this module has no sentence for is still a stored fact, and a
    # later decision could add one. Naming it plainly beats dropping rows the
    # database holds and the reader asked about.
    known = set(_RELATION_ORDER)
    extra = sorted({str(edge.get("relation") or "") for edge in rows} - known - {""})
    for relation in extra:
        named = _join([row[0] for row in _relation_group(rows, relation)])
        lines.append(f"Also on record, {relation}: {named}.")

    if not lines:
        return f"Nothing is on record for {ticker} yet; ingest it first."
    return "\n".join(lines)


def _articles_of(move: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    articles = move.get("articles")
    if not isinstance(articles, Sequence) or isinstance(articles, (str, bytes)):
        return []
    return [row for row in articles if isinstance(row, Mapping)]
