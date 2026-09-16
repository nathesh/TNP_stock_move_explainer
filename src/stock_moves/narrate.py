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
        )


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def attribution_sentence(facts: MoveFacts) -> str | None:
    """Split the move into market, sector and company in points of the move.

    The components are `beta * factor return`, so they sum to the day's return
    and can be said as "8.6 of those 14.5 points" -- a share of the move the
    reader can check by adding up. A component whose sign opposes the move is
    called out as pushing the other way, because silently reporting its size
    would read as if it had contributed.
    """
    labelled = [
        ("the wider market", facts.mkt_component),
        ("the rest of the sector", facts.sector_component),
        (f"{facts.company} itself", facts.idio_component),
    ]
    present = [(label, value) for label, value in labelled if value is not None]
    # Said rather than skipped: a reader who is told how large the move was and
    # then hears nothing about what drove it should know the split is missing,
    # not be left to assume it was unremarkable.
    absent = "There is no market-versus-company breakdown for this day."
    if not present:
        return absent

    total = sum(abs(value) for _, value in present)
    if total <= 0:
        return absent
    present.sort(key=lambda item: abs(item[1]), reverse=True)
    top_label, top_value = present[0]
    points = abs(facts.ret) * 100

    # The verdict already names the dominant component, so the breakdown that
    # follows it must not name it again ("Most of it was Tesla itself: 8.6 came
    # from Tesla itself" is what naming it twice reads like).
    # The two directions are grouped so "pushing the other way" is said once
    # for however many components did it, not once each.
    with_move: list[str] = []
    against_move: list[str] = []
    for label, value in present[1:]:
        bucket = against_move if (value < 0) != (facts.ret < 0) else with_move
        bucket.append(f"{abs(value) * 100:.1f} from {label}")
    breakdown = list(with_move)
    if against_move:
        tail = "both pushing the other way" if len(against_move) > 1 else "pushing the other way"
        breakdown.append(f"{_join(against_move)} {tail}")

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
        sentence += f", with {_join(breakdown)}"
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
    of "was this the company or the sector?" available without a model."""
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
        return f"{peer_text}, so the whole group moved together."
    if not together:
        return f"{peer_text} -- the opposite direction, so this was {facts.ticker}'s own move."
    if ratio <= _PEERS_ALONE:
        return f"{peer_text} -- far less, so {facts.ticker} moved largely on its own."
    return f"{peer_text}, so some of this was shared across the group."


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


def narrate_moves(
    ticker: str,
    moves: Sequence[Mapping[str, Any]],
    company: str = "",
    direction: str | None = None,
) -> str:
    """The synthesis handed to a reader in place of one row per move.

    Structure: what this set is, what it has in common, then each move as its
    own dated block, then the caveat once.
    """
    if not moves:
        return f"No stored moves for {ticker}."

    facts = [MoveFacts.from_mapping(move, ticker, company) for move in moves]
    singular = {"down": "biggest fall", "up": "biggest gain"}.get(direction or "", "biggest move")
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


def _articles_of(move: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    articles = move.get("articles")
    if not isinstance(articles, Sequence) or isinstance(articles, (str, bytes)):
        return []
    return [row for row in articles if isinstance(row, Mapping)]
