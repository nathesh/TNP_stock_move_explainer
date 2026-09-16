# stock-move-explainer

Explains major daily stock moves using news, a factor decomposition and a
language model.

## Approach

The "why" behind a one-day move is **attribution, not causation** — one event
day plus a bag of articles is not an identifiable causal problem. So the system
does two things. First, a **quantitative decomposition** of every trading day
(daily return, its z-score against a trailing 20-day vol, overnight gap vs
in-session return, volume z, and a trailing 60-day OLS of the stock on SPY and
its sector ETF) says *where* to look — `company`, `industry` or `macro` —
before a single article is read. Second, **grounded retrieval plus a model**
writes the explanation: headlines are pulled from a ±1 day window with a query
built for that routing bucket, scored for relevance and category, and the top
few are handed to the explainer, which cites the article ids it used, reports a
confidence, and is explicitly allowed to answer `unexplained` when the evidence
is weak. Earnings proximity, a hardcoded FOMC/CPI calendar, market and sector
regime, and same-day peer co-movement are part of the same context. Everything
external sits behind an interface (`NewsSource`, `ModelProvider`), so a paid
news feed or a different model is a swap, not a rewrite.

**Written for a reader, not for a quant.** A z-score, a factor loading and a
component in percentage points are all meaningful to someone who already knows
what they are, and noise to everyone else. `narrate.py` is the single place
that decides how each quantity is *said* -- a z-score becomes "about four times
the size of a typical day", the decomposition becomes "8.6 of the 14.5 points
came from Tesla itself" -- and it feeds both the keyless renderer and the
keyed providers' prompts, so an explanation reads the same however it was
produced. An answer about several moves leads with what they have in common
before listing them, because a reader who wanted rows would have read the
table.

**Zero keys required.** With no API key the keyless `HeuristicProvider` scores
headlines by entity hits and narrates the explanation from the decomposition;
news comes from keyless Google News RSS. An `OPENAI_API_KEY` (or an
`ANTHROPIC_API_KEY`) upgrades the scoring and the prose (and the chat endpoint
to a real tool-calling loop); if a model call fails the app degrades to the
heuristic instead of erroring, and the stored explanation is labelled with the
provider that actually wrote it.

## v1.5: the relationship layer

Today the app knows one fact per company: its peers. v1.5 stores a short list of
typed facts per company (competitors, suppliers, customers, countries, and
price-derived factor betas), and a small table of dated geopolitical headline
counts per country. It uses them in exactly two places: routing (a move gets a
`sub_routing` such as `share_shift`, `supply_chain`, `oil`, `country:TW`) and
retrieval (the news query expands along the edges). One gate rule stops
geopolitical headlines from being credited unless the company has the country
edge and the country's factor actually moved that day.

**The edges are one table and one read.** `company_edges(src, dst, relation,
weight, source)` holds all five relations; `competitor` replaces the old peer
list rather than sitting beside it. `GET /tickers/{ticker}/relations` returns
them, filterable with `?relation=`:

```bash
curl -s 'http://127.0.0.1:8000/tickers/NVDA/relations' | python -m json.tool
curl -s 'http://127.0.0.1:8000/tickers/NVDA/relations?relation=country' | python -m json.tool
```

```json
{
  "ticker": "NVDA",
  "edges": [
    {"dst": "AMD", "relation": "competitor", "weight": 0.9, "source": "model"},
    {"dst": "TSM", "relation": "supplier", "weight": 0.8, "source": "model"},
    {"dst": "TW",  "relation": "country",   "weight": 0.7, "source": "model"},
    {"dst": "oil", "relation": "factor",    "weight": -0.11, "source": "prices"}
  ]
}
```

It is a pure read: edges are written by an ingest, and asking for them is not
asking to go and build them, so a stored ticker with none answers `[]`.

**Five new fields on every move.** `sub_routing` is the sub-bucket inside the
unchanged `company|industry|macro` routing — `share_shift` when competitors
moved the *opposite* way, `supply_chain` when suppliers or customers moved the
*same* way, and `oil|dollar|rates|gold|country:XX` on a macro day — and it is
`null` when no rule fires. `macro_driver` and `macro_driver_component` name the
factor proxy with the largest contribution that day (`USO`, `UUP`, `TLT`, `GLD`
and one country ETF per country edge) and how much of the move it accounts for,
counted only when the proxy itself moved (`abs(z) >= 1.5`). `rival_comove` and
`chain_comove` are the same-day average returns of the `competitor` edges and of
the `supplier`/`customer` edges — the two numbers `sub_routing` is read off.

**The gate rule, in one sentence.** A headline matching the geopolitical
vocabulary is capped at relevance `0.30` unless *both* hold — the title names a
country the company has a `country` edge to, and the move's `macro_driver` is
that country or `oil`/`dollar` — so a tariff story is never credited for a move
the exposure and the prices do not both support.

**No key means no country edges, so the geo gate stays closed.** Country,
supplier and customer edges come from one `suggest_relations` call per company;
the keyless provider returns competitors from the sector ETF's holdings and
nothing else. So without a working key there is no country to match, the gate
never opens, and no `country:XX` sub-routing can fire. That is the honest
behaviour rather than a bug, and it is why the smoke check below comes first.

**Run the smoke check first, then the eval.** `scripts/smoke_model.py` is the
first thing to run on any machine: it makes one call through the app's own
provider path and one raw SDK call, because the providers degrade to the
heuristic on any exception, so a dead key otherwise looks like a quiet answer
(exit 0 proven, 1 degraded or failed, 2 no key configured). `scripts/eval_v15.py`
then runs the three ticker-days fixed in `docs/v1.5-plan.md` before the build —
NVDA 2025-04-16, NVDA 2025-01-27, AAPL 2025-04-03 — printing each move's
numbers, prose and cited headlines, and judging `routing` and `sub_routing`
separately. It uses the network, is not part of pytest, and ingests into a fresh
temporary database unless given `--db PATH`:

```bash
uv run python scripts/smoke_model.py
uv run python scripts/eval_v15.py
```

## Setup

```bash
uv sync
```

No keys are needed. Optional settings go in a `.env` file in the repo root:

```dotenv
# .env  (all optional)
OPENAI_API_KEY=sk-proj-your-key-here
OPENAI_MODEL=gpt-4.1            # the workhorse tier; any chat model works
ANTHROPIC_API_KEY=sk-ant-your-key-here
ANTHROPIC_MODEL=claude-sonnet-5
NEWS_SOURCE=google_rss          # or: gdelt
DB_PATH=data/app.db
```

Both vendors are implemented against the same `ModelProvider` protocol and
share their prompts, so the choice is one environment variable. OpenAI wins
when both keys are set. `gpt-4.1` is the default because this app makes
roughly one call per move: a reasoning model's latency is paid ten times over
on a single ingest, and the task is writing, not reasoning.

## Run

```bash
uv run stock-moves
```

Then open:

- <http://127.0.0.1:8000/> — a one-page chat UI
- <http://127.0.0.1:8000/docs> — Swagger

## Deploy

The app runs on Vercel as a single Python function. Three things make that work,
and each is a consequence of how serverless hosting differs from a laptop:

- **`app.py`** at the repo root. Vercel loads a `FastAPI` instance named `app`
  from a fixed set of filenames; the real application lives in a package, so
  this file is the bridge.
- **`DB_PATH=/tmp/app.db`** (set in `vercel.json`). The filesystem is read-only
  apart from `/tmp`, so the default `data/app.db` cannot be written.
- **`data/snapshot.db.gz`**, a pre-ingested database of ~110 large caps, two
  years each. `/tmp` is per-instance and wiped on a cold start, so without a
  seed the first visitor would meet an empty page; `stock_moves.seed` expands
  the snapshot when no database is present. Rebuild it with
  `uv run python scripts/build_snapshot.py`. After a change to the phrasing,
  `uv run python scripts/renarrate_snapshot.py` rewrites the snapshot's keyless
  explanations in place and re-gzips it, off the network and without a key.

  **When and how to rebuild it.** The committed snapshot is a v1.5 one: 110
  large caps, two years each, every new table and column present, and every
  stored explanation keyless. It was built with no key, so its only edges are
  `competitor` from the sector ETFs' holdings and `factor` from prices — there
  are no model-suggested supplier, customer or country edges in it, and so no
  geo events either, because a geo query is only run for a country the company
  has an edge to. A rebuild with a key set adds both, and writes the
  explanations through the keyed provider. Rebuild when the schema gains a
  column, when the two years have aged out from under the demo, or to bring
  those keyed edges in. Delete the working `data/snapshot.db` first — schema
  creation adds missing tables but never columns to an existing one — then
  rerun `scripts/build_snapshot.py`.

```bash
vercel deploy          # preview
vercel deploy --prod   # production
```

An `OPENAI_API_KEY` set in the Vercel project's environment variables
upgrades scoring, prose and chat exactly as it does locally; the committed
snapshot keeps whichever provider wrote each stored explanation. With no key
set the deployment still works -- it answers from the heuristic, in the same
plain English.

Two properties of the deployment worth stating plainly, because they are
limits and not surprises: `/tmp` is not shared between instances, so an ingest
one visitor triggers is not visible to another, and the snapshot's data ages
until the next rebuild — a request for a stale ticker re-ingests it live.

## API, by example

Ingest two years of prices, detect the moves, fetch and score news, and
explain the five biggest (two years, not one, so the 200-day SMA behind the
regime labels has warmed up):

```bash
curl -s -X POST 'http://127.0.0.1:8000/tickers/AAPL/ingest?period=2y&top_n=5' | python -m json.tool
```

```json
{
  "ticker": "AAPL", "period": "2y", "top_n": 5,
  "n_prices": 500, "n_moves": 86, "n_articles": 291, "n_explanations": 5,
  "provider": "heuristic", "news_source": "google_rss",
  "n_edges": 11, "n_geo_events": 0
}
```

Those are the keyless numbers, which is what a fresh checkout gets: the 11
edges are `competitor` from the sector ETF's holdings plus the price-fitted
`factor` betas, and `n_geo_events` is 0 because a geo query is only run for a
country the company has an edge to. With `OPENAI_API_KEY` set the same call
adds model-suggested supplier, customer and country edges, the geo events those
countries make possible, and reads `"provider": "openai"`.

Read the moves back, biggest first, down days only:

```bash
curl -s 'http://127.0.0.1:8000/tickers/AAPL?direction=down&limit=5' | python -m json.tool | head -80
```

```json
{
  "ticker": "AAPL",
  "company": {"name": "Apple Inc.", "sector": "Technology",
              "industry": "Consumer Electronics", "sector_etf": "XLK",
              "peers": ["AMD", "AVGO", "CSCO", "INTC", "MSFT", "MU", "NVDA"]},
  "moves": [
    {
      "date": "2026-07-31",
      "ret": -0.073539, "ret_z": -4.03945,
      "gap_ret": -0.085835, "intraday_ret": 0.013451, "vol_z": 2.60752,
      "routing": "company", "near_earnings": true,
      "mkt_component": 0.010, "sector_component": 0.001, "idio_component": -0.084,
      "peer_comove": -0.000527,
      "explanation": {
        "summary": "Apple was down 7.4% on Friday, 31 July 2026. That is roughly four times the size of a typical day for AAPL. Most of it was Apple itself, worth 8.4 points on its own -- more than the 7.4-point move, with 1.0 from the wider market and 0.1 from the rest of the sector both pushing the other way. The move landed within a day of the company's own earnings. Comparable companies were down 0.1% on average that day -- far less, so AAPL moved largely on its own. Reported that day: the day's coverage led with “Apple stock falls on weak revenue forecast as CEO Tim Cook flags 'increasing impact' from memory shortage” (Yahoo Finance).",
        "primary_category": "company", "confidence": 0.9,
        "cited_article_ids": [31, 33, 36], "unexplained": false
      },
      "articles": [
        {"id": 31, "title": "Apple stock falls on weak revenue forecast as CEO Tim Cook flags 'increasing impact' from memory shortage",
         "source": "Yahoo Finance", "published_at": "2026-07-31T...",
         "relevance": 0.91, "category": "company"}
      ]
    }
  ]
}
```

The rest:

```bash
# only the sharpest days, numbers without news
curl -s 'http://127.0.0.1:8000/tickers/AAPL?z_threshold=2.5&include_news=false' | python -m json.tool | head -40

# one move with everything attached (explained on demand if not cached)
curl -s 'http://127.0.0.1:8000/tickers/AAPL/moves/2026-07-31' | python -m json.tool

# ask in English; the reply is grounded in the same read functions
curl -s -X POST http://127.0.0.1:8000/chat \
  -H 'content-type: application/json' \
  -d '{"message":"why did AAPL drop the most this year?","ticker":"AAPL"}' | python -m json.tool

# the chat page
curl -s http://127.0.0.1:8000/ | head -5
```

A relative phrase in a chat question — "yesterday", "last week", "this year" —
is resolved on the server from the exchange's clock (America/New_York, not the
server's zone) and echoed back in the response's `window`; the model is never
shown a date field and never asked for a date.

`GET /tickers/{ticker}` also takes `start`, `end`, `pct_threshold`,
`category`, `min_relevance`, `include_prices` and `refresh`.

## How the thresholds work

A day is a **move** when `abs(ret_z) >= z_threshold` (default 2.0) **OR**
`abs(ret) >= pct_threshold` (default 0.02) — the two are OR-ed, so a quiet
stock trips on z and a volatile one trips on the percentage.

The thresholds are used in two different places:

- **At ingest time** they decide which days are written to the `moves` table
  (and only stored moves get news fetched and explanations written).
- **On a GET** they only *filter* what is already stored.

So raising a GET threshold shows fewer moves immediately, but lowering it below
the value used at ingest will not conjure new ones — re-ingest with the lower
threshold (`POST .../ingest?z_threshold=1.5`) to make more days into moves.

## Tests

```bash
uv run pytest -q
```

636 tests, and **none of them touch the network**: prices, news and both
model providers are all faked. They cover move detection and the decomposition
on synthetic price frames, the RSS and GDELT parsers on canned payloads, the
keyless scorer and explainer, the phrasing rules in `narrate` (including that
no jargon reaches the reader and that each sentence stays true of the numbers
it came from), each keyed provider's schema handling and its
degrade-to-heuristic behaviour, idempotent ingest, the query layer's filters,
every API route through FastAPI's `TestClient`, and both chat tool loops. One
test is marked `network` and is skipped by default; it is the only live check.

## Design

See [DESIGN.md](DESIGN.md) for the decisions and why,
[docs/architecture-v1.md](docs/architecture-v1.md) for what was built, and
[docs/architecture-v2.md](docs/architecture-v2.md) for where it goes next.

## Limitations

- Attribution, not causation: the explanation says what the news said on the day, not what moved the price.
- Both news sources give headlines only; scoring cannot see article bodies. Coverage of small caps is thin.
- RSS urls are Google redirects, so the same story reached through two redirect urls is stored twice; GDELT is throttled to one request per five seconds.
- The 60-day OLS is a rough factor model; betas are noisy on volatile names and near regime changes.
- Sector ETF mapping is a static dict; conglomerates and misclassified `yfinance` sectors route badly.
- Without a model key, peers are the sector ETF's top holdings rather than true competitors, so peer co-movement is a sector proxy.
- Confidence is uncalibrated; `unexplained` reflects the model's opinion, not a measured error rate.
- No scheduled ingest; data goes stale until `?refresh=true`.
- Single-day moves only; multi-day drifts and gap-then-reversal patterns are not modeled.
- No executive or social commentary sources.
- On the serverless deployment the database is a per-instance copy in `/tmp`, re-seeded from the snapshot on every cold start: what one instance ingests, another never sees, and neither keeps it for long.
