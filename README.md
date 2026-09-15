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

**Zero keys required.** With no API key the keyless `HeuristicProvider` scores
headlines by entity hits and templates the explanation from the decomposition;
news comes from keyless Google News RSS. An `ANTHROPIC_API_KEY` upgrades the
scoring and the prose (and the chat endpoint to a real tool-calling loop); if
a model call fails the app degrades to the heuristic instead of erroring, and
the stored explanation is labelled with the provider that actually wrote it.

## Setup

```bash
uv sync
```

No keys are needed. Optional settings go in a `.env` file in the repo root:

```dotenv
# .env  (all optional)
ANTHROPIC_API_KEY=sk-ant-your-key-here
ANTHROPIC_MODEL=claude-sonnet-5
NEWS_SOURCE=google_rss          # or: gdelt
DB_PATH=data/app.db
```

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
- **`data/snapshot.db.gz`**, a pre-ingested database of ~110 large caps, one
  year each. `/tmp` is per-instance and wiped on a cold start, so without a
  seed the first visitor would meet an empty page; `stock_moves.seed` expands
  the snapshot when no database is present. Rebuild it with
  `uv run python scripts/build_snapshot.py`.

```bash
vercel deploy          # preview
vercel deploy --prod   # production
```

An `ANTHROPIC_API_KEY` set in the Vercel project's environment variables
upgrades scoring, prose and chat exactly as it does locally; the committed
snapshot keeps whichever provider wrote each stored explanation.

Two properties of the deployment worth stating plainly, because they are
limits and not surprises: `/tmp` is not shared between instances, so an ingest
one visitor triggers is not visible to another, and the snapshot's data ages
until the next rebuild — a request for a stale ticker re-ingests it live.

## API, by example

Ingest a year of prices, detect the moves, fetch and score news, and explain
the five biggest:

```bash
curl -s -X POST 'http://127.0.0.1:8000/tickers/AAPL/ingest?period=1y&top_n=5' | python -m json.tool
```

```json
{
  "ticker": "AAPL", "period": "1y", "top_n": 5,
  "n_prices": 251, "n_moves": 43, "n_articles": 150, "n_explanations": 5,
  "provider": "anthropic", "news_source": "google_rss"
}
```

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
      "gap_ret": -0.010958, "intraday_ret": -0.023859, "vol_z": 1.799384,
      "routing": "company", "near_earnings": true,
      "mkt_component": 0.010, "sector_component": 0.001, "idio_component": -0.084,
      "peer_comove": -0.000527,
      "explanation": {
        "summary": "AAPL fell 7.4% on 2026-07-31, a 4.0-sigma day ... The day was inside an earnings window. Related headlines: Apple stock falls on weak revenue forecast as CEO Tim Cook flags 'increasing impact' from memory shortage (Yahoo Finance); ...",
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

246 tests, and **none of them touch the network**: prices, news and the model
provider are all faked. They cover move detection and the decomposition on
synthetic price frames, the RSS and GDELT parsers on canned payloads, the
keyless scorer and explainer, the Anthropic provider's schema handling and its
degrade-to-heuristic behaviour, idempotent ingest, the query layer's filters,
every API route through FastAPI's `TestClient`, and the chat tool loop. One
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
- SQLite on local disk; not deployable as-is to a serverless host.
