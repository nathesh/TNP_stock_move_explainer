# Design: explaining major stock moves with news

Take-home, 4-hour window, Python. Decisions below are the source of truth for
`docs/architecture-v1.md`, `docs/architecture-v2.md`, and `docs/v1-plan.md`.

## 0. Framing

**The "why" is attribution, not causation.** One event day and a bag of articles
is not an identifiable causal problem. The system does two things:

1. A **quantitative decomposition** of each move that says *where* to look
   (company / industry / macro) before any article is read.
2. **Grounded retrieval + a language model** that writes the explanation with
   citations and a confidence, and is explicitly allowed to return
   `unexplained` when the evidence is weak.

v1 is a local prototype on free, keyless data. Every external dependency sits
behind an interface so v2 can swap in paid news, our own classifiers, and AWS.

## 1. Prices and move detection

Source: `yfinance` daily OHLCV for the ticker, `SPY`, and the sector ETF
(sector from `yfinance` `info`, mapped to XLK/XLF/XLE/... via a static dict).

Per trading day we compute and **store as columns** (thresholds are API
filters, not code constants):

| column | meaning |
|---|---|
| `ret` | close-to-close daily return |
| `ret_z` | `ret` / trailing 20-day std of `ret` (the primary "major" definition, flag at abs >= 2.0) |
| `gap_ret`, `intraday_ret` | prev close→open, open→close (overnight news vs in-session news) |
| `vol_z` | volume / trailing 20-day mean volume |
| `mkt_component`, `sector_component`, `idio_component` | from a trailing 60-day OLS of stock on SPY and sector ETF; `idio = ret - beta_mkt*ret_spy - beta_sec*ret_etf` |
| `routing` | `company` if abs(idio) dominates, `industry` if sector dominates, `macro` if market dominates |
| `regime_mkt`, `regime_sector` | `bull`/`bear` from 50 vs 200 day SMA of SPY / sector ETF on that date |
| `near_earnings` | true if within ±1 trading day of a `yfinance` earnings date |
| `near_fomc`, `near_cpi` | true if within ±1 trading day of a Fed decision or CPI release, from a hardcoded 2025–2026 calendar (the macro equivalent of the earnings flag) |

A **move** is a day with `abs(ret_z) >= z_threshold` (default 2.0) OR
`abs(ret) >= pct_threshold` (default 0.02, kept configurable because the prompt
suggests it; z is the real definition). Both thresholds are query params.

Move detection is a pure function over a DataFrame → unit-testable with no
network.

## 2. Company ontology (seed of the v2 knowledge graph)

Table `companies(ticker, name, sector, industry, sector_etf, peers_json,
updated_at)`. Name/sector/industry from `yfinance`; `peers` from one cached LLM
call; keyless fallback: the top holdings of the sector ETF via `yfinance`. Peers give a **second, data-driven industry
signal**: same-day co-movement of peer prices. If AMD and NVDA both fell 5%, the
move is industry before any article is read.

## 3. News

Probed 2026-09-15 with no keys: **Google News RSS** returns 100 dated,
sourced items per query with `after:`/`before:` operators and no throttle;
**GDELT DOC 2.0** works but enforces one request per five seconds; yfinance
`news` is recent-only. So:

- `NewsSource` protocol: `search(query: str, start: date, end: date, limit: int) -> list[Article]`.
- **`GoogleNewsRSS` is the v1 primary** (keyless, historical, mainstream outlets;
  cost: title/source/url/date only, no body, and a Google redirect URL).
- **`GDELTSource` is the second implementation**, behind a 5-second client-side
  throttle, used as a fallback or when `NEWS_SOURCE=gdelt`.
- Queries are built per routing bucket, and always anchored to the market so
  a bare company name does not return obituaries:
  - `company` → `"<company name>" stock` (plus ticker) in a ±1 day window
  - `industry` → company name OR peer names OR industry term, same window
  - `macro` → fixed macro vocabulary (Federal Reserve, CPI, tariff, ...) same window

Plus **earnings dates from `yfinance`** as a structured, keyless event source.
Twitter / executive commentary: out of v1 (no API path in 4h), listed in v2.
v2 adds a paid full-text source (Exa) behind the same protocol.

Articles are deduped on URL, stored once, and linked to moves through
`move_articles(move_id, article_id, relevance, category)`.

## 4. Scoring and explanation (the model layer)

`ModelProvider` interface with three implementations:

- `OpenAIProvider` (used when `OPENAI_API_KEY` is set) and `AnthropicProvider`
  (when `ANTHROPIC_API_KEY` is): each scores an article (relevance 0–1,
  category company/industry/macro) and writes the per-move explanation. They
  share their prompts, their structured-output schemas and their prompt
  rendering in `providers/prompts.py`; what differs is transport, so the two
  answer alike and neither can drift when a prompt is edited. OpenAI is
  preferred when both keys are set.
- `HeuristicProvider` (no key): keyword rules + the decomposition. Relevance =
  name/peer/macro-term hits; category = routing bucket; explanation = a
  templated sentence from the numbers. **The app runs with zero keys.**

**The two providers are combined, not either/or.** The heuristic scorer runs
first on every headline (free) and keeps the top-K per move (default 15).
When a key is present the Anthropic provider sees only those K: it re-scores
them and writes the explanation. So model usage is about one call per move,
roughly ten per ticker, not one per headline. Final relevance with a key is
the mean of the heuristic and model scores; without a key it is the heuristic
alone. Model: `claude-sonnet-5`, fixed default in settings (never Opus).

Explanation input: decomposition numbers, regime, earnings proximity, peer
co-movement, the top-K scored articles. Output is structured:

```json
{"summary": "...", "primary_category": "company|industry|macro|unexplained",
 "confidence": 0.0-1.0, "cited_article_ids": [...], "unexplained": false}
```

**Saying it in English (`narrate.py`).** The decomposition is the answer to
"why", but a z-score and a component in percentage points are not an answer a
reader can use. One dependency-free module owns the phrasing: `|z|` becomes a
multiple of an ordinary day, the components become points of the move itself,
`near_fomc` becomes "a Federal Reserve rate decision". It is used twice — the
keyless provider renders it directly, and the keyed providers are handed the
same sentences inside the prompt, so the model weighs headlines instead of
converting factor loadings. A set of moves also gets one sentence about the
set, stated only when a claim holds for a clear majority; that synthesis is
the part a list of rows cannot do for the reader. The same ban on jargon is
asserted in the tests for the keyless half and written into
`prompts.PLAIN_ENGLISH` for the keyed half.

Cached per move in `explanations`. Explanations are computed for the top-N
moves by `abs(ret_z)` on ingest (default N=10) and on demand for any other move
— this is the cost/time control.

v2: a third provider, a Hugging Face classifier (relevance + category)
fine-tuned on the accumulated `move_articles` labels, replaces the heuristic;
the LLM stays only for the prose. It is out of v1 because torch/transformers
would make "run one command" fragile for a reviewer.

## 5. Storage

**SQLite via SQLModel** (`data/app.db`). No vector DB in v1: the join between
moves and news is a **date window**, not semantics.

Tables: `companies`, `prices`, `moves`, `articles`, `move_articles`,
`explanations`, `chat_messages`.

**Lazy population with a freshness rule**: on every read, if the newest
stored price for the ticker is older than the last completed trading day
(daily bars are final after the 4pm ET close), ingest the gap. Ingest is
idempotent on `(ticker, date)`, so only new days are inserted, only new moves
detected, and only new moves explained. So the update frequency is "on read,
at most once per trading day per ticker"; `?refresh=true` forces it. A
per-ticker `threading.Lock` stops two simultaneous first requests from both
ingesting. No intraday data in v1. v2 moves the refresh to a scheduled job
after the close so reads never pay for ingest.

## 6. API (FastAPI)

- `POST /tickers/{ticker}/ingest?period=1y&top_n=10` — explicit ingest.
- `GET /tickers/{ticker}` — prices + moves + linked news + explanations.
  Filters: `start`, `end`, `z_threshold`, `pct_threshold`, `direction`
  (`up|down`), `category`, `min_relevance`, `include_prices`, `include_news`,
  `limit`.
- `GET /tickers/{ticker}/moves/{date}` — one move with everything attached;
  computes the explanation on demand if missing.
- `POST /chat` — `{ "message", "ticker"?, "session_id"? }` → `{ "reply",
  "session_id", "tool_calls" }`. Plain JSON, no SSE. A tool-calling loop whose
  tools are the read functions above (`list_moves`, `get_move`, `search_news`).
  With no key, the heuristic provider answers from the same tools with a
  templated reply.
- `GET /` — a single static HTML chat page (fetch → `/chat`), plus Swagger at
  `/docs`. No JS framework.

## 7. Out of scope for v1 (named in v2)

Paid news APIs with full text; semantic dedupe (pgvector); scheduled
incremental ingest; multi-day move windows; executive/social commentary;
confidence calibration against a labeled move set; a Hugging Face
classifier as the local provider; implied volatility from the options chain
as an "expected move" baseline (yfinance has only the current chain, so it
cannot explain past moves anyway); learned cross-ticker relationships
(rolling correlation, lead-lag tests, hidden-state regime model); deployment
(v1 runs locally; Vercel would need the DB off disk).

## 8. Naming

The public repo and package must not contain the company name. Repo:
`stock-move-explainer`; package: `stock_moves`.
