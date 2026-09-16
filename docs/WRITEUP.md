# Submission write-up

Repo: <https://github.com/nathesh/TNP_stock_move_explainer>
Live app: <https://stock-move-explainer.vercel.app> (chat UI at `/`, Swagger at `/docs`)

## Briefly describe your process for completing the project from start to finish. Include details on assumptions made and key decisions taken / tradeoffs made for each step (if relevant).

I split the work into V1, V1.5 and V2 and wrote each one down before coding (`DESIGN.md` and
`docs/`). V1 is what I am submitting: daily prices from yfinance, major moves flagged by rolling
z-score, each move split into market / sector / company components, that split used to route a
dated news query (Google News RSS, GDELT fallback), headlines scored and turned into a cited
explanation with a confidence, all behind a FastAPI API with a chat endpoint. Tested and deployed
on Vercel. V1.5 (competitors, suppliers, customers, countries, and a gate for geopolitical news)
is planned in `docs/v1.5-plan.md` but not built. V2 is the production architecture in
`docs/architecture-v2.md`.

Key assumptions: "why" is attribution, not causation, so the model may answer `unexplained`.
"Major" is `abs(z) >= 2` on trailing 20-day volatility, not a fixed percent, with the 2% option
kept as a filter. Daily bars only. Keyless data only, so scoring is on headlines, not bodies.
SQLite, no vector store, since moves and news join on a date window. It runs with zero keys on a
heuristic provider, and an OpenAI key upgrades scoring, prose and chat.

## Are you happy with your solution? Why or why not?

Kind of. V1 works end to end on live data with no keys, and it is deployed. But it is simple. I
was hoping to get to V1.5, which adds the relationships I wanted on geopolitics, customers and
competitors, and I did not get there.

## What would you do differently if you got to do this over again?

Three things. I would use only OpenAI. I built the provider layer for both OpenAI and Anthropic,
with tests for each, which makes sense in production but was overkill here. I would add evals.
There are none, so I cannot say how often the explanation is right. And I would use trained
models where possible, for example a RoBERTa-style classifier for headline relevance and
category instead of keyword rules. None of that was done, and the overkill elsewhere is what
cost me the time.

## Did you get stuck anywhere? How'd you get unstuck?

Yes. When V1 was first done it used no model for scoring or explanations, and the output was not
clean. Wiring in OpenAI meant rewriting the output structures and the prompts, which took longer
than planned. I kept the heuristic path as the fallback and added the model behind the same
interface, so nothing broke while I switched. That is the time that would have gone to V1.5.

## v1.5 addendum

The answers above are the submission as it stood; v1.5 is the extension built after it, so
where they say v1.5 was planned but not built, this section is what changed. v1.5 adds the
relationship layer from `docs/v1.5-plan.md`: one `company_edges` table of typed facts per
company (competitor, supplier, customer, country, and price-derived factor betas), a
`geo_events` table of dated geopolitical headline counts per country, and exactly two uses
for them — a `sub_routing` on every move (`share_shift`, `supply_chain`, `oil`, `dollar`,
`rates`, `gold`, `country:XX`) beside the unchanged `company|industry|macro` routing, and a
news query that expands along the edges when one of those fires. One gate rule caps a
geopolitical headline at relevance `0.30` unless the company has the country edge *and* the
day's `macro_driver` agrees, so a tariff story is never credited on vocabulary alone. The
API keeps its shape and gains one read, `GET /tickers/{ticker}/relations`, plus a
`get_relations` chat tool.

The three fixes I said I would make came first, before any of that. The live model path is
now proven rather than assumed: `scripts/smoke_model.py` makes one call through the app's
own provider path and one raw SDK call, because the providers degrade to the heuristic on
any exception and a dead key otherwise looks like a quiet answer. The default period is two
years instead of one, so the 200-day SMA behind the regime labels has warmed up rather than
leaving the first months null. And the keyless confidence now requires an event verb in a
cited headline, so "the company was mentioned" can no longer read as high confidence.

There is also an eval now, three ticker-days fixed in the plan before the build:
`scripts/eval_v15.py` ingests each ticker once and judges `routing` and `sub_routing`
separately, exiting non-zero on any mismatch. It uses the network and is not part of pytest.

Run on 2026-09-15 with no working key, so every explanation below is the heuristic's:

- NVDA 2025-04-16 — routing: `industry`, sub_routing: none, expected `company` or `macro`
  with `supply_chain` / `country:CN` / `country:TW`. FAIL, on both halves. The day was
  -6.9% at z -1.2, so it entered on the 2% gate as the plan said it would, but the whole
  semi complex fell with it and the OLS put it in the sector. The sub-bucket it was meant
  to land in needs a supplier edge to TSM or a country edge to CN or TW, and both of those
  are model-suggested, so keyless there was nothing for the rule to read.
- NVDA 2025-01-27 — routing: `industry`, sub_routing: none, expected `industry` / none.
  PASS, on both halves. -17.0% at z -5.8 with rivals down 6.0% alongside it, which is the
  case the plan wanted: rivals moving *with* the stock yields `industry`, not `share_shift`.
  The cited headlines are the DeepSeek rout, led by CNBC's $600B market-cap line.
- AAPL 2025-04-03 — routing: `macro`, sub_routing: `oil`, expected `country:CN` or
  `dollar`. PASS on the routing, FAIL on the sub-bucket. `country:CN` was unreachable with
  no country edge; `dollar` was reachable and lost, because oil's contribution was the
  largest among the proxies that cleared `abs(z) >= 1.5` that day. That is decision 4 doing
  exactly what it says, and pointing at the wrong story: the day was the tariff
  announcement, and the prose says "macro, via oil".
- Overall: 1/3 passed, exit code 1, run on 2026-09-15 with the heuristic provider.

Two of the three misses are the keyless limitation stated below rather than broken code:
`supply_chain` and `country:XX` cannot fire without a working key, so the ceiling on this
eval keyless is 2/3, and the NVDA 2025-04-16 row cannot be scored at all. One miss is not:
`oil` beating `dollar` on the AAPL tariff day is the univariate attribution picking the
loudest proxy rather than the right one, and a key would not change it. The eval is
therefore half-run until it is run with a key, and the row it would still get wrong is the
one worth arguing about.

What v1.5 does not fix, stated as plainly as the limitations above:

- Factor attribution is univariate, one trailing 60-day beta per proxy, and the driver is
  the largest contribution among proxies whose own return that day cleared `abs(z) >= 1.5`.
  It is a heuristic for pointing at a story, not a joint factor model, and correlated
  proxies (oil and the dollar) will both claim the same move.
- Geopolitical events are headline counts, not a structured event feed. GDELT's Events
  files are 96 a day and its DOC API cannot filter by CAMEO code, so a "geo event" is a
  count of matching headlines per country per day with a few sample titles.
- Country edges are model-suggested, so without a working key there are none, the geo gate
  never opens, and no `country:XX` sub-routing can fire. The keyless provider returns
  competitors from the sector ETF's holdings and nothing else.
- The relations call happens at ingest, one model call per company per run, and
  `build_edges` replaces that company's edges wholesale each time. So the cost scales with
  ingests rather than with companies, and the edges are only as fresh as the last ingest of
  that ticker: a supplier relationship that ends, or a country exposure that starts, is not
  noticed until the ticker is ingested again. There is no scheduled refresh.
