# Submission write-up

`stock-move-explainer` is a local FastAPI + SQLite service that pulls a ticker's daily prices,
flags the major days by a rolling z-score, splits each one into market, sector and
idiosyncratic components, and uses that split to route a dated news query whose headlines a
provider scores and turns into a cited explanation. It runs end to end with no API keys,
because a keyless news source and a heuristic scorer/explainer sit behind the same two
interfaces that a paid feed and a language model would.

## Briefly describe your process for completing the project from start to finish. Include details on assumptions made and key decisions taken / tradeoffs made for each step (if relevant).

I probed the keyless sources first. Google News RSS gives 100 dated, sourced items a query
with `after:`/`before:` and no throttle; GDELT allows one request per five seconds. So RSS is
primary, GDELT the fallback behind one protocol, and scoring is headline-only. I then wrote
`DESIGN.md` as the single source of truth and a 16-task DAG with fixed signatures. Coding
agents ran the tasks in waves, each confined to its own files, no network in tests. I reviewed
every report, corrected the DAG against DESIGN (Sonnet not Opus; a wrong repo path), and made
the reconciliation calls: 15 headlines re-scored by the model, 8 to the explainer.

Key decisions: a move is `abs(ret_z) >= 2` on a trailing 20-day std, since 2% is big for a
utility and quiet for a small cap, with 2% OR-ed in because the prompt suggested it, both
query filters and not ingest constants; the 60-day OLS on SPY and the sector ETF routes the
news query rather than being the answer (attribution, not causation); SQLite and no vector
store, since the join is a date window; lazy population with a freshness rule and per-ticker
locks; and top-N explanations at ingest, else on demand, as the cost control.

## Are you happy with your solution? Why or why not?

Mostly. The pipeline is real end to end on live data with no keys: a 1-year AAPL run gives 251
prices, 43 moves, 150 articles and 5 explanations in about 2.4 seconds. The output is
structured, with citations, a confidence and an explicit `unexplained`, and the system is
honest about degradation, so a result written by the fallback is labelled as the fallback.
Every external dependency is behind an interface, and the tests run with no network at all
against synthetic price series with an injected shock.

What I am not happy about came out of hand-checking three explanations. Two are well grounded:
2026-07-31 (-7.4%, an earnings day) and 2026-06-25 (-6.1%) cite same-day headlines that name
the cause, and the decomposition agrees. The third, 2026-01-20 (-3.5%, z of -5.2), is dated
right but cites generic bullish Apple commentary, because the heuristic scorer rewards entity
mentions rather than event content, and its 0.85 confidence is overstated for that evidence.
The hedged wording stops it asserting a false cause, but confidence is uncalibrated, scoring
sees headlines only, and the model path went unproven live in this window.

## What would you do differently if you got to do this over again?

First, I would smoke-test the live model path in wave one with a single one-line call, so a bad
key surfaced at minute 10 instead of minute 150. That costs nothing and would have bought back
more time than anything else here. Second, I would make the heuristic confidence depend on
whether a cited headline contains an event verb (falls, cuts, beats, misses, recall, lawsuit)
rather than only on the count of entity matches; that is the defect behind the overstated
2026-01-20 confidence. Third, I would ingest two years of history by default so regime labels
fill the whole 1-year window. Fourth, I would hand the model the peer co-movement and the
article list in one call from the start, rather than arriving at two top-K knobs by
reconciliation.

Past that, `docs/architecture-v2.md` holds what I would build with more than four hours: paid
full-text news, a small classifier fine-tuned on the accumulated `move_articles` labels in
place of the heuristic scorer, calibration against a labelled move set, and scheduled ingest
after the close.

## Did you get stuck anywhere? How'd you get unstuck?

The costly one was the model path. The API key in my environment was rejected with a 401 during
integration, so the Anthropic provider is wired and unit-tested against a fake client but never
proven against the live API. While debugging it I found two honesty bugs: the provider degraded
silently to the heuristic, and the cached row was still stamped provider `anthropic`. I fixed
the labelling so a degraded result names the provider that actually wrote it, which is also why
the keyed run takes about 6 seconds rather than 2.4: every model call fails and falls back. I
could not fix the key, so I documented it rather than claim a live result.

Three smaller ones. A pydantic/SQLModel clash, where a column named `date` annotated with the
`date` type raises `PydanticUserError`, solved with a module-level alias. `moves.ret_z` is NOT
NULL but NaN during the 20-day warm-up for days caught by the percentage gate, so I store 0.0
and those days sort last. Ruff here enforced DTZ and BLE rules absent from `pyproject.toml`, so
I pinned the config to make lint reproducible. One I left: regime labels are null over most of
a 1-year window because the 50/200-day SMA needs 125-plus rows, so it is documented.

## Known limitations

- Attribution, not causation: the explanation says what the news said on the day, not what moved the price.
- Both news sources give headlines only, so scoring cannot see article bodies, and small-cap coverage is thin.
- RSS urls are Google redirects, so one story reached through two redirect urls is stored twice; GDELT is throttled to one request per five seconds.
- The 60-day OLS is a rough factor model; betas are noisy on volatile names and near regime changes.
- Sector ETF mapping is a static dict, so conglomerates and misclassified sectors route badly.
- Peers exist without a key, from the sector ETF's top holdings, so peer co-movement always runs; what a key adds is model-suggested peers, which are closer competitors than an index's largest weights.
- Confidence is uncalibrated, and `unexplained` reflects the provider's opinion rather than a measured error rate.
- Single-day moves only, no scheduled ingest (data goes stale until `?refresh=true`), and SQLite on local disk is not deployable as-is to a serverless host.
