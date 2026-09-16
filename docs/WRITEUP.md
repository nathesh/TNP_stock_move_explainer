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
