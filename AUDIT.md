# Scientific integrity audit

## Inventory / Phase A
2026-10-11: inspected all source, tests, config, dependencies, documentation and pending patches. No nested AGENTS.md. Local HEAD and live origin/main were both `49539f1b622aff46f93371643b2e80425f149c28`; no push is authorized. Six `quant_ml/models` modules existed only locally because broad `models/` ignored source. Root-anchored ignores restore tracked source. Prior reviewers’ authorized fixes are integrated, not discarded: model modules already include frozen-pipeline cloning and fold validation; other fixes are deferred to Phase B.

`audit/legacy-sha256.txt` records every original model, result, data and holdout file before implementation. These files must remain byte-identical. They are NOT shipped by a fresh clone.

## Confirmed defects
- Missing tracked model package made clean-clone imports fail. Fixed A.
- Last-open rebalance could open and instantly liquidate positions; annualization counted entry fees as an extra elapsed day. Reviewer fix pending Phase B verification.
- Insufficient input/config/calendar validation, unstable high-fee NAV iteration and frozen refits using mutable registry defaults. Restored A; pending Phase B regression verification.
- Original model run lacks immutable raw data snapshot; old results cannot be exactly reconstructed from revised cache. Preserve, do not repair evidence.
- Holdout 2024–2026 has already been viewed. New names/seeds do not restore independence.

## Suspected risks requiring explicit checks
Sparse prediction schedules could silently skip scheduled decisions; temporal perturbation must exercise fitted predictions and orders, not features alone. Current inference could accept a uniformly stale market date. Frozen metadata must prevent code/schema drift, not merely record it. Phase B addresses these gates.

## Unresolved data/model limitations
Yahoo retrospective adjusted OHLC is a total-return research proxy, NOT historical executable quotes or point-in-time data. Present-day eight-stock universe excludes delistings and creates selection/survivorship bias. No free cached source fixes historical membership, delisting returns, timestamped revisions, variable spreads, impact, auction liquidity, taxes or borrow. Missing held prices must fail; no invented liquidation. Historical costs are assumptions, fractional adjusted units not actual shares.

No reliable predictive edge or profitability established. See RESEARCH_REPORT.md for development evidence and read-only legacy assessment.

## Phase B verified disposition
The previously pending reviewer fixes now pass 49 tests. Fixed benchmark-session
cadence errors explicitly on missing scheduled signals; dense off-cadence forecasts
are informational by contract, terminal signal is liquidation boundary only.
Integrated future-price mutation leaves earlier fitted predictions and trades
unchanged. Frozen refits clone entire pipelines; new assessment requires matching
source/dependency versions. Current refits use unique separate result directories.
Uniformly stale current market dates (>7 calendar days) fail; missing candidates
fail before fitting. This is a conservative stale guard, not an exchange calendar.

## Phase D disposition
All declared development experiments completed, every base ML acceptance gate
failed. This is a research finding, not grounds for retuning. RF 20-day paired
net-return interval versus equal-weight spans approximately -19.68 to +6.44
percentage points/year (arithmetic, not CAGR; unadjusted, not significance).
A fresh isolated clone installed the dependency lock and completed deterministic
integration plus fresh real-data bounded smoke without any old ignored artifacts.
All 38 original files and 68 newly frozen research artifacts verified by SHA-256.
Scientific limits (PTI/delistings/true executable quotes/selection) remain unresolved.
