# Verification report

## Phase A — packaging gate
- Initial commit `d8e19e3` restored six model modules and root-anchored ignores. Clean clone exposed a real dependent import failure (`validate_features`), not visible in the working tree. Follow-up `75e2de0` integrated workflow/utils API fixes.
- Actual local clone: `git clone --no-hardlinks . /tmp/stock-scientific-phase-a`; isolated `python3 -m venv .venv`; `pip install -r requirements-lock.txt` succeeded (Python 3.10).
- Clean tracked checkout: model predict import, 19 original unittest tests, four CLI help commands, and pip check passed. Working tree reviewer regression suite separately passed 44 tests. The clone contains NO old ignored market/model/results artifacts.

## Phase B — critical scientific gate
- `.venv/bin/python -m unittest discover -s tests -v`: **49 passed**. Deterministic fixture warnings for constant features are expected; not market results.
- Added full input->features->purged walk-forward->prediction->trade future perturbation test; predictions and earlier orders unchanged. Added dense/sparse cadence equivalence, explicit missing-due failure, matched costed benchmark accounting, missing-held-price failure, and train-only imputation/scaling/selection statistics tests.
- Integrated reviewer tests cover labels, purging, costs, self-financing, terminal liquidation, elapsed-open annualization, schema/refit drift, config/CLI failures, stale candidate rejection, and artifact ignore policy.
- `sha256sum -c audit/legacy-sha256.txt`: all original files unchanged.
- New current refits append separate outputs, never overwrite frozen results; new assessments reject source/dependency drift. Data invalid rows fail rather than silently repair.
- Limitation: no test converts retrospective Yahoo adjustments into point-in-time information; historical execution remains an explicit proxy. No real trade or broker interaction.

Phase commit identifiers: `git log --oneline --grep='Phase '`. Later phase gates are appended after execution.

## Phase C — baseline correctness gate
Full unittest discovery: **52 passed**. Added explicit same-universe equal-weight
buy-and-hold with drift and analytically verified both-side fees; all baselines
share dates. Historical means/up-rates use only matured purged fold training;
future-label perturbations leave earlier baselines unchanged. Naive zero/.5
forecast has a cash allocation comparator (not arbitrary alphabetical tie trades).
Momentum60 and SMA200 are ordinal signals, not calibrated forecasts. SPY/full
same-universe buy-hold differ in concentration/exposure from capped top-three
strategies; fair dates/cost rules do not imply matched risk. Gate passed before
any new real-data performance was inspected.

## Phase D pre-execution gate
56 tests passed, including deterministic end-to-end fixture twice, output collision
refusal, preserved failed-attempt evidence, bound/calendar data validation and
seeded block-bootstrap tests. Initial fixture equality exposed unnormalized NaN
values for empty regime slices; fixed JSON-clean return consistency BEFORE market
experiments. No model/period adjustment. arch 7.2.0 reused rather than implementing
heavy custom resampling; dependencies installed and lock recaptured.

## Phase D — actual market execution / critical gate
- Full plan frozen in c25ce83 BEFORE any new performance; plan SHA-256 verified.
- Executed: .venv/bin/python research.py --cache data/prices.sqlite --output results/scientific-dev-20261011
- Completed all 3 families + 5 declared variants + baselines, all costs/year/regime/
  single-stock allocation/trade concentration/window/block diagnostics. No failed
  experiment, no parameter/period changes, no selection. 1,000 common signal dates,
  999 elapsed opens (2020-01-03 through 2023-12-21). Data ends 2023-12-29; target
  maturity trims the last six signals. All 3 model acceptance gates FAIL.
- Observed CAGR: linear -3.03%, RF 7.68%, HGB 8.26%; SPY 11.87%, equal-weight
  buyhold 16.24%, historical mean 23.54%, momentum 19.62%, trend 20.65%. These
  simple-baseline historical successes are not newly independent evidence.
- Verified all 68 new artifact hashes; all 38 original files byte-identical.
- Actual second clone /tmp/stock-scientific-final at c25ce83: installed locked
  dependencies into NEW .venv, 56 tests pass (1.8 s), compileall and pip check pass.
  All five CLI helps exit 0; missing run exits 1; invalid top/ticker exit 2.
- Clean clone had NO old data/models/results. Executed research.py --smoke --output
  results/clean-download-smoke with fresh real Yahoo 2015–2023 downloads and no
  cache/snapshot arguments. Completed bounded first-quarter-2020 linear/baseline
  smoke with negative gates. This proves workflow operation, not bitwise vendor
  reproducibility or independent performance. Ignored artifacts are not in git.
- Reports retain poor predictive metrics and wide unadjusted intervals. No legacy
  holdout retraining/reassessment. Legacy saved metrics read descriptively only.

## Phase E — future evidence / final scientific gate
- Full suite: **62 passed** (3.249 s); compileall, pip check and git diff --check pass.
- Added NYSE holiday/early-close, after-close +30min / next-open deadline,
  no pre-freeze capture, actual matured-outcome integration, duplicate/tamper/code
  drift refusal tests. Calendar implementation reused maintained package; no
  custom holiday calendar. Fixture outcomes are NOT future market evidence.
- Added real train->freeze->assess->reuse integration on temporary deterministic
  fixtures: bundle byte hash, preserved legacy audit/pointer, new append-only
  started/completed events. Historical assessments always exploratory; tracked
  audit/holdout-status.json prevents a fresh-clone absence being mistaken for new
  independence. Current source-incompatible legacy inference refuses instead of
  silently changing feature definitions. Global stale/future bars and provider
  date bounds/duplicates/invalid values explicitly checked.
- Frozen legacy data/models/results/holdout files unchanged; no model or research
  parameter changed because of performance. Prospective collection is manual,
  no scheduler, no broker, no orders, no promised future return. See ROADMAP.md.
- New code after c25ce83 is hardening/forward collection; the full development
  result remains bound to c25ce83 and its recorded lock/source hashes, not silently
  regenerated using later source. Exact replay can checkout that local commit
  and explicitly supply its preserved snapshot; current clean-clone workflow is
  separately verified below.
