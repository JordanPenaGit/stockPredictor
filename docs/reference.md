# CLI, configuration, and troubleshooting

[README](../README.md) · [Methodology](methodology.md)

Run commands from the repository root. Artifact paths are anchored to the project
location, not a hard-coded home directory. A supplied `--config` path is resolved
from the current working directory.

## Dependencies

Install `requirements.txt` in `.venv`. The root entry points re-execute using that
environment when it exists; they otherwise use the invoking Python. Python 3.10
is the existing environment version. `requirements-optional.txt` adds CPU XGBoost;
the supplied `config.json` lists it, whereas built-in defaults do not. No GPU is
required. Remove `xgboost` from a new config if not installing it.

`requirements-lock.txt` is the captured environment, including optional packages;
it is not a promise of cross-platform bitwise reproducibility. For a compatible
environment, install it instead of the ranged requirement files:

```bash
.venv/bin/python -m pip install -r requirements-lock.txt
```

## CLI reference

Every command supports `--help` and `--run ID`. IDs contain 1–64 letters, digits,
underscores, dots, or hyphens, beginning with a letter/digit; paths are rejected.

| Command | Additional arguments | Behavior |
| --- | --- | --- |
| `python3 train.py` | `--config FILE`, `--refresh` | Validate candidate families, select and freeze a new run; does not assess the final holdout. |
| `python3 backtest.py` | None | Assess once or reuse a saved assessment; no model reselection. |
| `python3 analyze.py SYMBOL` | `--cached` | Separate latest-data refit and one-symbol analysis. |
| `python3 screen.py` | `--top N` (default 20), `--tickers SYMBOL ...`, `--cached` | Separate latest-data refit and ranked candidates. |

Training defaults to run `research-v1`; other commands resolve an omitted ID via
`models/latest.json`. Training refuses any existing model/result directory, including partial failures. Existing latest.json is preserved; use --run explicitly for new runs.
Use an explicit new ID only for an intentional new experiment. Keep the shared
holdout audit: a different ID does not undo previous test access.

Training reads exact-request cache entries unless `--refresh` is supplied.
Analyze/screen instead refresh by default and set the data end to today
(exclusive), independent of the historical config end. `--cached` suppresses
forced refresh only: a changed end date or any cache miss can still download.
It is **not an offline-only switch**.

`--tickers` overrides candidates, not the original training universe. CLI symbols
are normalized to uppercase and deduplicated. Missing fresh features cause an error
rather than a mixed-date ranking. A small universe returns fewer than `--top N`.
Analyze/screen append unique results/current/<run>/<UTC timestamp>/ records; they never rewrite frozen reports, models or test results. Uniformly stale market dates older than seven calendar days fail.

## Configuration

[`config.json`](../config.json) is an explicit example. Omitted fields use
[`Config`](../quant_ml/config.py) defaults, **not** the contents of `config.json`.
In particular, `train.py` without `--config` uses today as the exclusive end and
the three non-XGBoost families. Values below describe the supplied example.

| Field | Example | Meaning |
| --- | --- | --- |
| `universe` | AAPL, MSFT, AMZN, GOOGL, JPM, XOM, JNJ, WMT | Nonempty unique uppercase investable symbols; excludes benchmark. |
| `benchmark` | `SPY` | Reference series and trading calendar. |
| `start`, `end` | `2015-01-01`, `2026-10-04` | ISO dates; end exclusive. |
| `validation_start`, `test_start` | `2020-01-01`, `2024-01-01` | Chronological split boundaries. |
| `horizon` | `5` | Positive next-open target horizon in sessions. |
| `target_threshold` | `0.0` | Positive class when target return strictly exceeds this decimal return. |
| `train_min_dates` | `756` | Minimum unique training dates in a fold after label purging. |
| `walk_forward_dates` | `126` | Prediction sessions per refit. |
| `training_window` | `0` | Expanding history; positive values bound the training date window. |
| `rebalance_days` | `5` | Portfolio rebalance cadence, independent of target horizon. |
| `starting_capital` | `100000.0` | Initial simulated cash. |
| `max_positions`, `max_weight` | `3`, `0.34` | Holdings limit and per-symbol NAV cap; remainder stays cash. |
| `sizing` | `equal` | `equal` or `inverse_volatility`. |
| `commission_bps`, `slippage_bps` | `5.0`, `5.0` | Additive costs per buy/sell side; one basis point = 0.01%. |
| `strategy` | `ranking` | `ranking` or `probability`; both rank eligible stocks by predicted return. |
| `return_threshold` | `0.0` | Minimum predicted decimal return in ranking mode. |
| `probability_threshold` | `0.55` | Minimum probability in probability mode. |
| `feature_set` | `all` | `all`, `technical`, `statistical`, or `market`; technical features are always included. |
| `select_features` | `24` | Number selected inside each training pipeline, capped by available features. |
| `seed` | `42` | Estimator and importance random seed. |
| `models` | `linear`, `random_forest`, `gradient_boosting`, `xgboost` | Candidate regression/classification pairs; XGBoost is optional. |

Require `start < validation_start < test_start < end`, enough history for the
200-session feature warmup and label purge, and enough training dates per fold.
A rolling window smaller than `train_min_dates` cannot satisfy that requirement.
Rates/returns use decimals, not percentages, except the explicit basis-point fields.
Choose all settings before holdout assessment; changing JSON does not alter a
frozen run’s saved configuration.

## Troubleshooting

- **Missing model / no latest run:** select an existing `--run` explicitly, or
  train an intentional new protocol. Do not overwrite `research-v1`.
- **Run already frozen:** reuse it. For a genuinely different experiment choose
  a fresh ID; later work remains exploratory after a viewed holdout.
- **Assessment previously started but incomplete:** preserve the audit and partial
  outputs, investigate the original error, and arrange explicit recovery. There
  is no CLI recovery flag. Do not delete evidence to force another test pass.
- **Market data changed since freeze:** do not replace the recorded hash to silence
  the error. New runs assess their saved snapshot; legacy artifacts can lack one.
  See the [research-v1 revision note](../results/research-v1/DATA_REVISION_NOTE.md).
- **Yahoo/network error:** check the symbol, dates, connectivity, and provider
  availability. Retry later if rate-limited. Cache reuse requires an exact request;
  missing symbols are not silently dropped or replaced by synthetic data.
- **Insufficient training history / no fresh features:** check date coverage,
  warmup, matured labels, and window size. Do not backfill future observations.
- **Missing held-price valuation:** the simulation intentionally stops; inspect
  source data rather than inventing fills for halts or delistings.
- **Missing XGBoost:** install `requirements-optional.txt` or remove that family
  from a new config. Do not silently change an existing frozen protocol.
- **Fewer screen rows than requested:** `--top` is a limit, not a universe expansion.
- **Backtest prints only a reuse message:** open `results/<run>/report.md` or
  `assessment.json`; the existing assessment is not automatically recomputed.

## Tests and provenance

The [README test commands](../README.md#tests) run deterministic synthetic fixtures
for feature causality/math, target shifts, global-date purging and rolling windows,
training-only transforms, cache roundtrips, next-open timing, missing prices,
self-financing accounting, two-sided costs, drifted turnover, weights/cash, and
benchmark alignment. Synthetic fixtures are never substituted for market results.

[`SPEC.md`](../SPEC.md) preserves the supplied brief. [`PROGRESS.md`](../PROGRESS.md)
and `results/foundation*` record the early baseline/data/target/backtester stages;
they are historical evidence, not a current status dashboard. Existing command
logs include `results/tests.log` and `results/foundation-tests.log`. Run the tests
again to validate your current checkout rather than assuming old logs still apply.

## Development-only command
python3 research.py --output results/NEW-ID downloads only 2015–2023 data.
--cache data/prices.sqlite reads the existing exact-request cache without writing.
--snapshot /trusted/path/development_snapshot.joblib explicitly imports archived
data; these ignored artifacts are NOT bundled by git. Never load untrusted pickle.
--smoke uses three validation months and linear only: workflow verification, not
full research or independent evidence. Output must not exist. Failures preserve
frozen_protocol.json and FAILED.json; no silent retry or experiment exclusion.
Outputs include summary, matched curves/orders/lots/predictions, fold audits,
source/dependency/data hashes and a development-only snapshot.


## Forward shadow CLI
python3 forward.py init --root results/shadow-NEW-ID freezes a new fixed protocol.
Then explicitly call capture after close+30min and before next open. evaluate
requires --signal-date YYYY-MM-DD and refuses immature targets. No --asof/backdate
flag exists. Inputs, refits, forecasts and matured outcomes are separate immutable
records; failed attempts are preserved. There is no scheduler, broker, trade or
net-portfolio-profit claim. See ROADMAP.md for review timing and evidence limits.
