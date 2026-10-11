# Methodology and interpretation

[README](../README.md) · [CLI and configuration](reference.md)

## Architecture and data

- `quant_ml/data/`: provider protocol, Yahoo downloader, SQLite exact-request
  cache. Requests store raw/adjusted OHLCV, dividends, and splits. Cache keys
  include provider/adjustment schema, symbol, start, and exclusive end; the
  downloader uses daily bars. Explicit refresh replaces that request snapshot.
  A failing universe member raises an error rather than silently changing the experiment.
- `quant_ml/features/`: technical, statistical, and market-relative modules.
  Features include returns, RSI/MACD, normalized SMA10/20/50/100/200 and EMA
  distances, crossovers, ATR/volatility/downside risk, volume ratios, market
  excess return/beta/correlation/relative strength, distribution statistics, and
  calendar signals. Features discard the first 200 sessions; missing market
  observations are not backfilled. Model pipelines separately impute feature gaps.
- `quant_ml/targets.py`: future return, threshold classification, entry and
  label-end dates. Dataset percentile ranks are available, but screening ranks
  expected-return regressors, not a dedicated learning-to-rank model.
- `quant_ml/models/`: Ridge/logistic baselines, random forest, histogram gradient
  boosting, optional CPU XGBoost; training-only median imputation, scaling, and
  feature selection. Fixed estimator settings live in `models/registry.py`.
- `quant_ml/backtest/`: strategy construction and a self-financing cash/adjusted-share ledger.
- `quant_ml/research/`: Markdown reports and headless research/calibration charts.
- `quant_ml/workflow.py`: validation-only selection, frozen metadata, data hashes,
  one-time historical assessment, and append-only holdout-access events.

## Timing and leakage contract

A feature row dated **t is available after close(t)**. Earliest execution is
**open(t+1)**. For horizon h (for example 1, 5, or 20 sessions):

```text
target_return = adjusted_open(t+h+1) / adjusted_open(t+1) - 1
target_up = target_return > target_threshold
```

This is not a close(t)-to-close(t+h) return. Dates follow the benchmark calendar.
The target horizon does not force each portfolio holding to be closed after h
sessions: `rebalance_days` independently controls portfolio decisions.

Split boundaries use global unique dates, never flat panel row counts. Training
requires **target_end < prediction cutoff**; equality is purged. Expanding folds
are the default; `training_window` bounds the training date window.
`walk_forward_dates` sets retraining cadence. Each fold’s unchanged estimator
predicts later dates from then-available close features. All imputation, scaling,
and feature selection fit only the fold’s training rows. Histogram boosting
sets `early_stopping=False` to avoid internal random validation splits.

## Selection, holdout, and current inference

The supplied protocol uses history from 2015, validation in 2020–2023, and a test
period beginning in 2024. Model family selection uses **validation strategy
Sharpe after fixed costs**, with validation RMSE as tie-break. Validation labels
and fills crossing into the test interval are excluded. Feature set, horizon,
thresholds, costs, sizing, and hyperparameters must be fixed before testing.

Config/data hashes, seeds, versions, hyperparameters, and fold dates are persisted
before final assessment. New runs also save `market_snapshot.joblib`, and
assessment verifies its hash. The original `research-v1` predates that snapshot
fix; [its revision note](../results/research-v1/DATA_REVISION_NOTE.md) explains why
exact refitting from the current raw cache is not guaranteed.

Final assessment keeps the selected protocol but uses causal walk-forward
retraining: earlier test outcomes can enter training only once their labels
mature. This is a predeclared adaptive workflow, not test-driven reselection.
Completed assessments are reused, not recomputed. A started but incomplete audit
blocks silent retries; preserve partial outputs and investigate first.

Once a holdout has been viewed, later experiments are exploratory—even with new
run names. The local JSON audit is a guardrail, not tamper-proof governance.
Deleting it does not restore independence. Do not retune on test results and
present the same test interval as untouched.

Analyze/screen perform a separate latest-data refit of the selected family on
mature labels from the original training universe. They do not overwrite the
frozen evaluated estimator. Requested additional tickers are candidates, not
extra training stocks. Current refits append unique records, never rewriting frozen reports or results.

The displayed raw price is Yahoo’s latest daily close, not a live quote.
Probabilities are **uncalibrated model estimates** of the label event; calibration
is evaluated out of sample, not applied as a fitted probability correction.
No confidence percentage, calibrated interval, or profit guarantee is supplied.

## Portfolio and accounting

Long only, fractional adjusted-share research units, no leverage. Ranking mode
filters on expected return; probability mode filters on classification probability.
Both rank remaining candidates by predicted return and take up to `max_positions`.
Equal or inverse-volatility weights are capped at `max_weight`; unused allocation
stays cash rather than being silently redistributed.

Turnover uses actual drifted holdings, not target-weight differences. Target
post-cost NAV is solved so fees do not create hidden borrowing:

```text
cash_after = cash_before - sum(delta_quantity * open_price) - fees
cost_rate = (commission_bps + slippage_bps) / 10000
```

Costs apply per side, including initial allocation and final-open liquidation.
Slippage is a proportional cash debit, not a perturbation to quote prices.
Missing new-entry opens do not fill; a missing valuation for an existing holding
halts the simulation instead of inventing a price or liquidation. Production
delisting/market-halt handling requires better data.

SPY (or the configured benchmark) and buy-and-hold of every same-universe stock
use matched dates, costs, and adjusted-open execution. Dividend adjustment is
included once; dividends are not credited separately as cash.

## Reports and metrics

`results/<run>/report.md` summarizes the selected model and hyperparameters,
prediction quality, strategy/benchmark performance, risk, annual/regime slices,
and fixed cost-sensitivity diagnostics. CSVs contain predictions, orders, and
partial closed lots. `research.png` and `calibration.png` are generated headlessly.

- Portfolio: total return; CAGR using 252 sessions/year; sample annualized
  volatility; zero-risk-free Sharpe; downside-RMS Sortino; maximum drawdown
  including initial costs; Calmar; exposure; order count; gross one-way turnover.
- Closed lots: win rate, profit factor, and average return. Partial sales use
  weighted-average cost including entry fees; these are not independent completed
  positions. Profit factor is null if there are no losses (undefined).
- Regression: MAE, RMSE, R², sign accuracy, and correlation.
- Classification: accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
  Brier score, and calibration. Compare the majority baseline, after-cost returns,
  and risk; accuracy above 50% alone does not establish economic value.

Overlapping targets invalidate naive independent-sample significance claims.
The development runner adds date-block descriptive intervals; no multiplicity-adjusted significance or calibrated prediction intervals are asserted.
Native importance may be unavailable for histogram boosting; permutation
importance is produced during assessment. Correlated signals and temporal
dependence limit interpretation. SHAP is not included.

## Data and safe-use limitations

Yahoo adjusted history can be revised retrospectively. Consistently adjusted
OHLC avoids splitting/dividend accounting distortions but is **not actual
historical executable quotes or a point-in-time guarantee**. Data hashes and
snapshots preserve a research input, not historical data availability.

The small present-day universe has selection and survivorship bias; there is no
delisted-stock or historical-constituent database. Market impact, changing spreads,
taxes, complete calendar/currency support, and production monitoring are absent.
The example US stocks are treated as USD; other instruments need explicit
validation. Provider data can be wrong or unavailable, and data licenses/terms
still apply. Preserve and report negative performance honestly.

Educational research only, not investment advice. There are no broker credentials,
automatic orders, paid APIs, or scheduled jobs in this pipeline.

## References

- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html): training-only transforms.
- [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html): chronological principles; this project adds panel-date/label-end purging.
- [yfinance reference](https://ranaroussi.github.io/yfinance/reference/index.html): market data provider.

## Scientific mission additions (2026-10-11)
research.py is the preferred development-only path. The plan in
audit/development-plan.json freezes dates, families, neighborhoods, costs and
gates before performance. Cached extraction is sliced before 2024 before modeling;
adjustment factors remain retrospective. Snapshots preserve identity, not PTI truth.

Cadence is anchored at the first signal's next benchmark open. Dense off-cadence
forecasts intentionally do not trade. Missing due signals fail; sparse exact
cadence is equivalent. The final signal marks next-open liquidation, not a new
zero-duration investment. All controls share dates and both-side costs. Full SPY
and equal-weight buyhold differ in risk from capped top-three ML strategies.
Slippage is a per-side cash haircut for spread plus execution shortfall; no
separate spread charge is added. 0/10/25/50-bps sensitivity does NOT validate
those assumptions or model time-varying liquidity, auction impact or taxes.

Historical mean forecasts use ticker-specific matured training labels at identical
cutoffs. Zero/.5 forecasts are the no-skill metric control; cash at zero interest
is its economic control, not arbitrary alphabetical tie portfolios. Momentum60
and SMA200 are ordinal allocation signals, not calibrated forecasts. Maintained
arch circular-block bootstrap samples dates, not panel rows; fixed block
sensitivity and unadjusted intervals are descriptive only. See
AUDIT.md, RESEARCH_REPORT.md and audit/uncertainty-method.md for remaining limits.


## Prospective collection and immutable history
The tracked audit/holdout-status.json carries already-viewed status into clones.
Historical assessment is always marked exploratory, even when ignored local audit
files are absent. New started/completed access records are separate exclusive JSON
files under models/holdout_access; old holdout_audit.json is read-only. Existing
latest.json pointers are not replaced; specify --run explicitly for new artifacts.
New bundles have byte hashes checked before load. Current inference rejects
source/dependency drift instead of silently changing frozen feature definitions.
See ROADMAP.md for actual after-close collection and matured-label evaluation.
No future performance has yet been collected. The calendar is maintained
pandas-market-calendars 5.5.0; its NYSE schedule handles planned holidays/early
closes, not a guarantee against unforeseen exchange interruptions.
