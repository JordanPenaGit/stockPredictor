# Quant ML Research

A complete local, reproducible daily-stock research pipeline. **No brokerage integration or real trading. No profitability promise.** The default small present-day universe is intentionally manageable and **survivorship-biased**, not a historical S&P 500 membership dataset.

## Quick start

Python 3.10+ (tested 3.10). From this directory:


default installation:

a) python3 -m venv .venv
b) .venv/bin/python -m pip install -r requirements.txt
c) optional CPU boosting: .venv/bin/python -m pip install xgboost-cpu

The four root scripts automatically use this project's .venv, so no activation is required:

    python3 train.py --config config.json
    python3 backtest.py
    python3 analyze.py AAPL
    python3 screen.py --top 20

The installed/tested environment and completed research run already exist here. **Do not retrain over the existing frozen run.** Re-running backtest prints the existing assessment without reselection. For an intentional new protocol, use --run a-unique-name; after a holdout has been viewed, later runs are explicitly marked exploratory. Merely picking a new run name does not create an untouched test.

Use --help for every command. --run selects an artifact. analyze/screen update real Yahoo data by default; --cached avoids a forced refresh if an exact matching query exists. screen --tickers AAPL MSFT overrides the screening universe but not the model's training universe. The default universe has eight stocks: --top 20 returns up to eight, not twenty fabricated candidates.

## Architecture

- quant_ml/data: provider protocol, Yahoo implementation, SQLite exact-request cache. Downloads raw and adjusted OHLCV, dividends and splits. Cache keys include symbol, interval and adjustment schema; explicit refresh replaces that snapshot. All failing universe members raise errors instead of silently changing the experiment.
- quant_ml/features: separate technical/statistical/market modules, composed without future information. Returns, RSI/MACD, normalized SMA10/20/50/100/200 and EMA distances, crossovers, ATR/volatility/downside risk, volume features, market excess/beta/correlation/relative strength, rolling distribution statistics and calendar signals. Scale-free moving-average/volume ratios are used rather than raw nominal levels. 200-session warmup; missing observations are not backfilled.
- quant_ml/targets.py: regression future return, threshold classification, explicit entry and label-end dates. Dataset also includes cross-sectional percentile target ranks; stock ranking uses expected-return regressors (not a dedicated learning-to-rank loss).
- quant_ml/models: sklearn pipelines, baseline linear/logistic models, RF, histogram gradient boosting, optional XGBoost CPU. Training-only median imputation, scaling and feature selection. Predictive metrics/calibration and native/permutation importance.
- quant_ml/backtest: model-independent strategy construction and self-financing cash/adjusted-share ledger with actual position drift, fees on all buys and sells, and liquidation at the final open. No real trades.
- quant_ml/research: Markdown report plus equity/drawdown/risk/prediction/importance/calibration PNGs.
- quant_ml/workflow.py: development-only selection, immutable model manifest, one-time final assessment, data fingerprint and local holdout access audit.
- data/: SQLite market cache; models/: frozen joblib and manifest; results/: CSV, JSON, logs and reports.

## Timing and leakage contract

A feature row dated t is available **after close(t)**. An order based on it can execute no earlier than **open(t+1)**. For horizon h, target = adjusted_open(t+h+1) / adjusted_open(t+1) - 1. h=1,5,20 are all supported. This differs intentionally from a non-executable close(t)-to-close(t+h) target. target_threshold controls the classification event, while probability_threshold controls strategy construction; do not confuse them.

All split boundaries are global unique dates, not flat panel row counts. A training sample is admissible only when **target_end < prediction cutoff**. Equality is purged conservatively. Expanding folds are default; training_window limits the training date window. walk_forward_dates controls retraining cadence. Models are not refit every day by default; each fold's unchanged estimator predicts later dates from their then-available close features. Its label data is strictly older than the fold cutoff. All transformations/selection are fitted only on that fold's training rows. The histogram booster has early_stopping=False to avoid internal random validation splits.

Default dates: history 2015 onward, validation 2020–2023, test 2024 onward. Model family selection uses ONLY validation strategy Sharpe after fixed costs, with validation RMSE tie-break. The feature set, horizon, thresholds, costs, sizing and fixed hyperparameters are chosen before test. Validation labels/fills crossing into the test interval are removed. Config and data hashes, seeds, versions, hyperparameters and fold dates are persisted before final testing. Final test uses a frozen protocol with causal walk-forward retraining; prior test outcomes become training data only after labels mature, as they would in a predeclared live workflow. No test-driven reselection occurs.

Current analysis makes a separate, latest-data refit of the selected family on fully matured labels; it never replaces the evaluated frozen model. It reports a model estimate, not a guaranteed outcome. The displayed raw close is Yahoo's latest daily close, not a streaming quote. Confidence is deliberately **not fabricated**; probability calibration is assessed in the final test report, and probabilities are labeled uncalibrated.

## Portfolio/accounting conventions

Long only, fractional adjusted-share research units, no leverage. At each rebalance, a ranking threshold or probability threshold filters candidates; top max_positions are selected. Equal or inverse-volatility weights are capped at max_weight; remainder stays cash. Turnover is based on actual drifted holdings, not target-weight differences. Target post-cost NAV is solved so fees do not create hidden borrowing. Cash equation: cash_after = cash_before - sum(delta_quantity * open_price) - fees. Aggregate cost_rate = (commission_bps + slippage_bps)/10000 applied per side, including first allocation and final liquidation. Slippage is an equivalent proportional cash debit rather than a quote-price perturbation.

Orders with a missing new-entry open do not fill. Missing valuation of an existing holding halts the run instead of forward-filling a trade or inventing liquidation. This is deliberately conservative; a production delisting/market-halt model requires better data. Benchmarks SPY and buy-and-hold of **every same-universe stock** use exactly matched dates, costs and adjusted-open execution. Dividend adjustments are included once; no dividend cash is credited separately.

Yahoo adjusted history is retroactively revised. Consistently adjusted OHLC prevents splitting/dividend accounting distortions, but these are **not actual historical executable quotes** and not a point-in-time data guarantee. Explicit data hashes stop stale cache changes from silently altering a frozen experiment. Corporate-action point-in-time and survivorship issues remain limitations, not claimed solved.

## Configuration

config.json is a complete example. Fields: universe, benchmark, start/end (end exclusive), horizon, target_threshold, train_min_dates, validation_start, test_start, walk_forward_dates, training_window (0 expanding), rebalance_days, starting_capital, max_positions, max_weight, sizing (equal/inverse_volatility), commission_bps, slippage_bps, probability_threshold, return_threshold, strategy (ranking/probability), feature_set (all/technical/statistical/market), select_features, seed, models.

If XGBoost is unavailable, remove xgboost from models or install the optional CPU wheel. No GPU dependency is required. Additional provider/model/feature implementations can be registered in their separate modules.

## Reports and interpretation

results/research-v1/report.md includes selected model/hyperparameters, predictive metrics, signals, strategy and SPY comparisons, risk, yearly/regime slices, and fixed cost-sensitivity diagnostics. CSVs expose raw predictions, all orders and partial closed lots. research.png and calibration.png are generated headlessly.

Metrics: total return, CAGR using 252 sessions/year, sample annualized volatility, zero-risk-free Sharpe, downside-RMS Sortino, max drawdown including initial costs, Calmar, closed-lot win rate/profit factor/average return, order count, exposure and gross one-way turnover. Profit factor is null when there are no losses (undefined), never an invented finite number. Partial sale lot accounting uses weighted-average cost incl entry fees; these are not independent completed-position trades. Time-series target overlap invalidates naive iid significance claims. Regression MAE/RMSE/R²/sign accuracy/correlation and classification accuracy/precision/recall/F1/ROC-AUC/confusion/Brier/calibration are included. Above-50% accuracy alone does not establish economic value; compare majority baseline, after-cost performance and risk.

No bootstrap statistical significance or calibrated prediction interval is asserted. Native importance may be unavailable for histogram boosting; permutation importance is always produced. Correlated signals and temporal dependency limit interpretation. SHAP is not included (optional in the brief).

## Testing and reproducibility

    .venv/bin/python -m unittest discover -s tests -v
    .venv/bin/python -m compileall -q quant_ml *.py
    .venv/bin/python -m pip check

Synthetic deterministic fixtures are used **only in unit tests**, never substituted for real market data or report results. Tests cover feature causality/math, horizon shift, global-date purge/rolling windows, training-only transforms, cache roundtrip, validation, missing prices, next-open fill timing, no future-signal effects, self-financing ledger, costs both sides, drifted turnover, weights/cash and matching benchmarks.

SPEC.md is the exact supplied prompt. PROGRESS.md and results/foundation* preserve evidence that baseline/data/targets/walk-forward/backtester were run before advanced models/screening. requirements-lock.txt captures the tested environment. See TEST_REPORT.md for command outcomes.

## Limits and safe use

Educational/research software only. Not investment advice. No broker credentials, automatic orders, paid APIs or scheduled jobs. Data licenses/provider terms still apply. A small present-day universe has selection and survivorship bias; no delisted-stock or historical constituent database. Transaction impacts, time-varying spreads, taxes, full calendar/currency handling and production monitoring are absent. Default example US tickers are USD; other instruments/currencies need explicit validation. Data can be wrong or unavailable. Report negative performance honestly. A local JSON holdout audit is a research guardrail, not tamper-proof governance.

## Primary documentation consulted

- https://scikit-learn.org/stable/common_pitfalls.html (training-only transforms)
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html (chronological split principles; this project uses stricter panel date/label-end purging)
- https://ranaroussi.github.io/yfinance/reference/index.html (market data provider)
