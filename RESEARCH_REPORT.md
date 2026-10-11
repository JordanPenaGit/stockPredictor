# Research report — scientific development study

**No reliable predictive or investment edge established. This is exploratory development, not a new holdout.**

## Provenance and fixed design
Protocol committed before performance: c25ce836a94578329de33a8ac2e193a82650e201. Plan hash: f4da709fea83c3f5fe46a07b5f59353535e116c95c8303a02b03f8d25f4e0603. Input hash: 2697ed1baf493a5be31163bb548cc1c9e67546b2d08ae88f826b42b4dd129f9e.
Artifacts: results/scientific-dev-20261011. 53 features; 1000 common signal dates, eight correlated surviving stocks; five-session overlapping next-open targets. Train minimum 756 dates, expanding refit every 126 dates; all preprocessing/selection train-only, target_end strictly before cutoff.
Data fetched retrospectively, sliced to 2015–2023 before modeling. Adjusted OHLC is a total-return proxy, not historically executable or point-in-time information. The already-seen 2024–2026 period was not reassessed.
No outcome-dependent changes. Three fixed model families plus five predeclared local-neighborhood/window variants; no best-model selection or deployment. XGBoost omitted from this bounded design before results, not because of performance.
All allocation curves share first/last opens, initial capital and per-side commission+slippage. Cash earns zero. Top-three strategies use 34% cap; full SPY/equal-weight buyhold have different concentration/exposure, so excess return is not risk-adjusted alpha. Terminal open liquidates only.

## Investment evidence (10 bps per side)
| Strategy | Total | CAGR | Sharpe | Sortino | Max DD | Vol | Calmar | Turnover | Exposure | Costs $ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY_buyhold | 55.98% | 11.87% | 0.6362 | 0.8667 | -32.05% | 21.17% | 0.3703 | 1.9990 | 100.00% | 256.03 |
| equal_weight_buyhold | 81.62% | 16.24% | 0.7988 | 1.1072 | -26.20% | 21.85% | 0.6199 | 1.9990 | 100.00% | 281.70 |
| gradient_boosting | 36.99% | 8.26% | 0.4407 | 0.6341 | -33.50% | 25.26% | 0.2467 | 148.1919 | 82.06% | 15594.74 |
| historical_mean | 131.15% | 23.54% | 0.8457 | 1.2158 | -38.42% | 30.51% | 0.6126 | 5.1209 | 100.00% | 857.11 |
| linear | -11.50% | -3.03% | 0.0074 | 0.0101 | -48.05% | 25.51% | -0.0631 | 149.0670 | 92.60% | 15021.19 |
| linear_alpha20 | -14.22% | -3.79% | -0.0242 | -0.0331 | -47.13% | 25.44% | -0.0805 | 145.1483 | 91.94% | 14320.71 |
| linear_alpha5 | -12.33% | -3.26% | -0.0006 | -0.0008 | -44.94% | 25.62% | -0.0727 | 151.6884 | 92.93% | 14183.04 |
| momentum60 | 103.45% | 19.62% | 0.8898 | 1.2861 | -21.56% | 23.16% | 0.9101 | 71.8507 | 94.92% | 12240.65 |
| naive_zero_cash | 0.00% | 0.00% | n/a | n/a | 0.00% | 0.00% | n/a | 0.0000 | 0.00% | 0.00 |
| random_forest | 34.08% | 7.68% | 0.4131 | 0.5947 | -28.82% | 26.27% | 0.2665 | 121.6496 | 82.84% | 12151.82 |
| rf_leaf20 | 28.93% | 6.62% | 0.3767 | 0.5284 | -30.03% | 26.07% | 0.2205 | 123.0701 | 82.16% | 12783.63 |
| rf_leaf80 | 10.61% | 2.58% | 0.2280 | 0.3242 | -30.85% | 26.05% | 0.0835 | 122.3395 | 82.87% | 11183.75 |
| rf_window756 | -1.16% | -0.29% | 0.1212 | 0.1691 | -33.61% | 26.41% | -0.0088 | 144.0066 | 80.55% | 13477.41 |
| trend200 | 110.45% | 20.65% | 0.8984 | 1.2716 | -27.13% | 24.15% | 0.7611 | 43.9401 | 94.46% | 7329.37 |
Execution interval: 2020-01-03 to 2023-12-21; 999 elapsed open intervals. Sharpe/Sortino use zero risk-free return. Costs include both entry and liquidation.

## Prediction evidence
| Forecast | MAE | RMSE | R² | Sign accuracy | Correlation | Class accuracy | AUC | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gradient_boosting | 0.0322 | 0.0436 | -0.0918 | 50.76% | 0.0463 | 50.19% | 0.4955 | 0.2588 |
| historical_mean | 0.0305 | 0.0418 | -0.0033 | 53.55% | -0.0101 | 53.75% | 0.5092 | 0.2500 |
| linear | 0.0309 | 0.0423 | -0.0314 | 51.50% | 0.0054 | 52.54% | 0.4870 | 0.2556 |
| linear_alpha20 | 0.0309 | 0.0423 | -0.0283 | 51.49% | 0.0061 | 52.54% | 0.4870 | 0.2556 |
| linear_alpha5 | 0.0310 | 0.0424 | -0.0350 | 51.45% | 0.0045 | 52.54% | 0.4870 | 0.2556 |
| naive_zero | 0.0306 | 0.0419 | -0.0078 | 46.25% | n/a | 53.75% | 0.5000 | 0.2500 |
| random_forest | 0.0324 | 0.0439 | -0.1079 | 52.41% | 0.0446 | 50.91% | 0.4947 | 0.2609 |
| rf_leaf20 | 0.0326 | 0.0443 | -0.1306 | 52.36% | 0.0378 | 51.36% | 0.4932 | 0.2620 |
| rf_leaf80 | 0.0314 | 0.0425 | -0.0389 | 51.60% | 0.0485 | 50.64% | 0.4927 | 0.2572 |
| rf_window756 | 0.0329 | 0.0446 | -0.1444 | 51.15% | 0.0062 | 50.28% | 0.4861 | 0.2656 |
Zero-return/.5-probability is the no-skill forecast; its >=.5 classifier predicts up. Historical means/up-rates are ticker-specific and purged. Momentum/trend scores are ordinal, not calibrated return forecasts. Cash is the naive economic comparator. Full confusion matrices, precision/recall/F1 and calibration bins are in summary.json.
Probabilities estimate a gross label, NOT net-profit confidence. Accuracy above 50% alone is insufficient. Calibration is measured, not fitted on these evaluation labels.

## Calendar and regime stability
| Strategy | 2020 | 2021 | 2022 | 2023 |
| --- | --- | --- | --- | --- |
| SPY_buyhold | 17.81% | 29.64% | -18.67% | 25.58% |
| equal_weight_buyhold | 27.44% | 30.32% | -17.65% | 32.79% |
| linear | 8.43% | 20.93% | -43.31% | 19.06% |
| random_forest | -3.84% | 6.31% | -12.06% | 49.15% |
| gradient_boosting | -4.21% | 11.43% | -5.27% | 35.48% |
| Strategy | Regime (prior-close SPY SMA200) | Dates | Mean daily return | Ann vol |
| --- | --- | --- | --- | --- |
| linear | above_sma200 | 720 | 0.06% | 19.59% |
| linear | below_sma200 | 279 | -0.15% | 36.55% |
| random_forest | above_sma200 | 720 | 0.07% | 18.59% |
| random_forest | below_sma200 | 279 | -0.02% | 39.77% |
| gradient_boosting | above_sma200 | 720 | 0.06% | 17.77% |
| gradient_boosting | below_sma200 | 279 | -0.00% | 38.39% |
Regime means are disjoint conditional daily slices, not standalone tradable strategy CAGRs. Few regimes and four years cannot establish stationarity.

## Paired date-block uncertainty
Maintained arch CircularBlockBootstrap; 1,000 repetitions, seed 42, fixed 5/20/60-day blocks. Stocks averaged within dates for paired squared-error differences. Portfolio differences paired by date. Below: 20-day block 95% percentile intervals for ANNUALIZED ARITHMETIC net-return difference (not CAGR or factor alpha).
| Model | Control | Estimate | Low | High |
| --- | --- | --- | --- | --- |
| linear | SPY_buyhold | -13.28% | -26.64% | 0.96% |
| linear | equal_weight_buyhold | -17.27% | -31.16% | -4.24% |
| random_forest | SPY_buyhold | -2.62% | -15.72% | 11.62% |
| random_forest | equal_weight_buyhold | -6.60% | -19.68% | 6.44% |
| gradient_boosting | SPY_buyhold | -2.34% | -15.53% | 12.08% |
| gradient_boosting | equal_weight_buyhold | -6.32% | -19.61% | 6.96% |
All block sizes, lag-one correlations and paired date-level MSE intervals are preserved in summary.json. Serial correlation, overlapping labels and common stocks invalidate iid row-count inference. Approximate stationary-block assumptions can fail across regimes. Intervals are NOT multiplicity/selection-adjusted; no formal significance claim, including for intervals excluding zero. Earlier research also increases selection risk.

## Predeclared robustness (not new tuning)
| Model | 0 bps CAGR | 10 bps CAGR | 25 bps CAGR | 50 bps CAGR |
| --- | --- | --- | --- | --- |
| linear | 0.68% | -3.03% | -8.35% | -16.58% |
| random_forest | 11.04% | 7.68% | 2.83% | -4.76% |
| gradient_boosting | 12.39% | 8.26% | 2.36% | -6.78% |
| Model | Omit-one allocation CAGR range | Top-five positive lot PnL share |
| --- | --- | --- |
| linear | -5.61% to 8.62% | 16.28% |
| random_forest | 6.80% to 14.76% | 15.90% |
| gradient_boosting | 2.71% to 11.98% | 14.29% |
Omit-one uses unchanged fitted predictions: it diagnoses allocation concentration, NOT leave-one-stock-out retraining/generalization. Per-stock metrics and PnL retained in summary. Top lots use weighted-average partial closings, not independent completed trades. Removing winners is not a realizable counterfactual.
Fixed Ridge alpha 5/10/20, RF leaves 20/40/80, and RF 756-date window appear in the full investment/prediction tables. Neighborhoods are diagnostics only; none was promoted.

## Acceptance gates
| Model | all_block_lower_bounds_positive | all_pass | drawdown_not_worse_equal_weight | every_year_excess | net_excess_10_and_25bps | predictive |
| --- | --- | --- | --- | --- | --- | --- |
| linear | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| random_forest | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| gradient_boosting | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
Failed gates are negative findings, not bugs to tune away. Even a passing exploratory gate would require genuinely future, fixed-protocol confirmation. No trading recommendation or profitability guarantee.

## Legacy already-seen assessment (read-only description)
Original research-v1 saved RF test accuracy 0.570985 versus always-up 0.574453; AUC 0.496552; R² 0.0010505. Saved CAGR 32.94% versus SPY 21.08%, on 2024-01-03 through 2026-10-02. These are original saved figures, NOT rerun with repaired accounting. Its old independence=true flag is a historical record, not an independence claim now. The original raw snapshot is absent, so exact replay is unsupported. No old file was modified.
Classifier weakness, survivor universe, retrospective adjusted data and weak development evidence prevent interpreting that appealing portfolio return as validated predictive skill. See docs/model-review.md for fuller descriptive context.

## Unresolved limitations and next evidence
- No historical constituents, delisted-security returns, point-in-time revisions or true executable auction quotes. A complete research implementation does not solve missing evidence.
- Fixed proportional costs omit variable spreads/impact, tax, liquidity and cash yield. Adjusted units are not actual share counts; no separate dividend credit avoids double counting.
- Parameter selection, correlated features, reused development, few stocks/years and data revisions limit transportability. No calibrated interval on an individual stock forecast.
- Source/config/data/dependency hashes enable provenance, not adversarially tamper-proof governance. Ignored snapshots/results do not ship in a clone; exact replay requires explicit archived snapshot transfer. A clean clone can download real data and run the same workflow, with possible vendor revisions.
- No broker, order placement, paid source or remote push. See ROADMAP.md for hypotheses and future append-only evidence collection, not a promise of future performance.

Failed experiments recorded by completed run: []. Any interrupted/failed directories remain separate evidence. See TEST_REPORT.md for commands and actual execution gates.
