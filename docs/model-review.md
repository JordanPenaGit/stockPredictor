# Model quality review — October 11, 2026

[README](../README.md) · [Methodology](methodology.md)

## Verdict

A useful research prototype, **not yet a proven strong predictor**. The saved
portfolio backtest is encouraging, but the directional classifier fails simple
baselines and the ranking evidence is unstable. No new training, holdout
evaluation, or artifact modification was performed for this review.

## Existing evidence

The selected family is random forest: separate return regression and direction
classification, with 70 trees, depth 5, and minimum leaf size 40. Ridge/logistic,
histogram gradient boosting, and XGBoost were also evaluated. The experiment
uses eight present-day stocks and a five-session next-open-to-open target.

Sources: saved [`assessment.json`](../results/research-v1/assessment.json),
[`validation.json`](../results/research-v1/validation.json), and prediction CSVs.
These describe the original historical run, not results regenerated with the
subsequently corrected code.

| Metric | Selected RF validation | Historical test |
| --- | ---: | ---: |
| Strategy period | 2020–2023 | Jan 2024–Oct 2, 2026 |
| Strategy CAGR after assumed costs | 7.67% | 32.94% |
| Sharpe (zero risk-free rate) | 0.413 | 1.508 |
| Maximum drawdown | −28.82% | −16.96% |
| Classification accuracy | 50.96% | 57.10% |
| Always-up accuracy | 53.70% | 57.45% |
| ROC AUC | 0.4948 | 0.4966 |
| Regression R² | −0.1079 | 0.00105 |

An AUC around 0.5 means no demonstrated directional discrimination. On the test
set the classifier predicts up about 98.85% of the time. Its Brier score
(0.24606) is slightly worse than using the validation up-rate as a constant
probability (0.24586). Probabilities are uncalibrated estimates of a gross-return
label, **not probabilities of net trading profit**.

The original strategy ranks predicted returns; its return does not validate the
combined return/probability rule behind a displayed BULLISH signal. Saved test
CAGR is 32.94% versus SPY 21.08%, but strategy volatility is higher (20.25% versus
15.81%). At 25 bps per side the saved strategy CAGR falls to 26.84%. Annual
strategy returns were +50.05%, +49.15%, and −2.57% through October 2, 2026.

Descriptive datewise Spearman rank correlation from saved predictions averages
0.0052 in validation and 0.0523 in test; test annual averages are 0.1051, 0.0691,
and −0.0429. This does not establish a stable stock-selection advantage.

## Why the backtest is not enough

- Eight surviving stocks do not represent a point-in-time investable universe.
- Thousands of panel rows are not independent trials: stocks correlate and
  five-session targets overlap. Test labels cover only 685 dates.
- The historical test uses scheduled causal retraining, not one fixed model.
  The protocol was frozen after the market interval existed; the local audit
  cannot establish that no researcher previously knew those outcomes.
- Yahoo adjustments are retrospective. The original run lacks its raw frozen
  snapshot, so exact replay is not guaranteed.
- There are no saved confidence intervals, factor-adjusted alpha, or fair
  risk comparisons against a same-universe equal-weight and momentum strategy.
- Only 54 of 690 test portfolio dates fall below SPY’s 200-day average; bearish
  regime coverage is limited.

## Best next experiments

1. Preserve this run and freeze source, data, dependencies, and experiment logs.
   Do not tune against 2024–2026 and call that interval unseen again.
2. Add causal naive prediction baselines and same-universe equal-weight and
   momentum portfolios with matching dates, costs, and risk comparisons.
3. On development data, align the objective with the ranking decision: test
   excess-return or cross-sectional rank targets, remove redundant features,
   and retain regularized linear/shallow-tree baselines.
4. Improve point-in-time universe coverage, delistings, and execution data before
   assuming a larger model will help.
5. If probabilities are used, calibrate only on earlier purged windows;
   calibration alone cannot repair an AUC near 0.5.
6. Pre-register net excess-return, risk, and stability gates; use date-block
   uncertainty estimates and selection-aware comparisons, then forward paper
   validation on genuinely new observations.

**Bottom line:** a better stock-ranking model is plausible, not guaranteed.
Current evidence supports further research, not a claim of reliable profits.
