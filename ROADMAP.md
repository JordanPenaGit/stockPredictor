# Research roadmap — evidence before complexity

## Decision now
Do not promote any tested ML model. All three base models failed all predeclared
development gates. Better-looking simple baselines are still exploratory results
from a small retrospective survivor universe. Do not optimize against 2024–2026,
change seeds to claim independence, or start a larger model because results are poor.

## Prioritized hypotheses (value / effort)
Each item requires a NEW versioned plan before outcomes, the same purged date
splits, matched costs, negative-result retention and genuinely future confirmation.
No paid source/service is authorized; no trade or broker connection is proposed.

1. **Data adequacy and prospective provenance — highest value / medium effort.**
   Hypothesis: much apparent edge is universe/revision/execution bias. Rationale:
   survivor-only selection can dominate modest model improvements. Data: archive
   free exchange/issuer records, contemporaneous daily snapshots and corporate
   actions; historical constituents and complete delisting returns are currently
   unavailable, not inferred from surviving Yahoo tickers. Method: reconcile
   known splits/dividends/halts against raw and adjusted series; test an explicitly
   dated investability universe. Compute: minutes of validation, days/weeks of
   data curation, no model search. Accept data only with documented inclusion and
   delisting coverage and reconciliation; otherwise reject representative-universe
   performance claims. A license/procurement decision needs separate approval.

2. **Fixed shadow collection — high value / low compute, long elapsed time.**
   Hypothesis: frozen regularized linear forecasts beat purged historical means
   on genuinely new data. Rationale: simple baseline makes a strong falsifiable
   control, despite negative historical evidence. Data: timestamped after-close
   snapshots and first matured outcomes; keep all missing/failed days. Method and
   acceptance: see protocol below; compare paired loss/calibration by date and
   retain controls. CPU: one Ridge/logistic refit/day, no GPU. Reject if lower loss
   and stable calibration do not persist; never replace the model mid-cohort.

3. **Feature reduction / stronger regularization — high value / low effort.**
   Hypothesis: correlated technical features amplify estimation noise. Rationale:
   all tested regressors lost to historical-mean RMSE; increased capacity is not
   justified. Data: same admissible universe plus stronger provenance if available.
   Method: predeclare 6–10 nonredundant return/trend/volatility/market features;
   train-only selection, nested purged development comparison against fixed Ridge.
   Compute: <30 CPU minutes for a small fixed grid. Accept only lower date-block
   paired loss, acceptable calibration and net excess versus BOTH SPY/equal-weight
   across years/costs; reject if gains depend on one period or parameter. Do not
   revise current experiment to implement this hypothesis retroactively.

4. **Cross-sectional excess-return / rank objective — medium value / medium effort.**
   Hypothesis: relative-return prediction fits top-N allocation better than raw
   return prediction. Rationale: remove broad drift but avoid confusing exposure
   with stock-selection skill. Data: larger historical/PTI universe is a prerequisite;
   eight names are weak ranking evidence. Method: target benchmark-relative returns
   or cross-sectional rank, shared-date purges, datewise rank IC and matched
   allocations, predeclare normalization. Compute: linear/shallow-tree fits only,
   <1 CPU hour. Accept consistent out-of-window rank IC and costed risk-aware excess;
   reject unstable sign, one-stock dependence, or no advantage over momentum.

5. **Volatility-normalized targets and sizing — medium value / low effort.**
   Hypothesis: scaling by trailing volatility reduces heteroskedastic loss without
   manufacturing risk differences. Data: past-only rolling volatility and valid
   prices. Method: train-only target normalization using information at the signal
   date; unscale forecasts causally; compare equal/inverse-vol risk with identical
   caps and realized exposures. Compute: small linear/tree experiment (<30 min).
   Accept lower loss AND comparable-risk net performance; reject a result explained
   only by leverage, cash exposure or a different universe.

6. **Probability calibration — conditional value / low effort.**
   Hypothesis: train-on-past calibration improves probability reliability, not AUC.
   Data: disjoint earlier purged calibration window with adequate labels. Method:
   sigmoid/isotonic calibration on a strictly prior slice; outer predictions remain
   untouched; compare Brier/reliability against constant past up-rate. Compute:
   minutes. Accept stable lower Brier across years and enough observations per bin;
   reject overfit bins or no discrimination. Calibration cannot repair AUC near .5.

7. **Shallow boosting / regime interaction — lower value / medium effort.**
   Hypothesis: a small set of predeclared nonlinear interactions survives strict
   controls. Data: sufficiently many bear/bull regimes and PIT universe first.
   Method: bounded shallow model vs reduced linear/momentum, regime features known
   at close, no separate hand-tuned bull/bear systems. Compute: <1 CPU hour, no
   deep learning. Accept only incremental stable net evidence under all controls;
   reject if it needs extra complexity, selective regimes or hyperparameter hunting.

## Implemented append-only future path
No scheduler or unattended follow-up has been activated. These are explicit local
commands for an operator; no broker, orders or allocation execution occurs.

    python3 forward.py init --root results/shadow-NEW-ID
    python3 forward.py capture --root results/shadow-NEW-ID
    python3 forward.py evaluate --root results/shadow-NEW-ID --signal-date YYYY-MM-DD

- Init freezes code, actual dependency versions/lock, full pipeline template and
  configuration. A new ID cannot restore independence of old outcomes. It starts
  no earlier than the day AFTER initialization and before that signal day's open.
- Capture uses the real system clock (no CLI backdating). Maintained NYSE calendar
  handles weekends, holidays and early closes. Earliest capture is close(T)+30min,
  and it must finish BEFORE open(T+1). Every symbol's latest bar must be T.
- A fixed Ridge/logistic protocol refits DAILY on labels with target_end < T.
  This daily shadow refit is explicitly different from the historical 126-date
  cadence; it is not claimed to replicate the old strategy or chosen for better
  historical performance. Past observed outcomes may mature into future training;
  family/features/horizon/parameters cannot change within the cohort.
- Saves as-observed input snapshot, trained pair, feature matrix, forecast, past
  historical-mean/up-rate comparators, target weights and hashes. Weights are
  records, NOT trades; cash interest and investment PnL are not inferred.
- Evaluation waits through target exit session close+30min (conservative publication
  allowance) and saves outcome snapshot, model/historical-mean/zero-control metrics
  separately. It refuses immature or duplicate evaluation, changed inputs/templates
  or code/dependencies. Failures are retained; no retries that overwrite evidence.
- First formal review only after >=252 captured sessions AND >=126 matured
  non-overlapping five-session blocks (about 630 sessions, not 252 independent
  samples). Record gap/failure coverage, date-block paired loss and calibration.
  No stopping when performance looks good; no interim model selection. No alpha
  claim without a separately predeclared costed prospective portfolio assessment.
- Reject predictive hypothesis if lower paired loss against historical mean is
  not sustained with uncertainty and calibration checks. Even a positive forecast
  result does not establish net trading profitability or fix survivorship bias.
- Capture/evaluation dates follow the calendar as installed. Unscheduled exchange
  closures, vendor revisions, clock manipulation, data licenses and local file
  edits require investigation. Local hashes are NOT tamper-proof external attestation.

Tests exercise synthetic early-close/holiday timing, after-close/next-open bounds,
no pre-freeze capture, matured evaluation, duplicate refusal, metadata/input tamper
checks and end-to-end forecast/outcome storage. Synthetic tests are not future
performance. **No prospective performance is available as of October 11, 2026.**
