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
