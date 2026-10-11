# Stock prediction research — correctness before returns

Local Python research: cached Yahoo data, causal features, purged walk-forward
models, costed portfolios, matched baselines and append-only shadow forecasts.
**No broker, trades, paid service, or reliable-profit claim.**

## Current finding
All three tested ML families failed the predeclared development gates. On
2020–2023, CAGR after assumed costs was **−3.03% linear, 7.68% random forest,
8.26% gradient boosting**, versus **11.87% SPY and 16.24% same-universe buyhold**.
The old 2024–2026 holdout has already been seen; renaming it does not restore
independence. Full evidence and uncertainty: [RESEARCH_REPORT](RESEARCH_REPORT.md).

## Clean-clone setup (Python 3.10+; verified on 3.10/Linux)

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements-lock.txt
    .venv/bin/python -m unittest discover -s tests -v
    .venv/bin/python -m pip check

Root scripts prefer this project's .venv. The lock includes optional CPU XGBoost,
arch bootstrap and the exchange calendar. Other platforms may require compatible
wheels; the tested environment is recorded, not a cross-platform guarantee.
**Git contains source and compact evidence, NOT ignored data/models/results.**

## Reproduce the development workflow

    # Real Yahoo data, bounded linear/baseline workflow smoke:
    python3 research.py --smoke --output results/new-smoke
    # Full fixed development plan, all declared variants/diagnostics:
    python3 research.py --output results/new-development

Output directories must be new; failures are retained. Both commands download
only data before 2024. The frozen plan is [here](audit/development-plan.json).
An existing cache can be read without modifying it with --cache data/prices.sqlite.
Exact input replay needs an explicitly transferred TRUSTED development snapshot:
--snapshot /path/development_snapshot.joblib. Fresh downloads may reflect revisions.
Never load untrusted joblib/pickle files. No old local artifacts are implicitly needed.

## Original model workflow (separate from development-only research)

    python3 train.py --config config.json --run NEW-ID
    python3 backtest.py --run NEW-ID
    python3 analyze.py AAPL --run NEW-ID
    python3 screen.py --top 20 --run NEW-ID

Training freezes selection and snapshots; assessment reuses completed results.
These historical experiments are exploratory after holdout access, not new
independent tests. Existing runs/pointers/audits are preserved. Current refits
require matching frozen source/dependencies and append unique current records;
legacy source-incompatible models are not silently reused. --help lists flags.
The old research-v1 exists only in the original local checkout, not a fresh clone.

## Future evidence, not promised performance

    python3 forward.py init --root results/shadow-NEW-ID
    python3 forward.py capture --root results/shadow-NEW-ID
    python3 forward.py evaluate --root results/shadow-NEW-ID --signal-date YYYY-MM-DD

Capture must occur after close+30min and finish before next session open, on a
signal day AFTER initialization. Evaluation refuses immature labels. No scheduler
or order placement is enabled. See [ROADMAP](ROADMAP.md) for the exact frozen
shadow protocol, hypotheses, acceptance/rejection criteria and limitations.

## Where to look
- [AUDIT](AUDIT.md): confirmed defects, suspected risks, unresolved data limitations.
- [TEST_REPORT](TEST_REPORT.md): actual gates, clean clones, commands and outcomes.
- [RESEARCH_REPORT](RESEARCH_REPORT.md): negative/positive baselines, uncertainty, robustness.
- [ROADMAP](ROADMAP.md): prioritized falsifiable hypotheses and future collection.
- [Methodology](docs/methodology.md) / [CLI reference](docs/reference.md).

Important limits: eight present-day survivors; no historical constituents/delistings;
retrospective adjusted prices are NOT point-in-time executable quotes. Constant
costs omit variable spreads/impact/taxes. Correlated stocks/overlapping labels and
reused development limit inference. Model probabilities are not profit confidence.
