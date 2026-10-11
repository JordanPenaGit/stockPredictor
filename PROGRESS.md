# Verified build stages

1. Exact user specification saved in SPEC.md; isolated environment installed successfully.
2. Real yfinance daily raw/adjusted OHLCV and corporate actions downloaded for 8 stocks + SPY into SQLite.
3. Causal technical/statistical/market features and next-open horizon targets implemented.
4. Linear/logistic baseline, date-grouped purged expanding walk-forward, self-financing long-only portfolio implemented.
5. 9/9 foundation tests passed. Real 2020 baseline walk-forward and costed backtest executed; results/foundation/verification.json records actual metrics (negative strategy return, not hidden).
6. Foundation gate passed before implementing advanced models and screening. Working on advanced comparison, frozen final holdout, reports and CLIs.
