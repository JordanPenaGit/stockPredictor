# Method choice, before performance inspection
Use maintained arch 7.2.0 CircularBlockBootstrap, not a new bootstrap engine.
Official documentation checked 2026-10-11:
https://arch.readthedocs.io/en/latest/bootstrap/timeseries-bootstraps.html
Circular fixed blocks include end-to-start wrapping; ordinary iid bootstrap
would discard temporal dependence. Fixed 5/20/60-session blocks, 1000 replications,
seed 42, percentile 95% intervals are sensitivity diagnostics, NOT optimized block
selection, selection-adjusted significance, or a guarantee of stationarity.
For forecast loss, average paired squared-error differences across stocks within
each date first; resample dates in blocks. For portfolio comparisons resample
paired daily net-return differences, preserving paired benchmark alignment.
Interpret annualized arithmetic difference, not CAGR alpha. Cross-sectional,
serial, overlapping-target, regime-change and repeated-research dependence remain.
