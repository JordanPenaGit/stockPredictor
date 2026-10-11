"""Render saved development evidence only; never retrain or overwrite artifacts."""
import argparse,json
from pathlib import Path

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
def pct(x):return 'n/a' if x is None else f'{x:.2%}'
def num(x):return 'n/a' if x is None else f'{x:.4f}'

def render(root):
    s=json.loads((root/'summary.json').read_text());f=json.loads((root/'frozen_protocol.json').read_text())
    models=['linear','random_forest','gradient_boosting'];models=[m for m in models if m in s['performance']]
    lines=['# Research report — scientific development study','',
      '**No reliable predictive or investment edge established. This is exploratory development, not a new holdout.**','',
      '## Provenance and fixed design',
      f"Protocol committed before performance: {f['git_commit']}. Plan hash: {s['plan_hash']}. Input hash: {s['data_hash']}.",
      f"Artifacts: {root}. {s['features']} features; {s['prediction_dates']} common signal dates, eight correlated surviving stocks; five-session overlapping next-open targets. Train minimum 756 dates, expanding refit every 126 dates; all preprocessing/selection train-only, target_end strictly before cutoff.",
      'Data fetched retrospectively, sliced to 2015–2023 before modeling. Adjusted OHLC is a total-return proxy, not historically executable or point-in-time information. The already-seen 2024–2026 period was not reassessed.',
      'No outcome-dependent changes. Three fixed model families plus five predeclared local-neighborhood/window variants; no best-model selection or deployment. XGBoost omitted from this bounded design before results, not because of performance.',
      'All allocation curves share first/last opens, initial capital and per-side commission+slippage. Cash earns zero. Top-three strategies use 34% cap; full SPY/equal-weight buyhold have different concentration/exposure, so excess return is not risk-adjusted alpha. Terminal open liquidates only.','',
      '## Investment evidence (10 bps per side)',
      table(['Strategy','Total','CAGR','Sharpe','Sortino','Max DD','Vol','Calmar','Turnover','Exposure','Costs $'],
       [[n,pct(p['total_return']),pct(p['cagr']),num(p['sharpe']),num(p['sortino']),pct(p['max_drawdown']),pct(p['volatility']),num(p['calmar']),num(p['turnover']),pct(p['exposure']),f"{p['costs']:.2f}"] for n,p in s['performance'].items()]),
      f"Execution interval: {s['performance']['SPY_buyhold']['start']} to {s['performance']['SPY_buyhold']['end']}; {s['performance']['SPY_buyhold']['periods']} elapsed open intervals. Sharpe/Sortino use zero risk-free return. Costs include both entry and liquidation.",'',
      '## Prediction evidence',
      table(['Forecast','MAE','RMSE','R²','Sign accuracy','Correlation','Class accuracy','AUC','Brier'],
       [[n,num(p['mae']),num(p['rmse']),num(p['r2']),pct(p['directional_accuracy']),num(p['correlation']),pct(p['accuracy']),num(p['roc_auc']),num(p['brier_score'])] for n,p in s['prediction_metrics'].items()]),
      'Zero-return/.5-probability is the no-skill forecast; its >=.5 classifier predicts up. Historical means/up-rates are ticker-specific and purged. Momentum/trend scores are ordinal, not calibrated return forecasts. Cash is the naive economic comparator. Full confusion matrices, precision/recall/F1 and calibration bins are in summary.json.',
      'Probabilities estimate a gross label, NOT net-profit confidence. Accuracy above 50% alone is insufficient. Calibration is measured, not fitted on these evaluation labels.','',
      '## Calendar and regime stability',
      table(['Strategy',*list(s['annual']['SPY_buyhold'])],[[n,*[pct(v['return']) for v in s['annual'][n].values()]] for n in ['SPY_buyhold','equal_weight_buyhold',*models]]),
      table(['Strategy','Regime (prior-close SPY SMA200)','Dates','Mean daily return','Ann vol'],[[n,reg,v['dates'],pct(v['mean_daily_return']),pct(v['annualized_volatility'])] for n in models for reg,v in s['regimes'][n].items()]),
      'Regime means are disjoint conditional daily slices, not standalone tradable strategy CAGRs. Few regimes and four years cannot establish stationarity.','',
      '## Paired date-block uncertainty',
      'Maintained arch CircularBlockBootstrap; 1,000 repetitions, seed 42, fixed 5/20/60-day blocks. Stocks averaged within dates for paired squared-error differences. Portfolio differences paired by date. Below: 20-day block 95% percentile intervals for ANNUALIZED ARITHMETIC net-return difference (not CAGR or factor alpha).',
      table(['Model','Control','Estimate','Low','High'],[[m,k,pct(v['20']['estimate']),pct(v['20']['low']),pct(v['20']['high'])] for m in models for k,v in s['paired_return_intervals'].get(m,{}).items()]),
      'All block sizes, lag-one correlations and paired date-level MSE intervals are preserved in summary.json. Serial correlation, overlapping labels and common stocks invalidate iid row-count inference. Approximate stationary-block assumptions can fail across regimes. Intervals are NOT multiplicity/selection-adjusted; no formal significance claim, including for intervals excluding zero. Earlier research also increases selection risk.','',
      '## Predeclared robustness (not new tuning)',
      table(['Model','0 bps CAGR','10 bps CAGR','25 bps CAGR','50 bps CAGR'],[[m,*[pct(s['cost_sensitivity'][m][b]['cagr']) for b in ['0','10','25','50']]] for m in models]),
      table(['Model','Omit-one allocation CAGR range','Top-five positive lot PnL share'],[[m,f"{pct(min(v['cagr'] for v in s['omit_stock_allocation_only'][m].values()))} to {pct(max(v['cagr'] for v in s['omit_stock_allocation_only'][m].values()))}" if s['omit_stock_allocation_only'][m] else 'n/a',pct(s['trade_concentration'][m]['top5_positive_pnl_fraction'])] for m in models]),
      'Omit-one uses unchanged fitted predictions: it diagnoses allocation concentration, NOT leave-one-stock-out retraining/generalization. Per-stock metrics and PnL retained in summary. Top lots use weighted-average partial closings, not independent completed trades. Removing winners is not a realizable counterfactual.',
      'Fixed Ridge alpha 5/10/20, RF leaves 20/40/80, and RF 756-date window appear in the full investment/prediction tables. Neighborhoods are diagnostics only; none was promoted.','',
      '## Acceptance gates',
      table(['Model',*list(s['gates'][models[0]])],[[m,*['PASS' if v else 'FAIL' for v in s['gates'][m].values()]] for m in models]),
      'Failed gates are negative findings, not bugs to tune away. Even a passing exploratory gate would require genuinely future, fixed-protocol confirmation. No trading recommendation or profitability guarantee.','',
      '## Legacy already-seen assessment (read-only description)',
      'Original research-v1 saved RF test accuracy 0.570985 versus always-up 0.574453; AUC 0.496552; R² 0.0010505. Saved CAGR 32.94% versus SPY 21.08%, on 2024-01-03 through 2026-10-02. These are original saved figures, NOT rerun with repaired accounting. Its old independence=true flag is a historical record, not an independence claim now. The original raw snapshot is absent, so exact replay is unsupported. No old file was modified.',
      'Classifier weakness, survivor universe, retrospective adjusted data and weak development evidence prevent interpreting that appealing portfolio return as validated predictive skill. See docs/model-review.md for fuller descriptive context.','',
      '## Unresolved limitations and next evidence',
      '- No historical constituents, delisted-security returns, point-in-time revisions or true executable auction quotes. A complete research implementation does not solve missing evidence.',
      '- Fixed proportional costs omit variable spreads/impact, tax, liquidity and cash yield. Adjusted units are not actual share counts; no separate dividend credit avoids double counting.',
      '- Parameter selection, correlated features, reused development, few stocks/years and data revisions limit transportability. No calibrated interval on an individual stock forecast.',
      '- Source/config/data/dependency hashes enable provenance, not adversarially tamper-proof governance. Ignored snapshots/results do not ship in a clone; exact replay requires explicit archived snapshot transfer. A clean clone can download real data and run the same workflow, with possible vendor revisions.',
      '- No broker, order placement, paid source or remote push. See ROADMAP.md for hypotheses and future append-only evidence collection, not a promise of future performance.',
      '',f"Failed experiments recorded by completed run: {s['failed_experiments']}. Any interrupted/failed directories remain separate evidence. See TEST_REPORT.md for commands and actual execution gates."]
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('output');a=p.parse_args()
    with Path(a.output).open('x') as f:f.write(render(Path(a.run)))
