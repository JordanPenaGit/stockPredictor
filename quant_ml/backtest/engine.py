import pandas as pd
import numpy as np
from .portfolio import Portfolio

def construct_weights(predictions,config):
    p=predictions.copy()
    if p.ticker.duplicated().any():raise ValueError('Duplicate ticker in signal cross-section')
    if not np.isfinite(p[['predicted_return','probability']].to_numpy()).all():raise ValueError('Predictions must be finite')
    if not p.probability.between(0,1).all():raise ValueError('Probabilities must be in [0, 1]')
    if config.sizing=='inverse_volatility' and not (np.isfinite(p.volatility_20)&(p.volatility_20>=0)).all():
        raise ValueError('Inverse-volatility sizing requires finite nonnegative volatility')
    if config.strategy=='probability':p=p[p.probability>=config.probability_threshold]
    else:p=p[p.predicted_return>=config.return_threshold]
    p=p.sort_values(['predicted_return','ticker'],ascending=[False,True]).head(config.max_positions)
    if p.empty:return {}
    if config.sizing=='inverse_volatility':
        raw=1/p.volatility_20.clip(lower=.05);raw=raw/raw.sum()
    else:raw=pd.Series(1/len(p),index=p.index)
    # Cap; remainder stays cash (no hidden redistribution/leverage).
    return dict(zip(p.ticker,np.minimum(raw,config.max_weight)))

def simulate(prices,predictions,config,benchmark_symbol=None):
    calendar=prices[config.benchmark].index
    if not isinstance(calendar,pd.DatetimeIndex) or not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError('Benchmark calendar must be unique and increasing')
    if predictions.date.isna().any() or predictions.duplicated(['date','ticker']).any():
        raise ValueError('Prediction dates must be present and date/ticker pairs unique')
    dates=pd.DatetimeIndex(sorted(predictions.date.unique()))
    if dates.empty:raise ValueError('No predictions')
    positions=calendar.get_indexer(dates)
    if (positions<0).any():raise ValueError('Prediction dates must be benchmark sessions')
    if positions[-1]+1>=len(calendar):raise ValueError('Last prediction has no next execution open')
    if benchmark_symbol is not None and benchmark_symbol not in prices:raise ValueError('Missing benchmark prices')
    # Signal at close, next benchmark session open. Last open is liquidation only.
    first=positions[0]+1;last=positions[-1]+1
    sessions=calendar[first:last+1]
    if len(sessions)<2:raise ValueError('Insufficient execution sessions')
    portfolio=Portfolio(config.starting_capital);records=[];cost_rate=(config.commission_bps+config.slippage_bps)/10000
    grouped={d:g for d,g in predictions.groupby('date')}
    # Fixed benchmark-session cadence, anchored at the first executable signal.
    # Off-cadence forecasts are informational; missing due signals are errors.
    due=calendar[first-1:last-1:config.rebalance_days]
    if benchmark_symbol is None:
        missing=due.difference(dates)
        if len(missing):raise ValueError(f'Missing scheduled signal dates: {list(missing)}')
    for i,d in enumerate(sessions):
        prev=calendar[calendar.get_loc(d)-1]
        opening={s:float(f.at[d,'open']) for s,f in prices.items() if d in f.index}
        pre=portfolio.equity(opening);turnover=cost=0.0
        if benchmark_symbol:
            if i==0:turnover,cost=portfolio.rebalance({benchmark_symbol:1.0},opening,cost_rate,d)
        elif i<len(sessions)-1 and i%config.rebalance_days==0 and prev in grouped:
            w=construct_weights(grouped[prev],config)
            turnover,cost=portfolio.rebalance(w,opening,cost_rate,d)
        if i==len(sessions)-1:
            t,c=portfolio.rebalance({},opening,cost_rate,d);turnover+=t;cost+=c
        equity=portfolio.equity(opening)
        exposure=sum(q*opening[s] for s,q in portfolio.quantities.items() if q)/equity
        records.append({'date':d,'equity':equity,'cash':portfolio.cash,'exposure':exposure,'turnover':turnover,'cost':cost,'pretrade_equity':pre,'signal_date':prev,'decision_due':bool(i<len(sessions)-1 and i%config.rebalance_days==0)})
    curve=pd.DataFrame(records).set_index('date');curve['return']=curve.equity.pct_change(fill_method=None);curve.iloc[0,curve.columns.get_loc('return')]=curve.equity.iloc[0]/config.starting_capital-1
    curve['drawdown']=curve.equity/curve.equity.cummax().clip(lower=config.starting_capital)-1
    return curve,pd.DataFrame(portfolio.trades),pd.DataFrame(portfolio.realized)
