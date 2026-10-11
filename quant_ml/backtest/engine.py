import pandas as pd
import numpy as np
from .portfolio import Portfolio

def construct_weights(predictions,config):
    p=predictions.copy()
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
    dates=pd.DatetimeIndex(sorted(predictions.date.unique()))
    if dates.empty:raise ValueError('No predictions')
    # Signal at close, next benchmark session open. Extend through one last next open.
    first=calendar.get_indexer([dates[0]])[0]+1;last=calendar.get_indexer([dates[-1]])[0]+1
    sessions=calendar[first:min(last+1,len(calendar))]
    if len(sessions)<2:raise ValueError('Insufficient execution sessions')
    portfolio=Portfolio(config.starting_capital);records=[];cost_rate=(config.commission_bps+config.slippage_bps)/10000
    grouped={d:g for d,g in predictions.groupby('date')}
    for i,d in enumerate(sessions):
        prev=calendar[calendar.get_loc(d)-1]
        opening={s:float(f.at[d,'open']) for s,f in prices.items() if d in f.index}
        pre=portfolio.equity(opening);turnover=cost=0.0
        if benchmark_symbol:
            if i==0:turnover,cost=portfolio.rebalance({benchmark_symbol:1.0},opening,cost_rate,d)
        elif i%config.rebalance_days==0 and prev in grouped:
            w=construct_weights(grouped[prev],config)
            turnover,cost=portfolio.rebalance(w,opening,cost_rate,d)
        if i==len(sessions)-1:
            t,c=portfolio.rebalance({},opening,cost_rate,d);turnover+=t;cost+=c
        equity=portfolio.equity(opening)
        exposure=sum(q*opening[s] for s,q in portfolio.quantities.items() if q)/equity
        records.append({'date':d,'equity':equity,'cash':portfolio.cash,'exposure':exposure,'turnover':turnover,'cost':cost,'pretrade_equity':pre})
    curve=pd.DataFrame(records).set_index('date');curve['return']=curve.equity.pct_change(fill_method=None);curve.iloc[0,curve.columns.get_loc('return')]=curve.equity.iloc[0]/config.starting_capital-1
    curve['drawdown']=curve.equity/curve.equity.cummax().clip(lower=config.starting_capital)-1
    return curve,pd.DataFrame(portfolio.trades),pd.DataFrame(portfolio.realized)
