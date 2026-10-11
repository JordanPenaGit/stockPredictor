"""Declared simple comparators, not selected by observed performance."""
import pandas as pd
from ..models.validation import folds

COLUMNS=['date','ticker','target_return','target_up','target_end','volatility_20','market_trend']

def baseline_predictions(panel, config, start, end):
    """Constant zero/.5 and per-ticker past matured mean/up-rate forecasts.

    Momentum60 and SMA200 are ordinal allocation signals, NOT calibrated return
    forecasts; no prediction metric is assigned to those signal magnitudes.
    """
    pieces=[];audit=[]
    for tr,va,meta in folds(panel,start,end,config.walk_forward_dates,
                            config.train_min_dates,config.training_window):
        train=panel.loc[tr];out=panel.loc[va,COLUMNS].copy()
        means=train.groupby('ticker').target_return.mean()
        rates=train.groupby('ticker').target_up.mean()
        out['predicted_return']=out.ticker.map(means)
        out['probability']=out.ticker.map(rates)
        if out[['predicted_return','probability']].isna().any().any():
            raise ValueError('Baseline has a ticker without matured training history')
        pieces.append(out);audit.append(meta)
    if not pieces:raise ValueError('No baseline folds')
    mean=pd.concat(pieces).sort_values(['date','ticker']).reset_index(drop=True)
    naive=mean.copy();naive['predicted_return']=0.;naive['probability']=.5
    source=panel.set_index(['date','ticker'])
    out={'historical_mean':mean,'naive_zero':naive}
    for name,column in [('momentum60','return_60'),('trend200','sma_200')]:
        p=mean.copy()
        p['predicted_return']=source.loc[pd.MultiIndex.from_frame(p[['date','ticker']]),column].to_numpy()
        p['probability']=.5
        if p.predicted_return.isna().any():raise ValueError(f'Missing {name} signal')
        out[name]=p
    return out,audit
