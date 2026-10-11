import pandas as pd
import numpy as np
from .technical import technical
from .statistical import statistical
from .market import market

def make_features(df,benchmark,feature_set='all'):
    blocks=[technical(df)]
    if feature_set in ['all','statistical']:blocks.append(statistical(df))
    if feature_set in ['all','market']:blocks.append(market(df,benchmark))
    if feature_set not in ['all','technical','statistical','market']:raise ValueError('Unknown feature set')
    return pd.concat(blocks,axis=1).replace([np.inf,-np.inf],np.nan).iloc[200:]

def make_dataset(prices,config):
    from ..targets import make_targets
    rows=[];feature_names=None
    for symbol in config.universe:
        frame=prices[symbol].reindex(prices[config.benchmark].index)
        features=make_features(frame,prices[config.benchmark],config.feature_set)
        targets=make_targets(frame,config.horizon,config.target_threshold)
        joined=features.join(targets).join(frame[['close','raw_close','open']])
        joined=joined.loc[joined.close.notna()];joined['ticker']=symbol
        feature_names=list(features.columns);rows.append(joined)
    panel=pd.concat(rows).reset_index().rename(columns={'index':'date'}).sort_values(['date','ticker']).reset_index(drop=True)
    panel['target_rank']=panel.groupby('date')['target_return'].rank(pct=True)
    return panel,feature_names
