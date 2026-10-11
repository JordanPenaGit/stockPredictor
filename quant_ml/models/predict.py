from datetime import date
from dataclasses import replace
import numpy as np
import pandas as pd
from ..data.downloader import MarketData
from ..features import make_dataset
from ..models.train import fit_pair,predict_pair
from ..workflow import load_bundle,resolve_run,validate_features
from ..config import ROOT
from ..utils import save_json,source_hash,versions

def current_predictions(run=None,symbols=None,refresh=True):
    run=resolve_run(run);bundle=load_bundle(run);c=bundle['config']
    import re
    requested=list(dict.fromkeys(s.strip().upper() for s in (c.universe if symbols is None else symbols)))
    if not requested or any(not re.fullmatch(r'[A-Z0-9^][A-Z0-9.^=-]{0,19}',s) for s in requested):
        raise ValueError('Requested tickers must be nonempty valid symbols')
    # Latest inference is a separate refit, never overwrites the frozen evaluated artifact.
    universe=list(dict.fromkeys([*c.universe,*requested]));latest=replace(c,end=date.today().isoformat(),universe=universe)
    data=MarketData().universe(latest,refresh);panel,features=make_dataset(data,latest)
    validate_features(features,bundle)
    asof=data[c.benchmark].index[-1]
    training=panel[(panel.ticker.isin(c.universe))&(panel.target_end<asof)&panel.target_return.notna()]
    if c.training_window:training=training[training.date.isin(sorted(training.date.unique())[-c.training_window:])]
    if training.date.nunique()<c.train_min_dates:raise ValueError('Insufficient matured training history for current refit')
    candidates=panel[(panel.date==asof)&panel.ticker.isin(requested)]
    missing=set(requested)-set(candidates.ticker)
    if missing:raise ValueError(f'No fresh features for: {sorted(missing)}; refusing stale mixed-date screen')
    pair=fit_pair(training,features,bundle['selected_model'],c,template=bundle['frozen_pair'])
    result=predict_pair(pair,candidates,features).drop(columns=['target_return','target_up','target_end'])
    result['price']=candidates.raw_close.to_numpy();result['rsi']=candidates.rsi.to_numpy();result['momentum_20']=candidates.return_20.to_numpy();result['distance_sma50']=candidates.sma_50.to_numpy();result['relative_strength']=candidates.relative_strength.to_numpy() if 'relative_strength' in candidates else np.nan
    result['score']=result.predicted_return
    result['signal']=np.select([(result.predicted_return>c.return_threshold)&(result.probability>=c.probability_threshold),(result.predicted_return<0)&(result.probability<.5)],['BULLISH','BEARISH'],default='NEUTRAL')
    # Do not invent a confidence percentage from an uncalibrated model.
    result['confidence_note']='Uncertainty not quantified; probability is an uncalibrated model estimate. See OOS calibration report.'
    result['signal_note']='BULLISH/BEARISH is a descriptive screen label, not the backtested portfolio selection rule.'
    result=result.sort_values(['score','ticker'],ascending=[False,True]).reset_index(drop=True);result.index+=1
    root=ROOT/'results'/run;result.to_csv(root/'current_predictions.csv',index_label='rank')
    save_json(root/'current_prediction_metadata.json',{'asof':str(asof.date()),'source_hash':source_hash(),'versions':versions(),'horizon':c.horizon,'label':'next open through open h sessions later','training_max_label_end':str(training.target_end.max()),'trained_through':str(training.date.max()),'refit_separate_from_frozen_test':True,'refit_uses_frozen_pipeline_parameters':True,'screen_signal_rule':'BULLISH: predicted return > return_threshold AND probability >= probability_threshold; BEARISH: predicted return < 0 AND probability < 0.5','backtested_strategy':c.strategy,'screen_signal_is_backtested_strategy':False,'universe':requested})
    if (root/'assessment.json').exists():
        import json
        from ..research.reports import write_report
        manifest=json.loads((ROOT/'models'/run/'manifest.json').read_text())
        assessment=json.loads((root/'assessment.json').read_text())
        write_report(root,manifest,assessment,pd.read_csv(root/'permutation_importance.csv'))
    return result,c

def print_analysis(table,config):
    for _,r in table.iterrows():
        print(f'Ticker: {r.ticker} | as of {r.date.date()} close\nLatest unadjusted close: USD {r.price:,.2f}\n{config.horizon}-session expected next-open return: {r.predicted_return:+.2%}\nProbability return exceeds {config.target_threshold:+.2%}: {r.probability:.1%}\nModel signal: {r.signal}\nModel confidence: NOT ESTIMATED (not a probability of profit)\nRSI: {r.rsi:.1f} | 20-day momentum: {r.momentum_20:+.2%} | volatility: {r.volatility_20:.1%}\nDistance SMA50: {r.distance_sma50:+.2%} | relative strength vs benchmark: {r.relative_strength:+.2%}\n{r.confidence_note}\n{r.signal_note}\n')
