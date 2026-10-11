"""Local prospective SHADOW evidence, never orders. Clock cannot be set by CLI."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from dataclasses import replace
from importlib.metadata import version
import joblib
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from ..config import Config,ROOT
from ..data.downloader import YahooProvider
from ..features import make_dataset
from ..models.registry import make_model
from ..models.train import fit_pair,predict_pair,evaluate
from ..backtest.engine import construct_weights
from ..utils import source_hash,data_hash,versions,save_json,fingerprint


def utcnow():return pd.Timestamp(datetime.now(timezone.utc))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def environment():return {**versions(),'calendar':version('pandas_market_calendars'),'lock_sha256':sha(ROOT/'requirements-lock.txt')}
def schedule(start,end):return mcal.get_calendar('NYSE').schedule(start_date=start,end_date=end)


def init_protocol(root):
    """Fixed regularized linear shadow protocol, not chosen from positive results."""
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    now=utcnow()
    c=replace(Config(),models=['linear'],end=str((now+pd.Timedelta(days=1)).date()))
    templates={task:make_model('linear',task,c,1000) for task in ['regression','classification']}
    joblib.dump(templates,root/'pipeline_template.joblib')
    protocol={'created_at':str(now),'source_hash':source_hash(),'environment':environment(),
       'config':c.dict(),'family':'linear','template_hash':sha(root/'pipeline_template.joblib'),
       'policy':'Daily after-close capture; daily matured-label refit, immutable protocol; no trades',
       'comparators':['historical_mean','naive_zero'],
       'not_before_signal_date':str((now.tz_convert('America/New_York').normalize()+pd.Timedelta(days=1)).date()),
       'evaluation':'Wait until target exit session close +30min; gross adjusted-open target, NOT net trading profit',
       'shadow_allocation':'Record target weights only. Daily captures are not a costed portfolio backtest.',
       'evaluation_gate':'First formal review after >=252 captured sessions and >=126 matured non-overlapping horizon blocks; preserve all failures/gaps; no interim selection.',
       'warning':'Local clock/hashes are not tamper-proof. Negative development; no demonstrated edge.'}
    save_json(root/'protocol.json',protocol)
    return protocol


def load_protocol(root):
    root=Path(root);p=json.loads((root/'protocol.json').read_text())
    if p['source_hash']!=source_hash() or p['environment']!=environment():
        raise ValueError('Code/dependencies differ from forward freeze; do not mutate the old protocol')
    if p['template_hash']!=sha(root/'pipeline_template.joblib'):raise ValueError('Template changed')
    return p


def capture_window(now, protocol):
    local=now.tz_convert('America/New_York')
    sch=schedule(str((local-pd.Timedelta(days=10)).date()),str((local+pd.Timedelta(days=60)).date()))
    complete=sch[sch.market_close+pd.Timedelta(minutes=30)<=now]
    if complete.empty:raise ValueError('No completed session available')
    signal=complete.index[-1];i=sch.index.get_loc(signal)
    h=protocol['config']['horizon']
    if i+h+1>=len(sch):raise ValueError('Insufficient future calendar')
    entry=sch.iloc[i+1].market_open;exit_open=sch.iloc[i+h+1].market_open
    if now>=entry:raise ValueError('Too late for next-open prospective capture; wait until after current session close')
    if str(signal.date())<protocol['not_before_signal_date'] or pd.Timestamp(protocol['created_at'])>=sch.loc[signal,'market_open']:
        raise ValueError('Signal precedes prospectively frozen protocol')
    return signal,entry,exit_open,sch.iloc[i+h+1].market_close+pd.Timedelta(minutes=30)


def download_universe(c,end):
    provider=YahooProvider()
    return {s:provider.download(s,c.start,end) for s in [*c.universe,c.benchmark]}


def capture(root):
    root=Path(root);protocol=load_protocol(root);now=utcnow()
    signal,entry,exit_open,mature_after=capture_window(now,protocol)
    out=root/'captures'/str(signal.date());out.mkdir(parents=True,exist_ok=False)
    try:
        c=Config(**protocol['config']);end=str((signal+pd.Timedelta(days=1)).date())
        prices=download_universe(c,end)
        for symbol,frame in prices.items():
            if frame.index[-1]!=signal:raise ValueError(f'Stale or future bar for {symbol}')
        panel,features=make_dataset(prices,c)
        train=panel[(panel.target_end<signal)&panel.target_return.notna()]
        if c.training_window:train=train[train.date.isin(sorted(train.date.unique())[-c.training_window:])]
        if train.date.nunique()<c.train_min_dates:raise ValueError('Insufficient matured history')
        candidates=panel[panel.date==signal]
        if set(candidates.ticker)!=set(c.universe):raise ValueError('Incomplete current cross-section')
        pair=fit_pair(train,features,'linear',c,template=joblib.load(root/'pipeline_template.joblib'))
        pred=predict_pair(pair,candidates,features).drop(columns=['target_return','target_up','target_end'])
        if not np.isfinite(pred[['predicted_return','probability']].to_numpy()).all():raise ValueError('Invalid forecasts')
        joblib.dump(prices,out/'as_observed_prices.joblib',compress=3)
        joblib.dump(pair,out/'refit.joblib',compress=3)
        candidates[['date','ticker',*features]].to_csv(out/'features.csv',index=False)
        pred['historical_mean_return']=pred.ticker.map(train.groupby('ticker').target_return.mean())
        pred['historical_up_rate']=pred.ticker.map(train.groupby('ticker').target_up.mean())
        pred.to_csv(out/'forecasts.csv',index=False)
        completed=utcnow()
        if completed>=entry:raise ValueError('Computation completed after entry deadline; not prospective')
        record={'protocol_hash':fingerprint(protocol),'captured_at':str(completed),'signal_date':str(signal.date()),
          'entry_open':str(entry),'target_exit_open':str(exit_open),'evaluate_after':str(mature_after),
          'training_max_label_end':str(train.target_end.max()),'data_hash':data_hash(prices),
          'features':features,'target_weights':construct_weights(pred,c),'orders_placed':False,
          'files':{f.name:sha(f) for f in out.iterdir() if f.is_file()}}
        save_json(out/'record.json',record)
        return record
    except Exception as exc:
        save_json(out/'FAILED.json',{'error':str(exc),'at':str(utcnow()),'policy':'Preserve, do not overwrite or backdate'});raise


def evaluate_capture(root,signal_date):
    root=Path(root);protocol=load_protocol(root)
    # Validate date before using it as a local path component.
    if str(pd.Timestamp(signal_date).date())!=signal_date:raise ValueError('Require ISO signal date')
    folder=root/'captures'/signal_date;record=json.loads((folder/'record.json').read_text())
    if record['protocol_hash']!=fingerprint(protocol):raise ValueError('Protocol mismatch')
    for filename,digest in record['files'].items():
        if sha(folder/filename)!=digest:raise ValueError('Captured artifact changed')
    if utcnow()<pd.Timestamp(record['evaluate_after']):raise ValueError('Target not yet matured for evaluation')
    out=root/'evaluations'/signal_date;out.mkdir(parents=True,exist_ok=False)
    try:
        c=Config(**protocol['config']);entry=pd.Timestamp(record['entry_open']).tz_convert('America/New_York').tz_localize(None).normalize()
        exit_date=pd.Timestamp(record['target_exit_open']).tz_convert('America/New_York').tz_localize(None).normalize()
        pred=pd.read_csv(folder/'forecasts.csv',parse_dates=['date'])
        provider=YahooProvider();observed={};returns=[]
        for symbol in pred.ticker:
            f=provider.download(symbol,str(entry.date()),str((exit_date+pd.Timedelta(days=1)).date()))
            if entry not in f.index or exit_date not in f.index:raise ValueError(f'Missing target open for {symbol}')
            observed[symbol]=f;returns.append(float(f.at[exit_date,'open']/f.at[entry,'open']-1))
        pred['target_return']=returns;pred['target_up']=(pred.target_return>c.target_threshold).astype(float)
        pred['target_end']=exit_date
        pred.to_csv(out/'matured_predictions.csv',index=False)
        joblib.dump(observed,out/'as_observed_outcomes.joblib',compress=3)
        summary={'evaluated_at':str(utcnow()),'capture_record_hash':sha(folder/'record.json'),
                 'prediction_metrics':evaluate(pred),
                 'historical_mean_metrics':evaluate(pred.assign(predicted_return=pred.historical_mean_return,probability=pred.historical_up_rate)),
                 'naive_metrics':evaluate(pred.assign(predicted_return=0.,probability=.5)),
                 'dates':1,'rows':len(pred),
                 'warning':'One correlated cross-section, no significance or net investment-performance claim',
                 'files':{f.name:sha(f) for f in out.iterdir() if f.is_file()}}
        save_json(out/'evaluation.json',summary);return summary
    except Exception as exc:
        save_json(out/'FAILED.json',{'error':str(exc),'at':str(utcnow()),'policy':'Preserve failed evaluation'});raise
