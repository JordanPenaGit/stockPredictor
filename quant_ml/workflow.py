"""Freeze validation-selected configuration before one-time final test assessment."""
from pathlib import Path
from dataclasses import replace
from datetime import datetime,timezone
import json, logging, os, re
import joblib
import pandas as pd
from .config import ROOT
from .data.downloader import MarketData
from .features import make_dataset
from .models.train import fit_pair,predict_pair,walk_forward,evaluate
from .models.interpret import importance
from .backtest.engine import simulate
from .backtest.metrics import performance
from .utils import save_json,fingerprint,data_hash,versions,source_hash
LOG=logging.getLogger(__name__)

def validate_run(run):
    if not isinstance(run,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}',run):
        raise ValueError('Run ID must be a simple name, not a path')
    return run

def validate_features(features,bundle):
    if list(features)!=bundle['features']:
        raise ValueError('Feature schema changed since freeze; refusing incompatible inference')

def load_bundle(run):
    run=validate_run(run);folder=ROOT/'models'/run;path=folder/'bundle.joblib'
    if not path.exists():raise FileNotFoundError(f'Train first: missing {path}')
    manifest=json.loads((folder/'manifest.json').read_text())
    bundle=joblib.load(path)
    config=bundle['config'];config.validate()
    if manifest.get('run')!=run or fingerprint(config.dict())!=manifest.get('config_hash') or config.dict()!=manifest.get('config'):
        raise ValueError('Frozen configuration metadata does not match bundle')
    if bundle['selected_model']!=manifest.get('selected_model'):
        raise ValueError('Frozen model metadata does not match bundle')
    validate_features(manifest.get('features',[]),bundle)
    for pair in bundle['frozen_pair'].values():
        if hasattr(pair,'feature_names_in_'):
            validate_features(pair.feature_names_in_,bundle)
    return bundle

def resolve_run(run=None):
    if run is not None:return validate_run(run)
    p=ROOT/'models'/'latest.json'
    if not p.exists():raise FileNotFoundError('No trained model. Run python3 train.py first.')
    return validate_run(json.loads(p.read_text())['run'])

def train_research(config,run,refresh=False):
    config.validate();run=validate_run(run)
    from .models.registry import available_models
    missing=set(config.models)-set(available_models())
    if missing:raise ValueError(f'Model dependencies unavailable: {sorted(missing)}')
    folder=ROOT/'models'/run;result=ROOT/'results'/run
    if folder.exists() or result.exists():raise FileExistsError('Run already frozen. Reuse it; never overwrite an evaluated artifact. Choose a genuinely new protocol/run if needed; holdout audit persists.')
    if (ROOT/'models'/'holdout_audit.json').exists():
        LOG.warning('Holdout has been viewed before. A new run is exploratory; it is NOT an independent untouched test.')
    data=MarketData().universe(config,refresh);panel,features=make_dataset(data,config)
    folder.mkdir(parents=True,exist_ok=False);result.mkdir(parents=True,exist_ok=False)
    # Immutable research snapshot survives later live cache refreshes.
    joblib.dump(data,folder/'market_snapshot.joblib',compress=3)
    rows=[];details={};out_predictions={};all_folds={}
    # No test rows/features/labels passed to fitting, prediction or validation scoring.
    development=panel[panel.date<pd.Timestamp(config.test_start)].copy()
    for name in config.models:
        LOG.info('Validation walk-forward: %s',name)
        pred,audit=walk_forward(development,features,name,config,config.validation_start,config.test_start)
        # Cut signal dates far enough before test boundary that all fills and labels remain development-only.
        pred=pred[pred.target_end<pd.Timestamp(config.test_start)]
        curve,trades,realized=simulate(data,pred,config)
        perf=performance(curve,trades,realized,config.starting_capital);metrics=evaluate(pred,config.test_start)
        rows.append({'model':name,'validation_sharpe':perf['sharpe'],'validation_total_return':perf['total_return'],'validation_rmse':metrics['rmse'],'validation_brier':metrics['brier_score']})
        details[name]={'performance':perf,'prediction_metrics':metrics};out_predictions[name]=pred;all_folds[name]=audit
        pred.to_csv(result/f'validation_{name}_predictions.csv',index=False)
    table=pd.DataFrame(rows).sort_values(['validation_sharpe','validation_rmse'],ascending=[False,True],na_position='last')
    selected=str(table.iloc[0].model)
    mature=(panel.date<pd.Timestamp(config.test_start))&(panel.target_end<pd.Timestamp(config.test_start))&panel.target_return.notna()
    if config.training_window:
        available=sorted(panel.loc[mature,'date'].unique())[-config.training_window:];mature &= panel.date.isin(available)
    frozen=fit_pair(panel[mature],features,selected,config)
    manifest={'run':run,'status':'selection_frozen_historical_exploratory','frozen_at':datetime.now(timezone.utc).isoformat(),'config':config.dict(),'selected_model':selected,'selection_rule':'highest costed validation Sharpe; RMSE tie-break','features':features,'data_hash':data_hash(data),'config_hash':fingerprint(config.dict()),'versions':versions(),'source_hash':source_hash(),'train_period':[str(panel.loc[mature,'date'].min().date()),str(panel.loc[mature,'date'].max().date())],'last_training_label_end':str(panel.loc[mature,'target_end'].max().date()),'validation_period':[config.validation_start,config.test_start],'test_period':[config.test_start,config.end],'prior_holdout_viewed':(ROOT/'models'/'holdout_audit.json').exists(),'hyperparameters':{task:frozen[task].named_steps['model'].get_params() for task in frozen},'validation':details,'folds':all_folds}
    # Freeze metadata before any holdout predictions. No feature importance on test used for selection.
    save_json(folder/'manifest.json',manifest);joblib.dump({'config':config,'features':features,'selected_model':selected,'frozen_pair':frozen},folder/'bundle.joblib')
    table.to_csv(result/'model_comparison.csv',index=False);save_json(result/'validation.json',details)
    save_json(ROOT/'models'/'latest.json',{'run':run});print(table.to_string(index=False));print(f'Frozen model: {selected}; no independence implied. Run python3 backtest.py --run {run}')
    return manifest

def assess(run):
    run=validate_run(run)
    folder=ROOT/'models'/run;result=ROOT/'results'/run
    manifest=json.loads((folder/'manifest.json').read_text());bundle=load_bundle(run);c=bundle['config']
    if (result/'assessment.json').exists():
        print('Using existing frozen assessment; no reselection/re-evaluation.');return json.loads((result/'assessment.json').read_text())
    if manifest.get('source_hash')!=source_hash():
        raise ValueError('Source changed or legacy source hash missing; refusing new assessment')
    if manifest.get('versions')!=versions():
        raise ValueError('Dependencies changed since freeze; refusing new assessment')
    snapshot=folder/'market_snapshot.joblib'
    data=joblib.load(snapshot) if snapshot.exists() else MarketData().universe(c)
    if data_hash(data)!=manifest['data_hash']:raise ValueError('Market data changed since freeze. Refusing incomparable final test.')
    panel,features=make_dataset(data,c)
    validate_features(features,bundle)
    auditfile=ROOT/'models'/'holdout_audit.json'
    history=json.loads(auditfile.read_text()) if auditfile.exists() else []
    if any(item['run']==run for item in history):raise RuntimeError('Assessment previously started but incomplete. Preserve partial outputs and use explicit recovery after investigating; do not silently peek again.')
    history.append({'run':run,'test_start':c.test_start,'end':c.end,'started_at':datetime.now(timezone.utc).isoformat(),'manifest_hash':fingerprint(manifest),'status':'started'})
    save_json(auditfile,history)
    # Frozen model family/hyperparameters; retraining uses only labels strictly matured at each fold.
    pred,fold_audit=walk_forward(panel,features,bundle['selected_model'],c,c.test_start,c.end,template=bundle['frozen_pair'])
    pred=pred[pred.date<data[c.benchmark].index[-1]]
    pred.to_csv(result/'test_predictions.csv',index=False);save_json(result/'test_folds.json',fold_audit)
    curve,trades,realized=simulate(data,pred,c);metrics=performance(curve,trades,realized,c.starting_capital)
    curve.to_csv(result/'equity.csv');trades.to_csv(result/'trades.csv',index=False);realized.to_csv(result/'closed_lots.csv',index=False)
    benchmarks={};benchmark_curves={}
    for s in [c.benchmark,*c.universe]:
        bc,bt,br=simulate(data,pred,c,benchmark_symbol=s);benchmarks[s]=performance(bc,bt,br,c.starting_capital);benchmark_curves[s]=bc
    benchmark_summary=pd.DataFrame(benchmarks).T;benchmark_summary.to_csv(result/'benchmarks.csv')
    # Fixed diagnostics, not selection. Label-end maturity cutoff excludes unresolved returns.
    model_metrics=evaluate(pred,c.end)
    test=panel[(panel.date>=pd.Timestamp(c.test_start))&(panel.target_end<pd.Timestamp(c.end))].dropna(subset=['target_return'])
    native,permutation=importance(bundle['frozen_pair'],test,features,c.seed)
    native.to_csv(result/'feature_importance.csv',index=False);permutation.to_csv(result/'permutation_importance.csv',index=False)
    # Fixed pre-specified annual and signal-time regime slices, descriptive not optimized.
    annual={str(year):{'return':float((1+g['return']).prod()-1),'volatility':float(g['return'].std()*252**.5),'days':len(g)} for year,g in curve.groupby(curve.index.year)}
    regimes={}
    market=data[c.benchmark].close;trend=(market/market.rolling(200).mean()-1).shift(1).reindex(curve.index)
    for name,mask in [('above_sma200',trend>=0),('below_sma200',trend<0)]:
        rr=curve.loc[mask,'return'];regimes[name]={'days':len(rr),'mean_daily_return':float(rr.mean()),'volatility':float(rr.std()*252**.5)}
    costs={}
    for bps in [0,10,25]:
        alt=replace(c,commission_bps=bps/2,slippage_bps=bps/2);cc,tt,rr=simulate(data,pred,alt);costs[str(bps)]=performance(cc,tt,rr,c.starting_capital)
    assessment={'run':run,'source_hash':source_hash(),'versions':versions(),'model':bundle['selected_model'],'test_performance':metrics,'prediction_metrics':model_metrics,'benchmarks':benchmarks,'annual':annual,'regimes':regimes,'cost_sensitivity_bps':costs,'holdout_independence':not manifest['prior_holdout_viewed'],'warning':'One selected, survivorship-biased universe; historical adjusted-price research, not a trading guarantee.'}
    save_json(result/'assessment.json',assessment)
    from .research.plots import research_plots
    from .research.reports import write_report
    research_plots(result,curve,benchmark_curves,pred,permutation)
    write_report(result,manifest,assessment,permutation)
    history[-1]['status']='complete';history[-1]['completed_at']=datetime.now(timezone.utc).isoformat();save_json(auditfile,history)
    print(json.dumps(metrics,indent=2));print(f'Report: {result / "report.md"}')
    return assessment
