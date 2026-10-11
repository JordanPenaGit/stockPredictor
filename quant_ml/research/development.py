"""Predeclared development experiments. No holdout reads or model selection."""
import hashlib,json,subprocess,traceback
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from ..config import Config,ROOT
from ..features import make_dataset
from ..models.train import walk_forward,evaluate,fit_pair
from ..models.registry import make_model
from ..backtest.engine import simulate
from ..backtest.metrics import performance
from ..utils import save_json,source_hash,data_hash,fingerprint,versions,clean_json
from .data import development_prices,validate_prices
from .baselines import baseline_predictions
from .uncertainty import block_mean_interval,elapsed_returns


def config_from_plan(plan):
    values={k:v for k,v in plan.items() if k in Config.__dataclass_fields__}
    values.update(start=plan['data_start'],test_start=plan['data_end_exclusive'],
                  end=str((pd.Timestamp(plan['data_end_exclusive'])+pd.Timedelta(days=1)).date()))
    return Config(**values).validate()


def run_development(plan,output,cache=None,snapshot=None,smoke=False,prices_override=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    # Created before acquisition, prediction, or performance; partial runs preserved.
    frozen={'plan':plan,'plan_hash':fingerprint(plan),'source_hash':source_hash(),
            'versions':versions(),'created_at':datetime.now(timezone.utc).isoformat(),
            'mode':'bounded smoke, NOT the declared full experiment' if smoke else 'full development',
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'dependency_lock_sha256':hashlib.sha256((ROOT/'requirements-lock.txt').read_bytes()).hexdigest()}
    save_json(output/'frozen_protocol.json',frozen)
    try:
        return _run(plan,output,cache,snapshot,smoke,prices_override,frozen)
    except Exception as exc:
        save_json(output/'FAILED.json',{'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),
                                      'policy':'Preserve partial evidence. Investigate correctness; no outcome-driven retuning.'})
        raise


def _run(plan,output,cache,snapshot,smoke,prices_override,frozen):
    c=config_from_plan(plan)
    if prices_override is not None:
        prices=prices_override;provenance={'fixture':'DETERMINISTIC TEST ONLY, NEVER MARKET EVIDENCE'}
    elif snapshot:
        prices=joblib.load(snapshot);provenance={'snapshot_path':str(Path(snapshot).resolve()),
                                             'sha256':hashlib.sha256(Path(snapshot).read_bytes()).hexdigest()}
    else:prices,provenance=development_prices(plan,cache)
    validate_prices(prices,c.universe,c.benchmark,c.start,c.test_start)
    joblib.dump(prices,output/'development_snapshot.joblib',compress=3)
    save_json(output/'data_metadata.json',{'hash':data_hash(prices),'provenance':provenance,
            'rows':{s:len(f) for s,f in prices.items()},'first':str(prices[c.benchmark].index[0]),
            'last':str(prices[c.benchmark].index[-1]),'retrospective_not_point_in_time':True})
    panel,features=make_dataset(prices,c)
    end=(str((pd.Timestamp(c.validation_start)+pd.DateOffset(months=3)).date()) if smoke else c.test_start)
    # Retain only complete matured cross-sections; missing targets inside this interval fail.
    eligible=panel[(panel.date>=c.validation_start)&(panel.date<end)&(panel.target_end<pd.Timestamp(end))]
    if eligible.empty:raise ValueError('No matured development evaluation rows')
    counts=eligible.groupby('date').ticker.nunique()
    if (counts!=len(c.universe)).any() or eligible.target_return.isna().any():raise ValueError('Incomplete evaluation cross-section')
    dates=pd.DatetimeIndex(sorted(eligible.date.unique()))
    def trim(p):
        q=p[p.date.isin(dates)].reset_index(drop=True)
        if len(q)!=len(eligible):raise ValueError('Unequal prediction coverage')
        return q
    predictions,audits=baseline_predictions(panel,c,c.validation_start,end)
    predictions={k:trim(v) for k,v in predictions.items()}
    save_json(output/'baseline_folds.json',audits)
    template_metadata={}
    names=['linear'] if smoke else plan['models']+plan['variants']
    for name in names:
        print('Development fit:',name,flush=True)
        family='linear' if name.startswith('linear') else 'random_forest' if name.startswith('rf_') else name
        cc=replace(c,training_window=756) if name=='rf_window756' else c
        template={task:make_model(family,task,cc,len(features)) for task in ['regression','classification']}
        if name.startswith('linear_alpha'):template['regression'].set_params(model__alpha=int(name.removeprefix('linear_alpha')))
        if name.startswith('rf_leaf'):
            for model in template.values():model.set_params(model__min_samples_leaf=int(name.removeprefix('rf_leaf')))
        template_metadata[name]={task:model.get_params(deep=True) for task,model in template.items()}
        pred,audit=walk_forward(panel,features,family,cc,c.validation_start,end,template=template)
        predictions[name]=trim(pred)
        save_json(output/f'{name}_folds.json',audit)
    # Strings preserve every estimator/preprocessor parameter including nested objects.
    save_json(output/'model_protocol.json',{'features':features,'parameters':json.loads(json.dumps(template_metadata,default=str))})
    curves={};stats={};forecast={};costs={};orders={};lots={}
    specs={name:(p,None) for name,p in predictions.items() if name!='naive_zero'}
    ref=predictions['naive_zero']
    specs.update(SPY_buyhold=(ref,{c.benchmark:1.}),
                 equal_weight_buyhold=(ref,{s:1/len(c.universe) for s in c.universe}),
                 naive_zero_cash=(ref,{}))
    for name,(pred,hold) in specs.items():
        curve,trades,realized=simulate(prices,pred,c,buy_hold_weights=hold)
        curves[name]=curve;orders[name]=trades;lots[name]=realized
        stats[name]=performance(curve,trades,realized,c.starting_capital)
        curve.to_csv(output/f'{name}_equity.csv');trades.to_csv(output/f'{name}_trades.csv',index=False)
        realized.to_csv(output/f'{name}_closed_lots.csv',index=False)
        costs[name]={}
        for bps in plan['cost_bps_per_side']:
            alt=replace(c,commission_bps=bps/2,slippage_bps=bps/2)
            costs[name][str(bps)]=performance(*simulate(prices,pred,alt,buy_hold_weights=hold),c.starting_capital)
    for name,pred in predictions.items():
        pred.to_csv(output/f'{name}_predictions.csv',index=False)
        if name not in ['momentum60','trend200']:forecast[name]=evaluate(pred,end)
    base=curves['SPY_buyhold'].index
    if not all(curve.index.equals(base) for curve in curves.values()):raise AssertionError('Unequal benchmark periods')
    annual={};regimes={};single_stock={};omit={};trade_concentration={};uncertainty={};loss_uncertainty={}
    trend=(prices[c.benchmark].close/prices[c.benchmark].close.rolling(200).mean()-1).shift(1)
    for name,curve in curves.items():
        r=elapsed_returns(curve)
        annual[name]={str(y):{'return':float((1+g).prod()-1),'dates':len(g)} for y,g in r.groupby(r.index.year)}
        regimes[name]={}
        for regime,mask in [('above_sma200',trend.reindex(r.index)>=0),('below_sma200',trend.reindex(r.index)<0)]:
            g=r[mask];regimes[name][regime]={'dates':len(g),'mean_daily_return':float(g.mean()),'annualized_volatility':float(g.std()*np.sqrt(252))}
        stats[name]['excess_total_return_vs_SPY']=stats[name]['total_return']-stats['SPY_buyhold']['total_return']
        stats[name]['excess_total_return_vs_equal_weight']=stats[name]['total_return']-stats['equal_weight_buyhold']['total_return']
    for name in (['linear'] if smoke else plan['models']):
        p=predictions[name];realized=lots[name]
        single_stock[name]={s:{'prediction_metrics':evaluate(g,end),'realized_pnl':float(realized.loc[realized.ticker==s,'pnl'].sum()) if len(realized) else 0.} for s,g in p.groupby('ticker')}
        omit[name]={s:performance(*simulate(prices,p[p.ticker!=s],c),c.starting_capital) for s in c.universe if len(c.universe)>1}
        gains=realized.pnl.clip(lower=0).sum() if len(realized) else 0.
        trade_concentration[name]={'closed_partial_lots':len(realized),'positive_pnl':float(gains),
                'top5_positive_pnl_fraction':float(realized.pnl.nlargest(5).clip(lower=0).sum()/gains) if gains else None,
                'note':'Weighted-average partial lots, not independent trades. Not a counterfactual strategy.'}
        if not smoke:
            uncertainty[name]={};loss_uncertainty[name]={}
            for control in ['SPY_buyhold','equal_weight_buyhold']:
                diff=elapsed_returns(curves[name])-elapsed_returns(curves[control])
                uncertainty[name][control]={str(block):block_mean_interval(diff,block,plan['bootstrap']['replications'],c.seed,252) for block in plan['bootstrap']['blocks']}
            error=(p.predicted_return-p.target_return).pow(2)-(predictions['historical_mean'].predicted_return-p.target_return).pow(2)
            loss=error.groupby(p.date).mean()
            loss_uncertainty[name]={str(block):block_mean_interval(loss,block,plan['bootstrap']['replications'],c.seed) for block in plan['bootstrap']['blocks']}
    gates={}
    for name in (['linear'] if smoke else plan['models']):
        controls=['SPY_buyhold','equal_weight_buyhold']
        gates[name]={'predictive':bool(forecast[name]['rmse']<forecast['historical_mean']['rmse'] and forecast[name]['brier_score']<forecast['historical_mean']['brier_score'] and forecast[name]['roc_auc']>.5),
             'net_excess_10_and_25bps':all(costs[name][b]['total_return']>costs[ctrl][b]['total_return'] for b in ['10','25'] for ctrl in controls),
             'every_year_excess':all(annual[name][y]['return']>annual[ctrl][y]['return'] for y in annual[name] for ctrl in controls),
             'drawdown_not_worse_equal_weight':stats[name]['max_drawdown']>=stats['equal_weight_buyhold']['max_drawdown'],
             'all_block_lower_bounds_positive':False if smoke else all(v['low']>0 for ctrl in controls for v in uncertainty[name][ctrl].values())}
        gates[name]['all_pass']=all(gates[name].values())
    summary={'label':frozen['mode'],'independent_holdout':False,'config':c.dict(),'data_hash':data_hash(prices),
          'plan_hash':frozen['plan_hash'],'features':len(features),'prediction_dates':len(dates),
          'performance':stats,'prediction_metrics':forecast,'annual':annual,'regimes':regimes,
          'cost_sensitivity':costs,'single_stock':single_stock,'omit_stock_allocation_only':omit,
          'trade_concentration':trade_concentration,'paired_return_intervals':uncertainty,
          'paired_mse_intervals':loss_uncertainty,'gates':gates,'failed_experiments':[],
          'warning':'Exploratory, survivor-biased, retrospective adjusted-price research. No selection or positive-backtest optimization.'}
    save_json(output/'summary.json',summary)
    # Save explicit hashes without self-referential checksum.
    save_json(output/'artifact_hashes.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(output.iterdir()) if f.is_file()})
    print(json.dumps({'output':str(output),'gates':gates},indent=2),flush=True)
    return clean_json(summary)
