"""Paired date-block descriptive intervals, not selection-adjusted significance."""
import numpy as np
from arch.bootstrap import CircularBlockBootstrap

def block_mean_interval(values, block=20, reps=1000, seed=42, scale=1.):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or len(values)<2*block or not np.isfinite(values).all():
        raise ValueError('Need finite one-dimensional observations and at least two blocks')
    if type(block) is not int or block<1 or type(reps) is not int or reps<10:
        raise ValueError('Invalid bootstrap settings')
    bs=CircularBlockBootstrap(block,values,seed=seed)
    ci=bs.conf_int(np.mean,reps=reps,method='percentile',size=.95)
    return {'estimate':float(values.mean()*scale),'low':float(ci[0,0]*scale),
            'high':float(ci[1,0]*scale),'dates':len(values),'block':block,'reps':reps,
            'lag1_autocorrelation':float(np.corrcoef(values[:-1],values[1:])[0,1]) if values.std()>0 else None,
            'interpretation':'Approximate stationary circular-block percentile interval; not multiplicity-adjusted, not proof of alpha'}

def elapsed_returns(curve):
    r=curve['return'].iloc[1:].copy()
    r.iloc[0]=(1+curve['return'].iloc[0])*(1+r.iloc[0])-1
    return r
