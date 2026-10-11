import json,hashlib,platform,subprocess
from pathlib import Path
import numpy as np
import pandas as pd

def clean_json(value):
    if isinstance(value,dict):return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean_json(v) for v in value]
    if isinstance(value,(np.floating,float)):return float(value) if np.isfinite(value) else None
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(pd.Timestamp,Path)):return str(value)
    return value

def save_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(clean_json(obj),indent=2,sort_keys=True,allow_nan=False))

def fingerprint(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()

def data_hash(prices):
    h=hashlib.sha256()
    for s in sorted(prices):h.update(s.encode());h.update(pd.util.hash_pandas_object(prices[s],index=True).values.tobytes())
    return h.hexdigest()

def versions():
    from importlib.metadata import version
    return {'python':platform.python_version(),**{k:version(k) for k in ['numpy','pandas','scikit-learn','scipy','yfinance','joblib','matplotlib']}}
