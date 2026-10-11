"""Bounded development inputs, separate from immutable legacy cache."""
import hashlib,json,sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from ..data.downloader import YahooProvider

def validate_prices(prices, universe, benchmark, start, end):
    expected=set(universe)|{benchmark}
    if set(prices)!=expected:raise ValueError('Price universe mismatch')
    calendar=prices[benchmark].index
    for symbol,f in prices.items():
        if f.empty or not isinstance(f.index,pd.DatetimeIndex) or not f.index.is_unique or not f.index.is_monotonic_increasing:
            raise ValueError(f'{symbol}: invalid calendar')
        if f.index.min()<pd.Timestamp(start) or f.index.max()>=pd.Timestamp(end):
            raise ValueError('Data outside declared development boundary')
        if not f.index.equals(calendar):raise ValueError(f'{symbol}: missing benchmark sessions; investigate rather than select survivors')
        cols=['open','high','low','close','raw_close','volume']
        if not np.isfinite(f[cols].to_numpy()).all() or (f[cols[:-1]]<=0).any().any() or (f.volume<0).any():
            raise ValueError(f'{symbol}: invalid prices')
        if (f.high<f.low).any() or (f.high+1e-8<f[['open','close']].max(axis=1)).any() or (f.low-1e-8>f[['open','close']].min(axis=1)).any():
            raise ValueError(f'{symbol}: inconsistent OHLC')

def development_prices(plan, cache=None):
    prices={};provenance={}
    for symbol in [*plan['universe'],plan['benchmark']]:
        if cache:
            path=Path(cache).resolve()
            key=json.dumps(['yahoo-adjusted-v1',symbol,plan['data_start'],plan['source_cache_request_end']])
            with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as conn:
                row=conn.execute('SELECT fetched,payload FROM requests WHERE key=?',(key,)).fetchone()
            if row is None:raise ValueError(f'Missing exact cache request for {symbol}; no silent fallback')
            frame=pd.DataFrame(json.loads(row[1]));frame['date']=pd.to_datetime(frame.date)
            frame=frame.set_index('date')
            provenance[symbol]={'request':key,'fetched_at':row[0],'source_payload_sha256':hashlib.sha256(row[1].encode()).hexdigest()}
        else:
            frame=YahooProvider().download(symbol,plan['data_start'],plan['data_end_exclusive'])
            provenance[symbol]={'source':'Yahoo fresh development-only download; may differ from frozen snapshot'}
        prices[symbol]=frame.loc[(frame.index>=plan['data_start'])&(frame.index<plan['data_end_exclusive'])].copy()
    validate_prices(prices,plan['universe'],plan['benchmark'],plan['data_start'],plan['data_end_exclusive'])
    return prices,provenance
