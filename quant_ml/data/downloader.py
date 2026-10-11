"""Real Yahoo data, retained raw OHLC plus consistently adjusted OHLC and actions."""
from typing import Protocol
import json, logging, re
import numpy as np
import pandas as pd
import yfinance as yf
from .cache import PriceCache
from ..config import ROOT
LOG=logging.getLogger(__name__)
class DataError(RuntimeError): pass
class Provider(Protocol):
    def download(self,symbol,start,end): ...
class YahooProvider:
    cache_namespace='yahoo'
    def download(self,symbol,start,end):
        try:
            df=yf.Ticker(symbol).history(start=start,end=end,interval='1d',auto_adjust=False,actions=True,raise_errors=True,timeout=25)
        except Exception as e:raise DataError(f'{symbol}: Yahoo request failed: {e}') from e
        if df.empty:raise DataError(f'{symbol}: no historical data in interval')
        df.index=pd.DatetimeIndex(df.index).tz_localize(None).normalize(); df.index.name='date'
        df=df.rename(columns={'Open':'raw_open','High':'raw_high','Low':'raw_low','Close':'raw_close','Adj Close':'adj_close','Volume':'volume','Dividends':'dividends','Stock Splits':'splits'})
        if 'adj_close' not in df: raise DataError(f'{symbol}: adjusted close missing')
        ratio=df.adj_close/df.raw_close
        for col in ['open','high','low','close']: df[col]=df['raw_'+col]*ratio
        for col in ['dividends','splits']:
            if col not in df:df[col]=0.0
        cols=['open','high','low','close','volume','raw_open','raw_high','raw_low','raw_close','adj_close','dividends','splits']
        df=df[cols].replace([np.inf,-np.inf],np.nan)
        valid=(df[['open','high','low','close','volume']].notna().all(axis=1)&(df[['open','high','low','close']]>0).all(axis=1)&(df.volume>=0))
        if not valid.all() or not df.index.is_unique:
            raise DataError(f'{symbol}: invalid or duplicate OHLCV rows; refusing silent repair')
        return df.sort_index()
class MarketData:
    def __init__(self,provider=None,cache_path=None):
        self.provider=provider or YahooProvider();self.cache=PriceCache(cache_path or ROOT/'data'/'prices.sqlite')
    def get(self,symbol,start,end,refresh=False):
        symbol=symbol.strip().upper()
        if not re.fullmatch(r'[A-Z0-9^][A-Z0-9.^=-]{0,19}',symbol):raise DataError('Invalid ticker format')
        key=json.dumps([getattr(self.provider,'cache_namespace',type(self.provider).__name__)+'-adjusted-v1',symbol,start,end])
        cached=None if refresh else self.cache.get(key)
        if cached is not None:return cached
        LOG.info('Downloading %s %s through %s',symbol,start,end)
        data=self.provider.download(symbol,start,end)
        if data.empty:raise DataError(f'{symbol}: empty cleaned data')
        self.cache.put(key,data);return self.cache.get(key)
    def universe(self,config,refresh=False):
        # Fail loudly: silently dropping a failed ticker changes the historical universe.
        return {s:self.get(s,config.start,config.end,refresh) for s in [*config.universe,config.benchmark]}
