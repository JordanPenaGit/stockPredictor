import numpy as np
import pandas as pd

def technical(df):
    c,h,l,v=df.close,df.high,df.low,df.volume
    f=pd.DataFrame(index=df.index)
    for n in [1,5,10,20,60]:f[f'return_{n}']=c.pct_change(n,fill_method=None)
    r=f.return_1; f['log_return']=np.log(c/c.shift())
    for n in [10,20,50,100,200]:f[f'sma_{n}']=c/c.rolling(n).mean()-1
    for n in [12,26,50]:f[f'ema_{n}']=c/c.ewm(span=n,adjust=False,min_periods=n).mean()-1
    f['sma_cross_10_50']=c.rolling(10).mean()/c.rolling(50).mean()-1
    f['sma_cross_50_200']=c.rolling(50).mean()/c.rolling(200).mean()-1
    diff=c.diff();gain=diff.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean();loss=(-diff.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    f['rsi']=100-100/(1+gain/loss.replace(0,np.nan));f.loc[(loss==0)&(gain>0),'rsi']=100;f.loc[(loss==0)&(gain==0),'rsi']=50
    macd=c.ewm(span=12,adjust=False,min_periods=26).mean()-c.ewm(span=26,adjust=False,min_periods=26).mean()
    f['macd']=macd/c;f['macd_signal']=macd.ewm(span=9,adjust=False,min_periods=9).mean()/c;f['macd_histogram']=f.macd-f.macd_signal
    f['roc']=f.return_20;f['momentum']=f.return_60
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    f['atr']=tr.rolling(14).mean()/c
    for n in [10,20,60]:f[f'volatility_{n}']=r.rolling(n).std()*np.sqrt(252)
    f['downside_volatility']=r.clip(upper=0).pow(2).rolling(20).mean().pow(.5)*np.sqrt(252)
    f['volume_change']=v.pct_change(fill_method=None);f['relative_volume']=v/v.rolling(20).mean()
    f['volume_mean_ratio']=v.rolling(5).mean()/v.rolling(60).mean();f['volume_price']=r*f.relative_volume
    f['volume_price_correlation']=r.rolling(20).corr(v.pct_change(fill_method=None))
    return f
