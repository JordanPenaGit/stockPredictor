import pandas as pd
import numpy as np
def market(df,benchmark):
    c=df.close;b=benchmark.close.reindex(df.index);r=c.pct_change(fill_method=None);m=b.pct_change(fill_method=None)
    f=pd.DataFrame(index=df.index)
    f['market_return']=m;f['excess_return']=r-m
    f['beta']=r.rolling(60).cov(m)/m.rolling(60).var().replace(0,np.nan)
    f['market_correlation']=r.rolling(60).corr(m)
    f['relative_strength']=c.pct_change(60,fill_method=None)-b.pct_change(60,fill_method=None)
    f['market_momentum']=b.pct_change(20,fill_method=None);f['market_trend']=b/b.rolling(200).mean()-1
    f['market_volatility']=m.rolling(20).std()*np.sqrt(252)
    return f
