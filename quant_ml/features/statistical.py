import pandas as pd
import numpy as np
def statistical(df):
    c=df.close;r=c.pct_change(fill_method=None);f=pd.DataFrame(index=df.index)
    f['rolling_mean']=r.rolling(20).mean();f['rolling_median']=r.rolling(20).median();f['rolling_std']=r.rolling(20).std()
    f['skew']=r.rolling(60).skew();f['kurtosis']=r.rolling(60).kurt()
    f['zscore']=(c-c.rolling(20).mean())/c.rolling(20).std().replace(0,np.nan)
    f['rolling_min_return']=r.rolling(20).min();f['rolling_max_return']=r.rolling(20).max()
    f['distance_high']=c/c.rolling(60).max()-1;f['distance_low']=c/c.rolling(60).min()-1
    f['day_of_week']=df.index.dayofweek;f['month']=df.index.month;f['quarter']=df.index.quarter
    return f
