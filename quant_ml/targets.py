"""Signal close t, entry open t+1, exit open t+h+1: h tradable sessions."""
import pandas as pd
import numpy as np
def make_targets(frame,horizon=5,threshold=0):
    if type(horizon) is not int or horizon<1:raise ValueError('Horizon must be a positive integer')
    if not np.isfinite(threshold):raise ValueError('Target threshold must be finite')
    if not isinstance(frame.index,pd.DatetimeIndex) or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError('Target dates must be unique and increasing')
    d=pd.Series(frame.index,index=frame.index)
    opening=frame.open.where(np.isfinite(frame.open)&(frame.open>0))
    target=opening.shift(-(horizon+1))/opening.shift(-1)-1
    target=target.where(np.isfinite(target))
    return pd.DataFrame({'target_return':target,'target_up':np.where(target.notna(),(target>threshold).astype(float),np.nan),'entry_at':d.shift(-1),'target_end':d.shift(-(horizon+1)),'feature_available_at':d},index=frame.index)
