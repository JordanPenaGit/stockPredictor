"""Signal close t, entry open t+1, exit open t+h+1: h tradable sessions."""
import pandas as pd
import numpy as np
def make_targets(frame,horizon=5,threshold=0):
    if horizon<1:raise ValueError('Horizon must be positive')
    d=pd.Series(frame.index,index=frame.index)
    target=frame.open.shift(-(horizon+1))/frame.open.shift(-1)-1
    return pd.DataFrame({'target_return':target,'target_up':np.where(target.notna(),(target>threshold).astype(float),np.nan),'entry_at':d.shift(-1),'target_end':d.shift(-(horizon+1)),'feature_available_at':d},index=frame.index)
