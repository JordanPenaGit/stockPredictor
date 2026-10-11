import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

def importance(pair,frame,features,seed=42):
    model=pair['regression'];names=np.asarray(features)[model.named_steps['select'].get_support()]
    est=model.named_steps['model']
    if hasattr(est,'feature_importances_'):native=est.feature_importances_
    elif hasattr(est,'coef_'):native=np.abs(est.coef_)
    else:native=np.full(len(names),np.nan)
    native=pd.DataFrame({'feature':names,'native_importance':native}).sort_values('native_importance',ascending=False)
    # Contiguous suffix, no randomized selection of temporal observations.
    sample=frame.dropna(subset=['target_return']).tail(800)
    perm=permutation_importance(model,sample[features],sample.target_return,scoring='neg_mean_squared_error',n_repeats=3,random_state=seed,n_jobs=1)
    return native,pd.DataFrame({'feature':features,'importance':perm.importances_mean,'std':perm.importances_std}).sort_values('importance',ascending=False)
