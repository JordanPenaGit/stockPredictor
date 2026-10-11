import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.base import clone
from .registry import make_model
from .validation import folds

def fit_pair(train,features,name,config,template=None):
    if train.target_up.nunique()<2:raise ValueError('Training labels need both classes')
    if template is None:
        reg=make_model(name,'regression',config,len(features));clf=make_model(name,'classification',config,len(features))
    else:
        # Clone drops fitted state but preserves the complete frozen preprocessing/model protocol.
        reg=clone(template['regression']);clf=clone(template['classification'])
    reg.fit(train[features],train.target_return);clf.fit(train[features],train.target_up.astype(int))
    return {'regression':reg,'classification':clf}

def predict_pair(pair,frame,features):
    out=frame[['date','ticker','target_return','target_up','target_end','volatility_20']].copy()
    out['market_trend']=frame['market_trend'] if 'market_trend' in frame else np.nan
    out['predicted_return']=pair['regression'].predict(frame[features]);out['probability']=pair['classification'].predict_proba(frame[features])[:,1]
    return out

def walk_forward(panel,features,name,config,start,end,template=None):
    predictions=[];audit=[]
    for tr,va,meta in folds(panel,start,end,config.walk_forward_dates,config.train_min_dates,config.training_window):
        pair=fit_pair(panel.loc[tr],features,name,config,template=template)
        predictions.append(predict_pair(pair,panel.loc[va],features));audit.append(meta)
    if not predictions:raise ValueError('No validation folds')
    return pd.concat(predictions).sort_values(['date','ticker']).reset_index(drop=True),audit

def evaluate(p,cutoff=None):
    p=p[p.target_return.notna()]
    if cutoff is not None:p=p[p.target_end<pd.Timestamp(cutoff)]
    if p.empty:raise ValueError('No matured evaluation labels')
    y=p.target_return;pred=p.predicted_return;actual=p.target_up.astype(int);prob=p.probability;binary=(prob>=.5).astype(int)
    frac,mean=calibration_curve(actual,prob,n_bins=8,strategy='quantile')
    return {'rows':len(p),'mae':float(mean_absolute_error(y,pred)),'rmse':float(np.sqrt(mean_squared_error(y,pred))),'r2':float(r2_score(y,pred)),'directional_accuracy':float(((pred>0)==(y>0)).mean()),'correlation':float(y.corr(pred)) if pred.std()>0 else None,'accuracy':float(accuracy_score(actual,binary)),'precision':float(precision_score(actual,binary,zero_division=0)),'recall':float(recall_score(actual,binary,zero_division=0)),'f1':float(f1_score(actual,binary,zero_division=0)),'roc_auc':float(roc_auc_score(actual,prob)) if actual.nunique()>1 else None,'brier_score':float(brier_score_loss(actual,prob)),'confusion_matrix':confusion_matrix(actual,binary,labels=[0,1]).tolist(),'calibration':{'mean_probability':mean.tolist(),'observed_fraction':frac.tolist()},'majority_class_accuracy':float(max(actual.mean(),1-actual.mean()))}
