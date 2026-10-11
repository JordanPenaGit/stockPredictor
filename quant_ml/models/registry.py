from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,HistGradientBoostingRegressor,HistGradientBoostingClassifier

def available_models():
    names=['linear','random_forest','gradient_boosting']
    try:import xgboost;names.append('xgboost')
    except ImportError:pass
    return names

def make_model(name,task,config,n_features):
    reg=task=='regression'
    if name=='linear':estimator=Ridge(alpha=10) if reg else LogisticRegression(C=.1,max_iter=1500,random_state=config.seed)
    elif name=='random_forest':
        cls=RandomForestRegressor if reg else RandomForestClassifier
        estimator=cls(n_estimators=70,max_depth=5,min_samples_leaf=40,max_features=.7,n_jobs=2,random_state=config.seed)
    elif name=='gradient_boosting':
        cls=HistGradientBoostingRegressor if reg else HistGradientBoostingClassifier
        estimator=cls(max_iter=70,max_leaf_nodes=9,learning_rate=.05,l2_regularization=10,min_samples_leaf=40,early_stopping=False,random_state=config.seed)
    elif name=='xgboost':
        from xgboost import XGBRegressor,XGBClassifier
        cls=XGBRegressor if reg else XGBClassifier
        estimator=cls(n_estimators=70,max_depth=3,learning_rate=.04,reg_lambda=10,subsample=1,colsample_bytree=.8,n_jobs=2,random_state=config.seed,tree_method='hist')
    else:raise ValueError(f'Unknown/unavailable model: {name}')
    return Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler()),('select',SelectKBest(f_regression if reg else f_classif,k=min(config.select_features,n_features))),('model',estimator)])
