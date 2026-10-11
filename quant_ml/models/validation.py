import pandas as pd

def folds(panel,start,end,step=126,min_train=756,window=0):
    for name,value,minimum in [('step',step,1),('min_train',min_train,1),('window',window,0)]:
        if type(value) is not int or value<minimum:raise ValueError(f'{name} must be an integer >= {minimum}')
    if pd.Timestamp(start)>=pd.Timestamp(end):raise ValueError('Fold start must precede end')
    if not panel.index.is_unique:raise ValueError('Panel row index must be unique')
    if panel.date.isna().any():raise ValueError('Panel dates cannot be missing')
    if 'ticker' in panel and panel.duplicated(['date','ticker']).any():raise ValueError('Duplicate date/ticker rows')
    dates=pd.DatetimeIndex(sorted(panel.date.unique()));eligible=dates[(dates>=pd.Timestamp(start))&(dates<pd.Timestamp(end))]
    for offset in range(0,len(eligible),step):
        test_dates=eligible[offset:offset+step];cutoff=test_dates[0]
        train=(panel.date<cutoff)&(panel.target_end<cutoff)&panel.target_return.notna()
        # Count eligible matured dates, consistent with frozen and current refits.
        if window:train &= panel.date.isin(sorted(panel.loc[train,'date'].unique())[-window:])
        valid=panel.date.isin(test_dates)
        if panel.loc[train,'date'].nunique()<min_train:raise ValueError(f'Insufficient training history before {cutoff}')
        assert panel.loc[train,'target_end'].max()<cutoff
        assert not set(panel.loc[train,'date']) & set(test_dates)
        yield panel.index[train],panel.index[valid],{'cutoff':str(cutoff.date()),'train_start':str(panel.loc[train,'date'].min().date()),'train_end':str(panel.loc[train,'date'].max().date()),'last_label_end':str(panel.loc[train,'target_end'].max().date()),'prediction_end':str(test_dates[-1].date()),'train_rows':int(train.sum())}
