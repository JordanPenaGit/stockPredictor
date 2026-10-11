import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def research_plots(root,curve,benchmarks,pred,importance):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig,axes=plt.subplots(3,2,figsize=(15,13))
    ax=axes[0,0];ax.plot(curve.index,curve.equity,label='Strategy',linewidth=2)
    for name,b in benchmarks.items():
        if name in ['SPY','AAPL']:ax.plot(b.index,b.equity,label=name,alpha=.75)
    ax.legend();ax.set_title('Costed out-of-sample equity (adjusted-price ledger)')
    axes[0,1].fill_between(curve.index,curve.drawdown*100,0,alpha=.5);axes[0,1].set_title('Drawdown %')
    rr=curve['return'];axes[1,0].plot(curve.index,rr.rolling(63).mean()/rr.rolling(63).std()*np.sqrt(252));axes[1,0].set_title('63-day rolling Sharpe (0 risk-free)')
    axes[1,1].plot(curve.index,rr.rolling(63).std()*np.sqrt(252)*100);axes[1,1].set_title('63-day annualized volatility %')
    p=pred.dropna(subset=['target_return']);axes[2,0].scatter(p.predicted_return,p.target_return,s=4,alpha=.12);axes[2,0].set(xlabel='Predicted return',ylabel='Realized horizon return',title='Untouched-period predictions')
    imp=importance.head(12).iloc[::-1];axes[2,1].barh(imp.feature,imp.importance);axes[2,1].set_title('Permutation importance (MSE increase; diagnostic only)')
    fig.tight_layout();fig.savefig(root/'research.png',dpi=140);plt.close(fig)
    from sklearn.calibration import calibration_curve
    f,m=calibration_curve(p.target_up,p.probability,n_bins=8,strategy='quantile');fig,ax=plt.subplots(figsize=(6,5));ax.plot([0,1],[0,1],'--',label='Ideal');ax.plot(m,f,'o-',label='Observed');ax.set(xlabel='Model probability',ylabel='Observed positive fraction',title='Final-period probability calibration');ax.legend();fig.tight_layout();fig.savefig(root/'calibration.png',dpi=140);plt.close(fig)
