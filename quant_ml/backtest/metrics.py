import numpy as np
import pandas as pd

def performance(curve,trades,realized,capital):
    r=curve['return'];n=len(r);total=curve.equity.iloc[-1]/capital-1
    years=max(n/252,1/252);cagr=(1+total)**(1/years)-1;vol=r.std(ddof=1)*np.sqrt(252)
    down=np.sqrt(np.mean(np.minimum(r,0)**2))*np.sqrt(252);dd=float(curve.drawdown.min())
    pnl=realized.pnl if len(realized) else pd.Series(dtype=float);gains=pnl[pnl>0].sum();losses=-pnl[pnl<0].sum()
    return {'total_return':float(total),'cagr':float(cagr),'volatility':float(vol),'sharpe':float(r.mean()*252/vol) if vol>0 else None,'sortino':float(r.mean()*252/down) if down>0 else None,'max_drawdown':dd,'calmar':float(cagr/abs(dd)) if dd<0 else None,'win_rate':float((pnl>0).mean()) if len(pnl) else None,'profit_factor':float(gains/losses) if losses>0 else None,'average_trade_return':float(realized['return'].mean()) if len(realized) else None,'number_of_trades':int(len(trades)),'closed_lots':int(len(realized)),'exposure':float(curve.exposure.mean()),'turnover':float(curve.turnover.sum()),'costs':float(curve.cost.sum()),'periods':n,'start':str(curve.index[0].date()),'end':str(curve.index[-1].date())}
