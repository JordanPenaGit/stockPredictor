import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from quant_ml.config import Config
from quant_ml.features import make_features
from quant_ml.targets import make_targets
from quant_ml.models.validation import folds
from quant_ml.backtest.portfolio import Portfolio
from quant_ml.backtest.engine import simulate
from quant_ml.backtest.metrics import performance

def fixture(n=400):
    # Synthetic deterministic UNIT TEST fixture; never used in market reports.
    idx=pd.bdate_range('2020-01-01',periods=n,name='date');p=100+np.arange(n)*.1+np.sin(np.arange(n)/8)
    return pd.DataFrame({'open':p,'high':p+1,'low':p-1,'close':p+.2,'raw_close':p+.2,'volume':1000+np.arange(n)},index=idx)
class Foundation(unittest.TestCase):
    def test_target_alignment(self):
        f=fixture();t=make_targets(f,5);self.assertAlmostEqual(t.target_return.iloc[10],f.open.iloc[16]/f.open.iloc[11]-1);self.assertEqual(t.target_end.iloc[10],f.index[16]);self.assertTrue(t.target_return.iloc[-6:].isna().all())
    def test_horizon_one(self):
        f=fixture();t=make_targets(f,1);self.assertAlmostEqual(t.target_return.iloc[0],f.open.iloc[2]/f.open.iloc[1]-1)
    def test_features_no_future(self):
        f=fixture();before=make_features(f,f);changed=f.copy();changed.iloc[320:,changed.columns.get_indexer(['open','high','low','close'])]*=10
        pd.testing.assert_frame_equal(before.loc[:f.index[319]],make_features(changed,changed).loc[:f.index[319]])
    def test_feature_math(self):
        f=fixture();r=make_features(f,f);self.assertAlmostEqual(r.return_5.iloc[-1],f.close.iloc[-1]/f.close.iloc[-6]-1);self.assertAlmostEqual(r.beta.iloc[-1],1)
    def test_purge_by_date_and_label_end(self):
        f=fixture();p=pd.concat([make_targets(f).assign(ticker=s) for s in ['A','B']]).reset_index();p['date']=pd.to_datetime(p.date)
        for tr,va,_ in folds(p,'2021-01-01','2021-07-01',30,100):
            self.assertLess(p.loc[tr,'target_end'].max(),p.loc[va,'date'].min());self.assertFalse(set(p.loc[tr,'date'])&set(p.loc[va,'date']))
    def test_portfolio_self_financing(self):
        p=Portfolio(1000);turn,cost=p.rebalance({'A':1},{'A':100},.01,'d');self.assertAlmostEqual(p.equity({'A':100}),1000-cost);self.assertGreaterEqual(p.cash,0)
        old=p.equity({'A':120});_,cost=p.rebalance({}, {'A':120},.01,'e');self.assertAlmostEqual(p.cash,old-cost);self.assertEqual(len(p.realized),1)
    def test_missing_open_no_fill(self):
        p=Portfolio(1000);p.rebalance({'A':1},{'A':float('nan')},.001,'d');self.assertEqual(p.cash,1000);self.assertFalse(p.trades)
    def test_missing_held_valuation_fails(self):
        p=Portfolio(1000);p.rebalance({'A':1},{'A':100},0,'d')
        with self.assertRaises(ValueError):p.equity({})
    def test_next_open_no_close_fill(self):
        f=fixture(8);p=pd.DataFrame({'date':f.index[:5],'ticker':'A','predicted_return':.1,'probability':.8,'volatility_20':.2})
        cfg=replace(Config(),universe=['A'],max_positions=1,max_weight=1,commission_bps=0,slippage_bps=0)
        curve,trades,realized=simulate({'A':f,'SPY':f},p,cfg)
        self.assertEqual(trades.iloc[0].date,f.index[1]);self.assertEqual(trades.iloc[0].price,f.open.iloc[1]);self.assertEqual(curve.index[-1],f.index[5])
        m=performance(curve,trades,realized,100000);self.assertAlmostEqual(m['total_return'],f.open.iloc[5]/f.open.iloc[1]-1)
if __name__=='__main__':unittest.main()
