import unittest,tempfile
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from tests.test_foundation import fixture
from quant_ml.config import Config
from quant_ml.data.cache import PriceCache
from quant_ml.data.downloader import MarketData,DataError
from quant_ml.features import make_dataset
from quant_ml.models.train import fit_pair,predict_pair
from quant_ml.models.validation import folds
from quant_ml.backtest.portfolio import Portfolio
from quant_ml.backtest.engine import construct_weights,simulate
from quant_ml.backtest.metrics import performance
class Safeguards(unittest.TestCase):
    def test_cache_roundtrip_and_no_second_download(self):
        class Fake:
            def __init__(self):self.calls=0
            def download(self,*args):self.calls+=1;return fixture()
        with tempfile.TemporaryDirectory() as d:
            p=Fake();m=MarketData(p,Path(d)/'cache.sqlite');a=m.get('A','2020','2022');b=m.get('A','2020','2022');pd.testing.assert_frame_equal(a,b);self.assertEqual(p.calls,1);m.get('A','2020','2022',True);self.assertEqual(p.calls,2)
    def test_invalid_symbol(self):
        with tempfile.TemporaryDirectory() as d:
            m=MarketData(cache_path=Path(d)/'x.db')
            with self.assertRaises(DataError):m.get('../../A','2020','2022')
    def test_rolling_window_and_label_equality_purge(self):
        from quant_ml.targets import make_targets
        f=fixture(700);p=make_targets(f).reset_index()
        for tr,va,meta in folds(p,'2021-01-01','2022-01-01',60,100,200):
            self.assertLessEqual(p.loc[tr,'date'].nunique(),200);self.assertTrue((p.loc[tr,'target_end']<p.loc[va,'date'].min()).all())
    def test_fit_transforms_only_training(self):
        f=fixture(700);c=replace(Config(),universe=['A'],train_min_dates=100)
        panel,names=make_dataset({'A':f,'SPY':f},c);train=panel.iloc[:250].dropna(subset=['target_return']);pair=fit_pair(train,names,'linear',c)
        original=pair['regression'].named_steps['scale'].mean_.copy();future=panel.iloc[300:310].copy();future[names]=1000000
        pair['regression'].predict(future[names]);np.testing.assert_array_equal(original,pair['regression'].named_steps['scale'].mean_)
    def test_cost_both_sides_and_drift_turnover(self):
        p=Portfolio(1000);p.rebalance({'A':.5,'B':.5},{'A':100,'B':100},.001,'a')
        turn,cost=p.rebalance({'A':.5,'B':.5},{'A':200,'B':100},.001,'b');self.assertGreater(turn,.3);self.assertGreater(cost,0)
        _,exitcost=p.rebalance({},{'A':200,'B':100},.001,'c');self.assertGreater(exitcost,0);self.assertGreaterEqual(p.cash,0)
    def test_weight_cap_and_cash_remainder(self):
        p=pd.DataFrame({'ticker':['A','B'],'predicted_return':[.1,.2],'probability':[.8,.9],'volatility_20':[.2,.4]})
        c=replace(Config(),max_weight=.3);w=construct_weights(p,c);self.assertAlmostEqual(sum(w.values()),.6)
    def test_probability_threshold(self):
        p=pd.DataFrame({'ticker':['A','B'],'predicted_return':[.1,.2],'probability':[.51,.9],'volatility_20':[.2,.4]})
        w=construct_weights(p,replace(Config(),strategy='probability',probability_threshold=.7));self.assertEqual(set(w),{'B'})
    def test_no_fills_from_future_signal(self):
        f=fixture(20);p=pd.DataFrame({'date':f.index[:15],'ticker':'A','predicted_return':.1,'probability':.8,'volatility_20':.2});c=replace(Config(),universe=['A'],max_positions=1,max_weight=1,rebalance_days=1)
        _,t,_=simulate({'A':f,'SPY':f},p,c);changed=p.copy();changed.loc[changed.date>=f.index[10],'predicted_return']=-1
        _,u,_=simulate({'A':f,'SPY':f},changed,c);pd.testing.assert_frame_equal(t[t.date<=f.index[10]].reset_index(drop=True),u[u.date<=f.index[10]].reset_index(drop=True))
    def test_initial_cost_in_drawdown(self):
        f=fixture(10);f.loc[:,['open','close']]=100;p=pd.DataFrame({'date':f.index[:6],'ticker':'A','predicted_return':.1,'probability':.8,'volatility_20':.2});c=replace(Config(),universe=['A'],max_weight=1,max_positions=1)
        curve,t,r=simulate({'A':f,'SPY':f},p,c);m=performance(curve,t,r,c.starting_capital);self.assertLess(curve.drawdown.iloc[0],0);self.assertLess(m['total_return'],0);self.assertAlmostEqual(c.starting_capital-curve.equity.iloc[-1],curve.cost.sum())
    def test_benchmark_exact_dates(self):
        f=fixture(12);p=pd.DataFrame({'date':f.index[:7],'ticker':'A','predicted_return':.1,'probability':.8,'volatility_20':.2});c=replace(Config(),universe=['A']);a,_,_=simulate({'A':f,'SPY':f},p,c);b,_,_=simulate({'A':f,'SPY':f},p,c,'SPY');self.assertTrue(a.index.equals(b.index))
if __name__=='__main__':unittest.main()
