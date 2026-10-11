import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from tests.test_foundation import fixture
from quant_ml.config import Config
from quant_ml.features import make_dataset
from quant_ml.models.validation import folds
from quant_ml.research.baselines import baseline_predictions
from quant_ml.backtest.engine import simulate

class BaselineTests(unittest.TestCase):
    def test_equal_weight_buyhold_drift_and_two_sided_cost(self):
        a=fixture(30);b=a.copy();b['open']=a.open.to_numpy()[::-1]
        p=pd.DataFrame({'date':a.index[:21],'ticker':'A','predicted_return':0.,'probability':.5,'volatility_20':.2})
        c=replace(Config(),universe=['A','B'])
        curve,trades,lots=simulate({'A':a,'B':b,'SPY':a},p,c,buy_hold_weights={'A':.5,'B':.5})
        rate=(c.commission_bps+c.slippage_bps)/10000
        gross=.5*a.open.iloc[21]/a.open.iloc[1]+.5*b.open.iloc[21]/b.open.iloc[1]
        self.assertAlmostEqual(curve.equity.iloc[-1],c.starting_capital/(1+rate)*gross*(1-rate))
        self.assertEqual(len(trades),4)
        self.assertEqual(len(lots),2)
        self.assertTrue((trades.cost>0).all())
        self.assertEqual(set(trades.date),{curve.index[0],curve.index[-1]})

    def test_all_baselines_share_dates_and_use_purged_training_means(self):
        f=fixture(600);c=replace(Config(),universe=['A'],train_min_dates=80,walk_forward_dates=40)
        panel,_=make_dataset({'A':f,'SPY':f},c)
        start=str(f.index[350].date());end=str(f.index[550].date())
        predictions,audit=baseline_predictions(panel,c,start,end)
        tr,va,_=next(folds(panel,start,end,40,80))
        first=predictions['historical_mean'].iloc[0]
        self.assertAlmostEqual(first.predicted_return,panel.loc[tr,'target_return'].mean())
        self.assertAlmostEqual(first.probability,panel.loc[tr,'target_up'].mean())
        for p in predictions.values():pd.testing.assert_frame_equal(p[['date','ticker']],predictions['naive_zero'][['date','ticker']])
        np.testing.assert_array_equal(predictions['naive_zero'].predicted_return,0)
        np.testing.assert_array_equal(predictions['naive_zero'].probability,.5)
        changed=panel.copy();changed.loc[changed.date>=f.index[450],['target_return','target_up']]=100
        after,_=baseline_predictions(changed,c,start,end)
        for name in predictions:
            cols=['predicted_return','probability']
            pd.testing.assert_frame_equal(predictions[name].loc[lambda x:x.date<f.index[450],cols],after[name].loc[lambda x:x.date<f.index[450],cols])

    def test_cash_baseline_and_conflicting_buyhold_reject(self):
        f=fixture(20);p=pd.DataFrame({'date':f.index[:11],'ticker':'A'})
        curve,trades,_=simulate({'A':f,'SPY':f},p,Config(),buy_hold_weights={})
        self.assertTrue(trades.empty);self.assertTrue((curve.equity==100000).all())
        with self.assertRaises(ValueError):simulate({'A':f,'SPY':f},p,Config(),'SPY',{'A':1})
