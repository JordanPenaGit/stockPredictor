"""Deterministic fixtures only; no network or research artifacts."""
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import pandas as pd
from tests.test_foundation import fixture
from quant_ml.config import Config
from quant_ml.features import make_dataset
from quant_ml.models.train import walk_forward, fit_pair
from quant_ml.backtest.engine import simulate
from quant_ml.backtest.metrics import performance

class ScientificTests(unittest.TestCase):
    def test_future_prices_leave_prior_predictions_and_orders_unchanged(self):
        f=fixture(700)
        c=replace(Config(),universe=['A'],train_min_dates=80,walk_forward_dates=40,
                  max_positions=1,max_weight=1,return_threshold=-1,rebalance_days=5)
        data={'A':f.copy(),'SPY':f.copy()}
        changed={s:d.copy() for s,d in data.items()}
        for d in changed.values():
            d.loc[f.index[550]:,['open','high','low','close','raw_close']]*=3
            d.loc[f.index[550]:,'volume']*=10
        outputs=[]
        for prices in [data,changed]:
            panel,features=make_dataset(prices,c)
            pred,audit=walk_forward(panel,features,'linear',c,str(f.index[350].date()),str(f.index[650].date()))
            curve,trades,lots=simulate(prices,pred,c)
            outputs.append((pred,trades))
            self.assertTrue(all(a['last_label_end']<a['cutoff'] for a in audit))
        cols=['date','ticker','predicted_return','probability']
        pd.testing.assert_frame_equal(outputs[0][0].loc[lambda p:p.date<f.index[550],cols],
                                      outputs[1][0].loc[lambda p:p.date<f.index[550],cols])
        pd.testing.assert_frame_equal(outputs[0][1].loc[lambda t:t.date<f.index[550]],
                                      outputs[1][1].loc[lambda t:t.date<f.index[550]])

    def test_schedule_dense_sparse_equivalence_and_missing_due_failure(self):
        f=fixture(30)
        p=pd.DataFrame({'date':f.index[:21],'ticker':'A','predicted_return':.1,'probability':.6,'volatility_20':.2})
        c=replace(Config(),universe=['A'],rebalance_days=5)
        data={'A':f,'SPY':f}
        dense=simulate(data,p,c)
        sparse=simulate(data,p.iloc[::5],c)
        for a,b in zip(dense,sparse):pd.testing.assert_frame_equal(a,b)
        self.assertEqual(dense[0].loc[lambda x:x.decision_due,'signal_date'].tolist(),list(f.index[[0,5,10,15]]))
        with self.assertRaisesRegex(ValueError,'scheduled'):
            simulate(data,p[p.date!=f.index[10]],c)

    def test_costed_identical_strategy_and_benchmark_match(self):
        f=fixture(30)
        p=pd.DataFrame({'date':f.index[:21],'ticker':'A','predicted_return':.1,'probability':.6,'volatility_20':.2})
        c=replace(Config(),universe=['A'],max_positions=1,max_weight=1,rebalance_days=30)
        a=simulate({'A':f,'SPY':f},p,c)
        b=simulate({'A':f,'SPY':f},p,c,'SPY')
        pd.testing.assert_frame_equal(a[0],b[0])
        self.assertEqual(performance(*a,c.starting_capital),performance(*b,c.starting_capital))
        self.assertEqual(len(a[1]),2)

    def test_missing_held_price_stops_not_stale_fill(self):
        f=fixture(30)
        p=pd.DataFrame({'date':f.index[:21],'ticker':'A','predicted_return':.1,'probability':.6,'volatility_20':.2})
        with self.assertRaisesRegex(ValueError,'Missing valuation'):
            simulate({'A':f.drop(f.index[3]),'SPY':f},p,replace(Config(),universe=['A']))

    def test_training_pipeline_statistics_match_only_training(self):
        f=fixture(500);c=replace(Config(),universe=['A'])
        panel,names=make_dataset({'A':f,'SPY':f},c)
        train=panel.iloc[:150].copy();train.loc[train.index[:5],names[0]]=np.nan
        pair=fit_pair(train,names,'linear',c)
        for pipeline in pair.values():
            np.testing.assert_allclose(pipeline['impute'].statistics_,np.nanmedian(train[names],axis=0))
            imputed=pipeline['impute'].transform(train[names])
            np.testing.assert_allclose(pipeline['scale'].mean_,imputed.mean(axis=0))
            support=pipeline['select'].get_support().copy()
            future=panel.iloc[150:].copy();future[names]=1e8
            pipeline.predict(future[names])
            np.testing.assert_array_equal(pipeline['select'].get_support(),support)

    def test_uniform_stale_market_fails_before_current_fit(self):
        from quant_ml.models.predict import current_predictions
        f=fixture(500);c=replace(Config(),universe=['A'],train_min_dates=80)
        data={'A':f,'SPY':f};_,features=make_dataset(data,c)
        bundle={'config':c,'features':features,'selected_model':'linear'}
        with patch('quant_ml.models.predict.load_bundle',return_value=bundle),patch('quant_ml.models.predict.MarketData') as market,patch('quant_ml.models.predict.fit_pair') as fit:
            market.return_value.universe.return_value=data
            with self.assertRaisesRegex(ValueError,'stale'):current_predictions('unit')
            fit.assert_not_called()

    def test_provider_bounds_duplicates_and_bad_values_fail(self):
        from quant_ml.data.downloader import validate_frame,DataError
        f=fixture(20)
        for bad,start,end in [(f,'2020-01-02','2021-01-01'),(f,'2020-01-01','2020-01-05'),(pd.concat([f,f.iloc[-1:]]),'2020-01-01','2021-01-01'),(f.assign(open=0),'2020-01-01','2021-01-01')]:
            with self.assertRaises(DataError):validate_frame(bad,'A',start,end)

if __name__=='__main__':unittest.main()
