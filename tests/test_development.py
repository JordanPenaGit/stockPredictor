import contextlib,io,json,tempfile,unittest
from pathlib import Path
import numpy as np
from tests.test_foundation import fixture
from quant_ml.config import ROOT
from quant_ml.research.development import run_development
from quant_ml.research.data import validate_prices
from quant_ml.research.uncertainty import block_mean_interval

class DevelopmentTests(unittest.TestCase):
    def test_block_interval_constant_and_deterministic(self):
        a=block_mean_interval(np.ones(100)*.01,5,100,42,252)
        self.assertAlmostEqual(a['low'],2.52);self.assertAlmostEqual(a['high'],2.52)
        x=np.sin(np.arange(100)/7)
        self.assertEqual(block_mean_interval(x,5,100),block_mean_interval(x,5,100))
        with self.assertRaises(ValueError):block_mean_interval([1,2,np.nan],5)

    def test_data_bounds_and_missing_sessions_fail(self):
        f=fixture(30)
        validate_prices({'A':f,'SPY':f},['A'],'SPY','2020-01-01','2021-01-01')
        for data,end in [({'A':f.iloc[:-1],'SPY':f},'2021-01-01'),({'A':f,'SPY':f},'2020-01-10')]:
            with self.assertRaises(ValueError):validate_prices(data,['A'],'SPY','2020-01-01',end)

    def test_fixture_end_to_end_preserves_plan_and_refuses_overwrite(self):
        f=fixture(750)
        plan=json.loads((ROOT/'audit/development-plan.json').read_text())
        plan.update(data_start='2020-01-01',data_end_exclusive='2023-01-01',validation_start='2021-06-01',
                    universe=['A','B'],train_min_dates=80,walk_forward_dates=40)
        data={'A':f,'B':f.copy(),'SPY':f.copy()}
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)/'run'
            summary=run_development(plan,root,smoke=True,prices_override=data)
            self.assertFalse(summary['independent_holdout'])
            self.assertTrue((root/'frozen_protocol.json').exists())
            self.assertTrue((root/'artifact_hashes.json').exists())
            with self.assertRaises(FileExistsError):run_development(plan,root,smoke=True,prices_override=data)
            before=(root/'summary.json').read_bytes()
            second=run_development(plan,Path(tmp)/'second',smoke=True,prices_override=data)
            self.assertEqual(summary,second)
            self.assertEqual(before,(root/'summary.json').read_bytes())

    def test_failed_attempt_is_preserved(self):
        plan=json.loads((ROOT/'audit/development-plan.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'bad'
            with self.assertRaises(ValueError):run_development(plan,root,prices_override={})
            self.assertTrue((root/'FAILED.json').exists())
            self.assertTrue((root/'frozen_protocol.json').exists())
