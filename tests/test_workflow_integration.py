import contextlib,io,json,tempfile,unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
from tests.test_foundation import fixture
from quant_ml.config import Config
from quant_ml.workflow import train_research,assess,load_bundle

class WorkflowIntegration(unittest.TestCase):
    def test_fixture_freeze_assess_reuse_and_preserve_audit_pointer(self):
        f=fixture(1100)
        c=replace(Config(),universe=['A','B'],models=['linear'],start='2020-01-01',
                  validation_start='2021-06-01',test_start='2023-01-01',end='2024-06-01',
                  train_min_dates=80,walk_forward_dates=100)
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);(root/'models').mkdir()
            legacy=root/'models/holdout_audit.json';legacy.write_text('[]')
            pointer=root/'models/latest.json';pointer.write_text('{"run":"original"}')
            with patch('quant_ml.workflow.ROOT',root),patch('quant_ml.workflow.MarketData') as market,patch('quant_ml.research.plots.research_plots'),patch('quant_ml.research.reports.write_report'):
                market.return_value.universe.return_value={'A':f,'B':f.copy(),'SPY':f.copy()}
                manifest=train_research(c,'fixture')
                self.assertIn('bundle_sha256',manifest)
                assessment=assess('fixture')
                self.assertFalse(assessment['holdout_independence'])
                self.assertEqual(legacy.read_text(),'[]')
                self.assertEqual(pointer.read_text(),'{"run":"original"}')
                self.assertTrue((root/'models/holdout_access/fixture-started.json').exists())
                self.assertTrue((root/'models/holdout_access/fixture-complete.json').exists())
                saved=(root/'results/fixture/assessment.json').read_bytes()
                assess('fixture')
                self.assertEqual((root/'results/fixture/assessment.json').read_bytes(),saved)
                with self.assertRaises(FileExistsError):train_research(c,'fixture')
                bundle=root/'models/fixture/bundle.joblib';bundle.write_bytes(bundle.read_bytes()+b'tamper')
                with self.assertRaisesRegex(ValueError,'bytes'):load_bundle('fixture')
