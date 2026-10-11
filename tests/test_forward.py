import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from tests.test_foundation import fixture
from quant_ml.research.forward import init_protocol,capture_window,capture,evaluate_capture,schedule,load_protocol,sha

class ForwardTests(unittest.TestCase):
    def test_after_close_next_open_and_early_close_calendar(self):
        protocol={'config':{'horizon':5},'created_at':'2025-11-20T18:00:00Z','not_before_signal_date':'2025-11-21'}
        signal,entry,exit_open,mature=capture_window(pd.Timestamp('2025-11-28T18:31:00Z'),protocol)
        self.assertEqual(str(signal.date()),'2025-11-28')
        self.assertEqual(str(entry), '2025-12-01 14:30:00+00:00')
        self.assertGreater(mature,exit_open)
        with self.assertRaises(ValueError):capture_window(pd.Timestamp('2025-11-28T18:10:00Z'),protocol)
        with self.assertRaises(ValueError):capture_window(pd.Timestamp('2025-12-01T15:00:00Z'),protocol)

    def test_prospective_capture_mature_evaluation_and_append_only(self):
        dates=schedule('2020-01-01','2026-10-20').index
        f=fixture(len(dates));f.index=pd.DatetimeIndex(dates,name='date')
        freeze=pd.Timestamp('2026-10-11T18:00:00Z');now=pd.Timestamp('2026-10-12T21:00:00Z')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'shadow'
            with patch('quant_ml.research.forward.utcnow',return_value=freeze):p=init_protocol(root)
            with self.assertRaises(FileExistsError):init_protocol(root)
            data={s:f.loc[:'2026-10-12'].copy() for s in [*p['config']['universe'],'SPY']}
            with patch('quant_ml.research.forward.utcnow',return_value=now),patch('quant_ml.research.forward.download_universe',return_value=data):
                r=capture(root)
                self.assertFalse(r['orders_placed'])
                self.assertLess(pd.Timestamp(r['captured_at']),pd.Timestamp(r['entry_open']))
                self.assertLess(pd.Timestamp(r['training_max_label_end']),pd.Timestamp(r['signal_date']))
                with self.assertRaises(FileExistsError):capture(root)
                with self.assertRaisesRegex(ValueError,'matured'):evaluate_capture(root,'2026-10-12')
            original=sha(root/'captures/2026-10-12/record.json')
            with patch('quant_ml.research.forward.utcnow',return_value=pd.Timestamp('2026-10-20T21:00:00Z')),patch('quant_ml.research.forward.YahooProvider') as provider:
                provider.return_value.download.return_value=f.loc['2026-10-13':'2026-10-20']
                result=evaluate_capture(root,'2026-10-12')
                self.assertEqual(result['dates'],1)
                with self.assertRaises(FileExistsError):evaluate_capture(root,'2026-10-12')
            self.assertEqual(sha(root/'captures/2026-10-12/record.json'),original)
            with patch('quant_ml.research.forward.source_hash',return_value='changed'):
                with self.assertRaises(ValueError):load_protocol(root)
            bad=root/'captures/2026-10-12/forecasts.csv';bad.write_text(bad.read_text()+'tamper')
            with self.assertRaisesRegex(ValueError,'changed'):evaluate_capture(root,'2026-10-12')

    def test_no_backdated_pre_freeze_signal(self):
        p={'config':{'horizon':5},'created_at':'2026-10-11T18:00:00Z','not_before_signal_date':'2026-10-12'}
        with self.assertRaisesRegex(ValueError,'precedes'):capture_window(pd.Timestamp('2026-10-11T19:00:00Z'),p)
