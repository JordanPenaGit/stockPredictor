"""Synthetic/temp-only regressions. Never access stored research or the network."""
import contextlib
import io
import json
import tempfile
import subprocess
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from quant_ml.backtest.engine import construct_weights, simulate
from quant_ml.backtest.portfolio import Portfolio
from quant_ml.backtest.metrics import performance
from quant_ml.cli import main
from quant_ml.config import Config, load_config
from quant_ml.features import make_dataset
from quant_ml.models.validation import folds
from quant_ml.targets import make_targets
from quant_ml.utils import fingerprint, source_hash
from quant_ml.workflow import load_bundle, resolve_run, validate_features
from tests.test_foundation import fixture


class RegressionTests(unittest.TestCase):
    def signals(self, frame, n=6):
        return pd.DataFrame({'date': frame.index[:n], 'ticker': 'A',
                             'predicted_return': .1, 'probability': .8,
                             'volatility_20': .2})

    def test_final_open_only_liquidates(self):
        f = fixture(10)
        predictions = self.signals(f)
        predictions.loc[5, 'ticker'] = 'B'
        c = replace(Config(), universe=['A', 'B'], max_positions=1,
                    max_weight=1, rebalance_days=1)
        curve, trades, _ = simulate({'A': f, 'B': f, 'SPY': f}, predictions, c)
        terminal = trades[trades.date == curve.index[-1]]
        self.assertEqual(terminal.ticker.tolist(), ['A'])
        self.assertTrue((terminal.quantity < 0).all())
        self.assertEqual(curve.cash.iloc[-1], curve.equity.iloc[-1])

    def test_last_signal_cannot_create_zero_duration_trade(self):
        f = fixture(10)
        p = self.signals(f)
        p.loc[:4, 'predicted_return'] = -1
        curve, trades, _ = simulate({'A': f, 'SPY': f}, p,
                                    replace(Config(), rebalance_days=1))
        self.assertTrue(trades.empty)
        self.assertEqual(curve.cost.sum(), 0)

    def test_execution_calendar_rejects_unknown_or_unexecutable_dates(self):
        f = fixture(10)
        for date in [pd.Timestamp('2019-12-31'), f.index[-1], pd.NaT]:
            p = self.signals(f)
            p.loc[5, 'date'] = date
            with self.subTest(date=date), self.assertRaises(ValueError):
                simulate({'A': f, 'SPY': f}, p, Config())

    def test_duplicate_predictions_rejected(self):
        f = fixture(10)
        p = self.signals(f)
        with self.assertRaisesRegex(ValueError, 'unique'):
            simulate({'A': f, 'SPY': f}, pd.concat([p, p.iloc[:1]]), Config())

    def test_high_cost_rebalance_is_self_financing(self):
        for fee in [.5, .99, .9999]:
            with self.subTest(fee=fee):
                p = Portfolio(1000)
                _, cost = p.rebalance({'A': 1}, {'A': 100}, fee, 'entry')
                self.assertAlmostEqual(p.equity({'A': 100}), 1000 / (1 + fee))
                self.assertAlmostEqual(p.equity({'A': 100}) + cost, 1000)
                _, cost2 = p.rebalance({'B': 1}, {'A': 100, 'B': 50}, fee, 'switch')
                self.assertAlmostEqual(p.equity({'B': 50}), 1000 - cost - cost2)
                self.assertGreaterEqual(p.cash, 0)

    def test_portfolio_rejects_nonfinite_weights_and_costs(self):
        for weights, rate in [({'A': np.nan}, .01), ({'A': np.inf}, .01),
                              ({'A': 1}, np.nan), ({'A': 1}, -1), ({'A': 1}, 1)]:
            p = Portfolio(1000)
            with self.subTest(weights=weights, rate=rate), self.assertRaises(ValueError):
                p.rebalance(weights, {'A': 100}, rate, 'd')
            self.assertEqual(p.cash, 1000)
            self.assertEqual(p.trades, [])

    def test_invalid_risk_and_prediction_inputs_fail(self):
        p = self.signals(fixture(10), 1)
        for column, value in [('volatility_20', np.nan), ('volatility_20', -1),
                              ('predicted_return', np.inf), ('probability', 2)]:
            changed = p.copy()
            changed[column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                construct_weights(changed, replace(Config(), sizing='inverse_volatility'))

    def test_invalid_target_opens_remain_unlabeled(self):
        for value in [0, -1, np.inf, np.nan]:
            f = fixture(20)
            f.loc[f.index[3], 'open'] = value
            target = make_targets(f, 1)
            for row in [1, 2]:
                self.assertTrue(pd.isna(target.target_return.iloc[row]))
                self.assertTrue(pd.isna(target.target_up.iloc[row]))

    def test_target_calendar_and_horizon_validation(self):
        f = fixture(20)
        for horizon in [0, -1, True, 1.5]:
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                make_targets(f, horizon)
        for bad in [f.iloc[::-1], pd.concat([f, f.iloc[-1:]])]:
            with self.assertRaises(ValueError):
                make_targets(bad)

    def test_missing_session_does_not_compress_target_horizon(self):
        f = fixture(300)
        stock = f.drop(f.index[251])
        panel, _ = make_dataset({'A': stock, 'SPY': f}, replace(Config(), universe=['A'], horizon=1))
        row = panel[panel.date == f.index[250]].iloc[0]
        self.assertTrue(pd.isna(row.target_return))
        self.assertEqual(row.target_end, f.index[252])

    def test_rolling_window_counts_matured_dates(self):
        f = fixture(400)
        panel = make_targets(f, 5).reset_index()
        tr, va, _ = next(folds(panel, str(f.index[300].date()), str(f.index[320].date()),
                              step=20, min_train=100, window=100))
        self.assertEqual(panel.loc[tr, 'date'].nunique(), 100)
        self.assertLess(panel.loc[tr, 'target_end'].max(), panel.loc[va, 'date'].min())

    def test_folds_reject_ambiguous_row_indices(self):
        panel = make_targets(fixture(400)).reset_index()
        panel.index = [0] * len(panel)
        with self.assertRaisesRegex(ValueError, 'index'):
            list(folds(panel, '2021-01-01', '2021-04-01', min_train=10))

    def test_config_rejects_invalid_values(self):
        changes = [{'start': '2015-02-30'}, {'horizon': 1.5}, {'rebalance_days': True},
                   {'starting_capital': np.inf}, {'commission_bps': np.nan},
                   {'target_threshold': np.nan}, {'select_features': 0},
                   {'training_window': -1}, {'training_window': 100},
                   {'models': []}, {'models': ['unknown']}, {'feature_set': 'unknown'},
                   {'universe': 'AAPL'}, {'universe': ['aapl']}, {'benchmark': '../SPY'}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(Config(), **change).validate()
        Config().validate()

    def test_config_file_requires_known_object_fields(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'config.json'
            for value in [[], {'typo': 5}]:
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_cli_rejects_bad_inputs_before_inference(self):
        for command, argv in [('screen', ['screen.py', '--top', '0']),
                              ('screen', ['screen.py', '--tickers', '../A']),
                              ('analyze', ['analyze.py', '../A'])]:
            with self.subTest(command=command), patch('sys.argv', argv), \
                    patch('quant_ml.models.predict.current_predictions') as infer, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(command)
                self.assertEqual(raised.exception.code, 2)
                infer.assert_not_called()

    def test_cli_normalizes_and_deduplicates_tickers(self):
        with patch('sys.argv', ['analyze.py', ' aapl ']), \
                patch('quant_ml.models.predict.current_predictions', return_value=(None, None)) as infer, \
                patch('quant_ml.models.predict.print_analysis'):
            self.assertEqual(main('analyze'), 0)
            infer.assert_called_once_with(None, ['AAPL'], refresh=True)

    def test_latest_run_pointer_cannot_escape_models(self):
        with tempfile.TemporaryDirectory() as d, patch('quant_ml.workflow.ROOT', Path(d)):
            (Path(d) / 'models').mkdir()
            (Path(d) / 'models/latest.json').write_text('{"run":"../outside"}')
            with self.assertRaises(ValueError):
                resolve_run()
            with self.assertRaises(ValueError):
                load_bundle('../outside')

    def test_frozen_bundle_metadata_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d, patch('quant_ml.workflow.ROOT', Path(d)):
            folder = Path(d) / 'models/unit'
            folder.mkdir(parents=True)
            c = Config()
            bundle = {'config': c, 'features': ['a', 'b'], 'selected_model': 'linear',
                      'frozen_pair': {'regression': SimpleNamespace(feature_names_in_=np.array(['a', 'b']))}}
            joblib.dump(bundle, folder / 'bundle.joblib')
            manifest = {'run': 'unit', 'config': c.dict(), 'config_hash': fingerprint(c.dict()),
                        'features': ['a', 'b'], 'selected_model': 'linear'}
            (folder / 'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(load_bundle('unit')['features'], ['a', 'b'])
            for key, value in [('config_hash', 'bad'), ('selected_model', 'random_forest'),
                               ('features', ['b', 'a']), ('run', 'other')]:
                changed = {**manifest, key: value}
                (folder / 'manifest.json').write_text(json.dumps(changed))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    load_bundle('unit')

    def test_model_source_not_ignored_as_artifacts(self):
        root = Path(__file__).resolve().parents[1]
        if not (root / '.git').exists():
            self.skipTest('Source archive without git metadata')
        proc = subprocess.run(['git', 'check-ignore', '--no-index', 'quant_ml/models/train.py',
                               'quant_ml/models/predict.py', 'quant_ml/models/validation.py'],
                              cwd=root, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        artifacts = subprocess.run(['git', 'check-ignore', 'models/unit/bundle.joblib',
                                    'results/unit/equity.csv'], cwd=root, capture_output=True, text=True)
        self.assertEqual(artifacts.returncode, 0)
        self.assertEqual(len(artifacts.stdout.splitlines()), 2)

    def test_frozen_pipeline_refit_does_not_use_registry(self):
        from quant_ml.models.train import fit_pair, walk_forward
        from quant_ml.models.registry import make_model
        c = replace(Config(), universe=['A'], train_min_dates=50, walk_forward_dates=20,
                    select_features=1)
        dates = pd.bdate_range('2020-01-01', periods=100)
        x = np.sin(np.arange(100))
        panel = pd.DataFrame({'date': dates, 'ticker': 'A', 'x': x,
                              'target_return': x / 100, 'target_up': (x > 0).astype(float),
                              'target_end': dates + pd.Timedelta(days=1), 'volatility_20': .2})
        template = {task: make_model('linear', task, c, 1)
                    for task in ['regression', 'classification']}
        template['regression'].set_params(model__alpha=123)
        template['classification'].set_params(model__C=.321)
        with patch('quant_ml.models.train.make_model', side_effect=AssertionError('registry drift')):
            pair = fit_pair(panel.iloc[:50], ['x'], 'linear', c, template=template)
            predictions, audit = walk_forward(panel, ['x'], 'linear', c,
                                               str(dates[60].date()), str(dates[99].date()),
                                               template=template)
        self.assertEqual(pair['regression'].named_steps['model'].alpha, 123)
        self.assertEqual(pair['classification'].named_steps['model'].C, .321)
        self.assertFalse(hasattr(template['regression'].named_steps['model'], 'coef_'))
        self.assertEqual(len(predictions), 39)
        self.assertEqual(len(audit), 2)

    def test_source_hash_is_content_based_and_excludes_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'quant_ml').mkdir()
            path = root / 'quant_ml/module.py'
            path.write_text('x = 1')
            before = source_hash(root)
            (root / 'models').mkdir()
            (root / 'models/artifact.py').write_text('not source')
            self.assertEqual(before, source_hash(root))
            path.write_text('x = 2')
            self.assertNotEqual(before, source_hash(root))

    def test_annualization_uses_elapsed_open_intervals(self):
        f = fixture(10)
        f['open'] = 100 * 1.001 ** np.arange(len(f))
        c = replace(Config(), universe=['A'], max_weight=1, max_positions=1,
                    commission_bps=0, slippage_bps=0)
        curve, trades, realized = simulate({'A': f, 'SPY': f}, self.signals(f), c)
        metrics = performance(curve, trades, realized, c.starting_capital)
        self.assertEqual(metrics['periods'], len(curve)-1)
        self.assertAlmostEqual(metrics['cagr'], 1.001**252-1)
        self.assertAlmostEqual(metrics['exposure'], 1)

    def test_entry_cost_is_in_first_elapsed_return(self):
        f = fixture(10)
        f['open'] = 100
        c = replace(Config(), universe=['A'], max_weight=1, max_positions=1,
                    rebalance_days=10)
        curve, trades, realized = simulate({'A': f, 'SPY': f}, self.signals(f), c)
        metrics = performance(curve, trades, realized, c.starting_capital)
        returns = curve['return'].iloc[1:].copy()
        returns.iloc[0] = (1+curve['return'].iloc[0])*(1+returns.iloc[0])-1
        self.assertAlmostEqual((1+returns).prod()-1, metrics['total_return'])
        self.assertAlmostEqual(metrics['volatility'], returns.std()*np.sqrt(252))
        self.assertAlmostEqual(metrics['cagr'], (1+metrics['total_return'])**(252/len(returns))-1)

    def test_current_refit_rejects_stale_candidate_before_training(self):
        from quant_ml.models.predict import current_predictions
        f = fixture(400)
        c = replace(Config(), universe=['A'], train_min_dates=100)
        data = {'A': f.iloc[:-1], 'SPY': f}
        _, features = make_dataset(data, c)
        bundle = {'config': c, 'features': features, 'selected_model': 'linear'}
        with patch('quant_ml.models.predict.load_bundle', return_value=bundle), \
                patch('quant_ml.models.predict.MarketData') as market, \
                patch('quant_ml.models.predict.fit_pair') as fit:
            market.return_value.universe.return_value = data
            with self.assertRaisesRegex(ValueError, 'fresh'):
                current_predictions('unit')
            fit.assert_not_called()

    def test_feature_order_is_part_of_inference_contract(self):
        with self.assertRaisesRegex(ValueError, 'schema'):
            validate_features(['b', 'a'], {'features': ['a', 'b']})


if __name__ == '__main__':
    unittest.main()
