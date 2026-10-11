from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import date
import json
import math
import re
ROOT = Path(__file__).resolve().parents[1]
@dataclass
class Config:
    universe: list = field(default_factory=lambda: ['AAPL','MSFT','AMZN','GOOGL','JPM','XOM','JNJ','WMT'])
    benchmark: str = 'SPY'
    start: str = '2015-01-01'
    end: str = field(default_factory=lambda: date.today().isoformat())
    horizon: int = 5
    target_threshold: float = 0.0
    train_min_dates: int = 756
    validation_start: str = '2020-01-01'
    test_start: str = '2024-01-01'
    walk_forward_dates: int = 126
    training_window: int = 0
    rebalance_days: int = 5
    starting_capital: float = 100000.0
    max_positions: int = 3
    max_weight: float = 0.34
    sizing: str = 'equal'
    commission_bps: float = 5.0
    slippage_bps: float = 5.0
    probability_threshold: float = 0.55
    return_threshold: float = 0.0
    strategy: str = 'ranking'
    feature_set: str = 'all'
    select_features: int = 24
    seed: int = 42
    models: list = field(default_factory=lambda:['linear','random_forest','gradient_boosting'])
    def validate(self):
        positive_ints = ['horizon', 'train_min_dates', 'walk_forward_dates',
                         'rebalance_days', 'max_positions', 'select_features']
        for name in positive_ints + ['training_window', 'seed']:
            value = getattr(self, name)
            minimum = 1 if name in positive_ints else 0
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if self.training_window and self.training_window < self.train_min_dates:
            raise ValueError('training_window must be zero or >= train_min_dates')
        for name in ['target_threshold', 'starting_capital', 'max_weight',
                     'commission_bps', 'slippage_bps', 'probability_threshold',
                     'return_threshold']:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{name} must be a finite number')
        def symbol(value):
            return isinstance(value, str) and re.fullmatch(r'[A-Z0-9^][A-Z0-9.^=-]{0,19}', value)
        if not isinstance(self.universe, list) or not self.universe or not all(symbol(s) for s in self.universe):
            raise ValueError('Universe must be a nonempty list of uppercase ticker symbols')
        if len(self.universe) != len(set(self.universe)):
            raise ValueError('Universe must be unique')
        if not symbol(self.benchmark):
            raise ValueError('Invalid benchmark ticker')
        if self.benchmark in self.universe:
            raise ValueError('Benchmark must not be in investable universe')
        for name in ['start', 'validation_start', 'test_start', 'end']:
            value = getattr(self, name)
            if not isinstance(value, str) or not re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', value):
                raise ValueError(f'{name} must be an ISO date (YYYY-MM-DD)')
            try:
                date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f'Invalid {name}: {value}') from exc
        if not self.start < self.validation_start < self.test_start < self.end:
            raise ValueError('Require start < validation_start < test_start < end')
        if not 0 < self.max_weight <= 1 or self.starting_capital <= 0:
            raise ValueError('Invalid portfolio configuration')
        if min(self.commission_bps, self.slippage_bps) < 0 or self.commission_bps + self.slippage_bps >= 10000:
            raise ValueError('Invalid costs')
        if self.strategy not in ['ranking', 'probability'] or self.sizing not in ['equal', 'inverse_volatility']:
            raise ValueError('Unknown strategy/sizing')
        if not 0 <= self.probability_threshold <= 1:
            raise ValueError('Invalid probability threshold')
        if self.feature_set not in ['all', 'technical', 'statistical', 'market']:
            raise ValueError('Unknown feature set')
        known_models = {'linear', 'random_forest', 'gradient_boosting', 'xgboost'}
        if (not isinstance(self.models, list) or not self.models
                or not all(isinstance(m, str) and m in known_models for m in self.models)
                or len(self.models) != len(set(self.models))):
            raise ValueError('Models must be a nonempty unique list of supported model names')
        return self
    def dict(self): return asdict(self)
def load_config(path=None):
    values=json.loads(Path(path).read_text()) if path else {}
    if not isinstance(values, dict):
        raise ValueError('Configuration must be a JSON object')
    unknown = set(values) - set(Config.__dataclass_fields__)
    if unknown:
        raise ValueError(f'Unknown configuration fields: {sorted(unknown)}')
    return Config(**values).validate()
