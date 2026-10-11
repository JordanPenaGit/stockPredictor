from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import date
import json
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
        if self.horizon < 1 or self.rebalance_days < 1 or self.walk_forward_dates < 1: raise ValueError('Intervals must be positive')
        if not self.universe or len(self.universe)!=len(set(self.universe)): raise ValueError('Universe must be nonempty and unique')
        if self.benchmark in self.universe: raise ValueError('Benchmark must not be in investable universe')
        if not self.start < self.validation_start < self.test_start < self.end: raise ValueError('Require start < validation_start < test_start < end')
        if not 0 < self.max_weight <= 1 or self.max_positions < 1 or self.starting_capital <= 0: raise ValueError('Invalid portfolio configuration')
        if min(self.commission_bps,self.slippage_bps) < 0 or self.commission_bps+self.slippage_bps>=10000: raise ValueError('Invalid costs')
        if self.strategy not in ['ranking','probability'] or self.sizing not in ['equal','inverse_volatility']: raise ValueError('Unknown strategy/sizing')
        if not 0<=self.probability_threshold<=1: raise ValueError('Invalid probability threshold')
        return self
    def dict(self): return asdict(self)
def load_config(path=None):
    values=json.loads(Path(path).read_text()) if path else {}
    return Config(**values).validate()
