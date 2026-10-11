Build a complete Python machine-learning quantitative analysis and research system for stocks.

The goal is NOT to create a toy "predict tomorrow's stock price" model. Build a proper quantitative research pipeline that can collect historical market data, engineer features, train ML models, backtest strategies, evaluate them against benchmarks, and identify potentially useful predictive signals while avoiding look-ahead bias and data leakage.

Use Python 3.

Core libraries:
- pandas
- numpy
- scikit-learn
- yfinance
- scipy
- matplotlib
- seaborn only if necessary
- xgboost or lightgbm if available
- joblib
- sqlite3 or another simple local database

Organize the project into clean modules rather than one giant script.

========================================
1. MARKET DATA PIPELINE
========================================

Build a data pipeline capable of downloading and caching historical stock and ETF data.

Support:
- OHLCV
- adjusted prices
- S&P 500 benchmark
- multiple tickers
- configurable date ranges
- daily data initially

Use yfinance initially, but design the data layer so another provider can be substituted later.

Cache downloaded data locally so it does not have to be downloaded every run.

Handle:
- missing data
- stock splits
- dividends
- invalid tickers
- API failures

========================================
2. FEATURE ENGINEERING
========================================

Generate quantitative features using ONLY information that would have been available at that point in time.

Include features such as:

Returns:
- 1-day return
- 5-day return
- 10-day return
- 20-day return
- 60-day return
- log returns

Momentum:
- RSI
- MACD
- rate of change
- price momentum
- distance from moving averages

Trend:
- SMA 10
- SMA 20
- SMA 50
- SMA 100
- SMA 200
- EMA features
- moving-average crossovers

Volatility:
- rolling volatility
- ATR
- rolling standard deviation
- downside volatility

Volume:
- volume change
- relative volume
- rolling average volume
- volume-price relationships

Market-relative features:
- stock return minus S&P 500 return
- rolling beta
- rolling correlation with market
- relative strength versus S&P 500

Statistical features:
- rolling mean
- rolling median
- rolling skew
- rolling kurtosis
- rolling z-score
- rolling min/max
- distance from rolling high/low

Calendar features:
- day of week
- month
- quarter

Make the feature system modular so new signals can easily be added.

========================================
3. TARGET VARIABLES
========================================

Do not only predict raw prices.

Support several configurable ML targets.

Regression examples:
- next-day return
- next-5-day return
- next-20-day return

Classification examples:
- probability next-day return is positive
- probability next-5-day return is positive
- probability next-20-day return exceeds a configurable threshold

Also support ranking stocks by expected future return.

Make prediction horizon configurable.

========================================
4. MACHINE LEARNING
========================================

Implement and compare several models:

Baseline:
- linear regression
- logistic regression

Tree models:
- random forest
- gradient boosting
- XGBoost or LightGBM if available

Allow additional models to be added later.

Use proper preprocessing pipelines.

Do NOT randomly shuffle financial time-series data.

Use:
- chronological train/test splits
- expanding-window validation
- walk-forward validation

Prevent:
- look-ahead bias
- target leakage
- survivorship assumptions where practical
- preprocessing leakage

Scaling and feature selection must be fitted ONLY on training data.

========================================
5. WALK-FORWARD BACKTESTING
========================================

Build a realistic backtesting engine.

At each historical point:

1. Train using only information available before that date.
2. Generate predictions.
3. Convert predictions into trading signals.
4. Execute simulated trades.
5. Record portfolio performance.
6. Move forward through time.

Support configurable:
- starting capital
- position sizing
- maximum number of positions
- transaction costs
- slippage
- long-only strategies initially
- rebalance frequency
- prediction threshold

Never allow the model to see future information.

========================================
6. STRATEGY LOGIC
========================================

Allow strategies such as:

LONG if:
predicted probability of positive return > threshold

or:

LONG the top N stocks ranked by predicted future return.

Allow thresholds and position sizing to be configured.

Keep the ML prediction system separate from portfolio construction.

========================================
7. PERFORMANCE ANALYTICS
========================================

Calculate:

- total return
- annualized return / CAGR
- volatility
- Sharpe ratio
- Sortino ratio
- maximum drawdown
- Calmar ratio
- win rate
- profit factor
- average trade return
- number of trades
- exposure
- turnover

Compare performance against:

- buy-and-hold of the same stock
- SPY / S&P 500

Plot:

- portfolio equity curve
- benchmark equity curve
- drawdown
- rolling Sharpe ratio
- rolling volatility
- prediction performance

========================================
8. MODEL EVALUATION
========================================

For regression calculate:

- MAE
- RMSE
- R²
- directional accuracy
- correlation between predictions and future returns

For classification calculate:

- accuracy
- precision
- recall
- F1
- ROC-AUC
- confusion matrix
- calibration

Do not claim that a model is useful simply because prediction accuracy is above 50%.

Evaluate whether predictions produce economically meaningful results after transaction costs.

========================================
9. FEATURE IMPORTANCE
========================================

Provide model interpretability.

Include:

- feature importance
- permutation importance

If practical, optionally support SHAP.

Show which signals contribute most strongly to predictions.

========================================
10. OVERFITTING PROTECTION
========================================

Treat overfitting as a major concern.

Implement:

- walk-forward validation
- out-of-sample testing
- regularization
- configurable training windows
- feature reduction
- comparison against simple baselines

Clearly separate:

TRAIN
VALIDATION
TEST

The final test period must remain untouched until model/strategy selection is complete.

========================================
11. QUANT RESEARCH REPORT
========================================

After a run, generate a report containing:

MODEL
Best-performing model and hyperparameters.

SIGNALS
Most important predictive features.

PREDICTIONS
Current model predictions for selected stocks.

BACKTEST
Historical strategy performance.

BENCHMARK
Performance versus SPY and buy-and-hold.

RISK
Volatility, drawdown, Sharpe, Sortino, and exposure.

ROBUSTNESS
Performance across different historical periods and market regimes.

LIMITATIONS
Clearly identify potential overfitting, data-quality problems, small sample sizes, and other limitations.

========================================
12. CURRENT STOCK ANALYSIS
========================================

Create a command such as:

python3 analyze.py AAPL

It should:

1. Download/update AAPL data.
2. Calculate features.
3. Load the trained model.
4. Generate predictions.
5. Display something like:

Ticker: AAPL
Current price: $XXX.XX

5-day expected return: +X.XX%
Probability of positive 5-day return: XX.X%

Model signal:
BULLISH / NEUTRAL / BEARISH

Model confidence: XX%

Important current signals:
- RSI
- momentum
- volatility
- distance from SMA50
- relative strength vs SPY

Do not present the prediction as certainty.

========================================
13. STOCK SCREENER
========================================

Create:

python3 screen.py

Analyze a configurable universe of stocks and rank them using the trained model.

Example output:

Rank | Ticker | Expected Return | Probability Up | Score
1    | XXXX   | +X.XX%          | XX%            | X.XX
2    | XXXX   | +X.XX%          | XX%            | X.XX

Allow:

python3 screen.py --top 20

========================================
14. PROJECT STRUCTURE
========================================

Use a structure similar to:

quant_ml/
    data/
        downloader.py
        cache.py

    features/
        technical.py
        statistical.py
        market.py

    models/
        train.py
        predict.py
        validation.py

    backtest/
        engine.py
        metrics.py
        portfolio.py

    research/
        reports.py
        plots.py

    config.py

data/
models/
results/

train.py
backtest.py
analyze.py
screen.py
requirements.txt
README.md

========================================
15. CONFIGURATION
========================================

Make important parameters configurable rather than hardcoded:

- ticker universe
- benchmark
- start/end date
- prediction horizon
- training window
- validation window
- transaction costs
- slippage
- starting capital
- model
- feature set
- probability threshold
- number of positions

========================================
16. TESTING
========================================

Write tests for critical components, especially:

- feature calculations
- target alignment
- chronological splitting
- prevention of future-data leakage
- portfolio accounting
- performance metrics

Run the tests yourself and fix failures.

========================================
17. README
========================================

Write a detailed README explaining:

- architecture
- installation
- training
- backtesting
- analyzing individual stocks
- screening stocks
- interpreting results
- limitations of ML predictions

========================================
IMPORTANT RULES
========================================

Do not create fake market data when real historical data can be downloaded.

Do not create fake backtest results.

Do not hardcode successful-looking results.

Do not use future information in historical predictions.

Do not randomly split time-series observations.

Do not optimize hyperparameters against the final test set.

Do not claim profitability unless demonstrated by a genuinely out-of-sample backtest after estimated transaction costs.

Build this incrementally and actually execute/test each stage.

Start with:
1. project structure
2. data pipeline
3. feature engineering
4. targets
5. baseline model
6. walk-forward validation
7. backtester

Only after those components work should you add more advanced models and the stock screener.