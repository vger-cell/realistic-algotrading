# Multi-Strategy Algorithmic Trading Backtest

## Overview
Unified backtesting framework that runs three distinct algorithmic trading strategies and combines them into a portfolio. Educational demonstration of multi-strategy portfolio construction with realistic implementation details.

## Strategies

### Strategy 1: Per-Symbol Optimizer
- **Markets**: EURUSD, AUDUSD
- **Timeframe**: H1
- **Logic**: Grid search optimization of parameters (horizon, TP/SL, probability thresholds) per symbol using logistic regression on technical features
- **Features**: Range, gaps, run length, permutation entropy, z-scores, etc.

### Strategy 2: Aggressive EURUSD
- **Markets**: EURUSD (primary), GBPUSD (optional)
- **Timeframe**: H1
- **Logic**: BOCPD + regime switching + EVT + Thompson Sampling bandit
- **Complexity**: Dynamic parameters based on market regime (calm/trend/turbulent) and recent performance patterns

### Strategy 3: TS2Vec + Classifier
- **Markets**: EURUSD
- **Timeframe**: M15
- **Logic**: TS2Vec-style embeddings + calibrated classifier (XGBoost/RandomForest)
- **Features**: Relative price features, embeddings from contrastive learning

## Real Results Summary (from actual output)
| Strategy | Trades | WinRate | Profit Factor | Net PnL (USD) | Max DD (USD) | Sharpe |
|----------|--------|---------|---------------|---------------|--------------|--------|
| STRAT1   | 1      | 100.00% | ∞             | +20.35        | 0.00         | 20.35  |
| STRAT2   | 36     | 41.67%  | 0.646         | -336.42       | 441.99       | -0.187 |
| STRAT3   | 66     | 46.97%  | 0.986         | -11.10        | 202.80       | -0.006 |
| **PORTFOLIO** | **103** | **45.63%** | **0.815** | **-327.17** | **457.67** | **-0.083** |

## Period
Approximately May 2024 - December 2025 (varies by symbol and data availability)

## Key Implementation Details

### Time Handling
- Proper timezone-naive normalization across all strategies
- Anti-leakage measures with horizon buffers
- Time series splits for train/validation/test

### Risk Management
- Fixed lot size: 0.1
- Spread and commission included in calculations
- Peak margin calculation for portfolio
- Daily stop losses and trade limits

### Educational Value
1. **Realistic backtest structure** with proper train/val/test splits
2. **Multiple strategy types** from simple to complex
3. **Unified portfolio metrics** showing real aggregation challenges
4. **Transparent parameter optimization** with grid search
5. **Advanced techniques** like BOCPD, EVT, TS2Vec, bandit algorithms

## Important Disclaimer
⚠️ **FOR EDUCATIONAL PURPOSES ONLY**

- This is a BACKTEST, not live trading results
- Only Strategy 1 showed profit in test period (1 trade)
- Strategies 2 and 3 were net losers
- The portfolio lost money overall
- **DO NOT USE FOR LIVE TRADING**
- Demonstrates challenges of combining multiple strategies

## Output Files
- `outputs_unified/all_trades.csv`: All trades from all strategies
- `outputs_unified/equity_all.png`: Portfolio and individual strategy equity curves

## Installation
```bash
pip install -r requirements.txt