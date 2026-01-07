# Noise-Based Scalping Strategy (EUR/USD M1)

**Author**: [Vladimir Korneev](https://t.me/realistic_algotrading)  
**Repository**: [github.com/vger-cell/realistic-algotrading](https://github.com/vger-cell/realistic-algotrading)

## Objective
Trade residual "noise" (price deviations from a linear trend) on EUR/USD M1 data, with realistic spread and volume filtering.

## Key Features
- **Noise Calculation**: `Close - Linear Regression Trend` (window: 30-100 bars)
- **Dynamic Threshold**: 90-97% quantile of absolute noise
- **Volume Filter**: Only trades when volume > 1000-bar median
- **Realistic Execution**: Accounts for spread (from MT5 CSV)
- **Validation**: 75% Walk-Forward optimization, 25% Out-of-Sample test

## Results (2025-09-29 to 2026-01-07)
| Period | Trades | PnL (USD) | Win Rate | Max Drawdown |
|--------|--------|-----------|----------|--------------|
| WF     | 4,268  | +3,666    | 51.77%   | $521         |
| **OOS**| **1,367**| **-65**   | **47.22%**| **$556**     |
| Buy & Hold (OOS) | - | **+21** | - | - |

## Critical Insight
⚠️ **Overfitting Alert**: The strategy shows exceptional WF performance but **fails in OOS** (loses money while Buy & Hold profits). This confirms that M1 "noise" lacks persistent predictive structure. The aggressive parameters (`quantile=0.90`) optimized for WF do not generalize.

## Usage
1. Place `EURUSDM1.csv` (UTF-16 LE BOM format from MT5) in the project folder
2. Run `python noise_strategy_optimized.py`
3. **Warning**: Not suitable for live trading. For educational purposes only.

> "If it works only in backtests, it’s not a strategy—it’s a mirage." — Vladimir Korneev
