# Vision-Based Multi-Horizon Trading Strategy

A realistic, walk-forward-optimized algorithmic strategy using only closing prices and visual pattern features.

## Features

- **Input**: Only OHLC (uses Close); no trigonometric raw features (per user spec)
- **Features**: ATR-normalized closing prices, linear regression fit, residuals, and angle-based sin/cos
- **Model**: Shared 1D-CNN predicting 3/5/10-step directional classes (Buy/Sell/Neutral)
- **Optimization**: Optuna with robust constraints (TP/SL ≥ 1.8, high-confidence filtering)
- **Validation**: Strict walk-forward with temporal split (no lookahead)
- **Risk Control**: Fixed TP/SL, max 10-bar trade duration, 0.1 lot size
- **Metrics**: Profit factor, win rate, max drawdown, equity curve

## Results (EURUSD H1, ~2023–2025)

- **Total Trades**: 60  
- **Win Rate**: 51.67%  
- **Net PnL**: +$218.80 (from $1000)  
- **Max Drawdown**: 40.65%  
- **Profit Factor**: 1.72  

> ⚠️ **Not financial advice.** High drawdown indicates aggressive risk profile. Performance not guaranteed in live markets.

## Usage

1. Install requirements: `pip install -r requirements.txt`
2. Ensure MetaTrader 5 terminal is running
3. Run `python strategy.py`

## Author

Vladimir Korneev  
Telegram: [t.me/realistic_algotrading](https://t.me/realistic_algotrading)  
Repo: [github.com/vger-cell/realistic-algotrading](https://github.com/vger-cell/realistic-algotrading)
