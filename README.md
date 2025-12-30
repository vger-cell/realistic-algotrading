# Multi-Strategy Forex Trading System with Cross-Validation

## 📊 Project Overview
This is an educational algorithmic trading system that demonstrates **proper backtesting methodology** for EURUSD trading. The system combines three technical strategies with rigorous validation techniques including time-series cross-validation and walk-forward testing.

**Important Disclaimer**: This system passed technical validation but showed **negative profitability** (-2.13%) in final out-of-sample testing. It serves as an educational example of proper methodology rather than a profitable trading system.

## 🎯 Key Features
1. **Three Trading Strategies**:
   - STOCH_buy: Stochastic oscillator in oversold zone
   - RSI_sell: RSI entering overbought territory
   - MACD_sell: MACD bearish crossover

2. **Robust Validation**:
   - Time-series cross-validation (3 folds)
   - Walk-forward validation (4 windows)
   - Final out-of-sample test on unseen data
   - Realistic trading costs (commission, slippage)

3. **Risk Management**:
   - Maximum drawdown limit (20%)
   - Position sizing control
   - Stop-loss and take-profit optimization

## 📈 Performance Results
### Final Out-of-Sample Test (Nov 24 - Dec 30, 2025)
- **Return**: -2.13% ($-213.10)
- **Win Rate**: 45.8%
- **Maximum Drawdown**: -10.6%
- **Total Trades**: 59
- **Average Profit/Trade**: $-3.61

### Validation Results
- **Walk-forward**: 2/3 profitable windows (66.7%)
- **Criteria Passed**: 3/4 (excluding profitability)
- **Methodology**: ✅ Correctly implemented

## 🛠️ Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
