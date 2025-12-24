# Liquidity Zone Trading Strategy Backtester

An optimized algorithmic trading strategy that identifies and trades toward liquidity zones in EURUSD.

## 📊 Strategy Overview

This strategy identifies potential liquidity zones (round-number levels and recent extremes) and enters trades toward these zones with optimized risk management.

### Key Features:
- **Zone Detection**: Identifies levels 20-35 pips from current price
- **Risk Management**: Dynamic stop-loss (70% of distance), minimum profit filters
- **Market Filters**: RSI-based overbought/oversold avoidance
- **Forward Testing**: Tests on unseen market data

## 📈 Backtest Results (EURUSD M15, 90 days)

| Metric | Result | Status |
|--------|--------|--------|
| Total Trades | 280 | ⚠️ |
| Win Rate | 46.4% | ✅ Improved |
| Total P&L | -2.9 pips | ❌ Break-even |
| Avg Trade | -0.01 pips | ❌ No edge |
| Best Zones | 30-35 pips (52% WR) | ✅ Promising |

### Key Insights:
1. **Win Rate improved** from 35.3% to 46.4%
2. **No profitability** despite improved win rate
3. **30-35 pip zones** show best performance (52% WR)
4. **Calculation bug** in P&L analysis identified

## 🛠 Installation

1. Install requirements:
```bash
pip install -r requirements.txt
