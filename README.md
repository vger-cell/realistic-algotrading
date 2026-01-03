# Optimized Cryptocurrency Momentum Scanner

## Overview
Real-time momentum scanner for cryptocurrency markets detecting trend-following (TREND_LONG) and mean-reversion (24H_SHORT) opportunities. Designed for educational purposes to demonstrate algorithmic trading principles.

## Features
- **Dual Strategy**: Combines trend-following and mean-reversion approaches
- **BTC Correlation Filter**: Avoids trading against major market trends
- **Risk Management**: Strict 2:1 risk-reward ratio (TP: 1.8%, SL: 0.9%)
- **Real-time Display**: Color-coded terminal output with session statistics
- **CSV Logging**: Complete trade history for analysis

## Strategy Logic
### TREND_LONG (Trend Following)
- Entry: Price change ≥0.3% + momentum ≥0.2%
- Exit: TP 1.8% / SL 0.9% / Timeout 40min
- BTC Filter: Don't enter if BTC falls >0.3% in 5min

### 24H_SHORT (Mean Reversion)
- Entry: 24h drop ≥3.0% + negative momentum ≤-0.2%
- Exit: TP 1.8% / SL 0.9% / Timeout 40min
- BTC Filter: Don't enter if BTC rises >0.3% in 5min

## Installation
```bash
git clone https://github.com/vger-cell/realistic-algotrading.git
cd realistic-algotrading
pip install -r requirements.txt
python momentum_scanner.py
