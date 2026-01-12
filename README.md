# M1 Randomness Trader

*A realistic assessment of short-term predictability in forex markets*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Telegram](https://img.shields.io/badge/Telegram-@realistic__algotrading-blue.svg)](https://t.me/realistic_algotrading)

## 📊 Honest Findings

This project demonstrates a crucial reality in algorithmic trading:

**EURUSD M1 data shows no statistically significant predictable patterns.**

### Key Results:
- Real 10-minute volatility: **2.6 pips**
- Model uncertainty: **2.8-3.3 pips** (well-calibrated)
- **Zero valid signals** in walk-forward testing
- Model correctly identifies **no tradable edge**

### Why This Matters:
1. **Realistic uncertainty estimation** prevents false discoveries
2. **Walk-forward validation** avoids look-ahead bias
3. **Pip-based targets** account for transaction costs
4. **Honest reporting** of "no signals" is a success, not failure

## 🏗️ Architecture

### Data Pipeline
- **Source**: MetaTrader 5 EURUSD M1 data
- **Features**: Normalized price windows + regression statistics
- **Targets**: Future price changes in pips (10/15 minutes ahead)

### Model Design
- **CNN** for pattern extraction from price/volume sequences
- **Regression features**: linear slope and R²
- **Gaussian output**: predicts mean and variance for uncertainty
- **Loss**: Negative log-likelihood for probabilistic forecasts

### Validation Strategy
1. **Walk-forward**: Train on past, test on recent 7 days
2. **Hold-out**: Final validation on last 10% of data
3. **Trading rules**: Only trade with sufficient edge (|mean|/σ > 1.5)

## 🚀 Quick Start

### Prerequisites
- Windows OS (for MT5)
- MetaTrader 5 installed
- Python 3.9+

### Installation
```bash
git clone https://github.com/vger-cell/realistic-algotrading.git
cd realistic-algotrading
pip install -r requirements.txt
