EURUSD → XAUUSD Lead-Lag Strategy
Overview
A lead-lag correlation trading strategy that uses EURUSD to generate signals for trading XAUUSD (Gold) on the M15 timeframe. The strategy identifies bounce signals from support/resistance levels with RSI confirmation.

Strategy Logic
Lead Instrument: EURUSD (signal generator)

Follow Instrument: XAUUSD (trade execution)

Signal Types: Bounce from support (long) or resistance (short)

Entry: 15-minute delay after signal

Exit: Fixed TP=180 pips, SL=120 pips (1.5:1 ratio)

Position Sizing: 1% risk per trade

Backtest Results (Best Configuration)
Total Return: 5.16% (annualized ~20.64%)

Win Rate: 52.9%

Profit Factor: 1.61

Max Drawdown: 3.03%

Sharpe Ratio: 3.64

Total Trades: 17 (5.6 trades/month)

Avg Trade Profit: $30.36

Key Findings
Only bounce signals worked - No breakout signals were generated

Fixed stops outperformed ATR-based (3.16% vs -0.28%)

Trend filter too restrictive - Generated 0 signals when enabled

Aggressive parameters (lower RSI thresholds) produced best results

Installation
bash
pip install -r requirements.txt
Usage
Install MetaTrader 5

Configure your MT5 account credentials

Run the backtest:

bash
python eurusd_xauusd_strategy.py
Project Structure
eurusd_xauusd_strategy.py - Main strategy code

requirements.txt - Python dependencies

backtest_results/ - Output directory for results

comparison_results/ - Configuration comparison data

Configuration
The strategy includes 5 test configurations:

Fixed TP/SL (baseline)

ATR-based stops

With trend filter (ADX)

Full system (all filters)

Aggressive (more signals) - Best performing

Disclaimer
This is educational software for backtesting purposes only. Past performance does not guarantee future results. Trading involves risk of loss.

Author
Vladimir Korneev
Telegram: @realistic_algotrading
Repository: github.com/vger-cell/realistic-algotrading
