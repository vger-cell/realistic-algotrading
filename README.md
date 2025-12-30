# Realistic Algorithmic Trading System

A comprehensive trading system that evaluates 5 common technical indicators on EURUSD data with realistic trading conditions.

## System Concept

This system implements a scientific approach to strategy evaluation by comparing two distinct trading methodologies:

1. **Direct Trading Approach**: Uses signals from the top-performing technical indicators based on historical performance
2. **Contrarian Trading Approach**: Inverts signals from the worst-performing indicators, testing if "doing the opposite" of losing strategies can be profitable

The system incorporates realistic trading parameters including slippage, commission, and risk management to provide a more accurate simulation of live trading conditions than typical academic backtests.

## Features

- **5 Technical Indicators**: RSI, Stochastic, MACD, Moving Averages, Bollinger Bands
- **Realistic Backtesting**: Includes slippage, commission, and realistic execution
- **Risk Management**: Fixed 10-pip TP, 20-pip SL, max 24-hour trade duration
- **Strategy Comparison**: Direct trading vs. contrarian (inverted) approach
- **Comprehensive Metrics**: Win rate, profit factor, drawdown, trade duration

## Installation

1. Install MetaTrader 5 and set up a demo account
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
