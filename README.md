# Realistic AlgoTrading: Walk-Forward ML Strategy for EURUSD

This project demonstrates a disciplined approach to developing and backtesting a machine learning trading strategy. It uses a walk-forward analysis to provide a realistic assessment of performance, avoiding the common pitfall of overfitting.

## Strategy Overview

The core idea is to use a simple `Ridge` regression model to predict the short-term direction of the EURUSD price. The model is trained on a rich set of 18 technical features engineered from M15, H1, and H4 timeframes. To ensure robustness, the strategy incorporates several layers of risk management and adaptive logic:

*   **Walk-Forward Analysis:** The model is retrained weekly on a rolling 180-day window and tested on the following 7 days.
*   **Dynamic Risk Management:** Stop Loss (SL) and Take Profit (TP) levels are optimized on each training window.
*   **Adaptive Trade Filtering:** 
    *   Uses asymmetric ADX thresholds to confirm trend strength for entries.
    *   Requires price to be above/below an H4 moving average for BUY/SELL signals.
    *   **Critically, it automatically disables all SELL trades if their recent Profit Factor drops below 0.5.**

## Key Results (Jan 2024 - Jan 2026)

The backtest yielded the following metrics:
*   **Total PnL:** +$973.50
*   **Total Trades:** 197
*   **Average Win Rate:** 52.1%
*   **Profit Factor:** 1.43
*   **Sharpe Ratio:** 2.39

### Critical Insight & Limitations

The most important finding is that the **SELL signals were consistently unprofitable**. The system's adaptive logic correctly identified this and disabled short trades for the vast majority of the test period. This means the strategy's profitability is **entirely dependent on a long-bias market regime** (like the one seen in EURUSD during 2024-2026).

**This is not a "holy grail" but a realistic example of how a model can adapt to market conditions.** Its future performance is highly uncertain if the market enters a strong bearish phase. This project serves as an educational tool on the importance of:
1.  Rigorous, leak-free backtesting.
2.  Adaptive risk management.
3.  Understanding the directional bias of your strategy.

## Getting Started

1.  Clone the repository.
2.  Install the required packages: `pip install -r requirements.txt`.
3.  Ensure you have MetaTrader 5 running and the `MetaTrader5` Python package configured.
4.  Run `walkforward_analysis.py`.

**Disclaimer:** This is for educational and research purposes only. Past performance is not indicative of future results. Trading involves significant risk of loss.

---
**Author:** Vladimir Korneev  
**Telegram:** [t.me/realistic_algotrading](https://t.me/realistic_algotrading)  
