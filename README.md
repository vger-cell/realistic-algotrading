# TS2Vec + Koopman — Fixed TP/SL Backtest

> **Advanced architecture, disappointing results: a lesson in signal vs noise**

## Objective
Predict directional EURUSD M15 moves ≥3 pips over 8 hours (32 bars) using:  
- **TS2Vec**: Self-supervised time-series encoder  
- **Koopman/PIKA**: Nonlinear dynamical system coordinates  
- **Calibrated Random Forest**: Final trade/no-trade classifier

## Key Design Choices
- ✅ **No data leakage**: Koopman trained only on train set  
- ✅ **Fixed TP/SL**: 40-pip take-profit, 30-pip stop-loss (retail realistic)  
- ✅ **Strict walk-forward**: 27k train / 9k test bars (2024–2025)  
- ✅ **Cost-aware**: 2-pip spread per trade

## Critical Results
| Metric               | Value         |
|----------------------|---------------|
| **Total Trades**     | 97            |
| **Win Rate**         | 41.24%        |
| **Profit Factor**    | 0.284         |
| **Net PnL**          | **–$202.00**  |
| **Max Drawdown**     | 2.02%         |
| **Model Accuracy**   | 99.2%*        |

> \* *Misleading due to 99.16% "no move" samples — model never learns real signals*

### Why It Failed
1. **Extreme class imbalance**: Only 0.84% of samples had meaningful moves  
2. **Signal collapse**: Model defaulted to "sell" for all bars after calibration  
3. **Fixed TP/SL mismatch**: 40/30 ratio too aggressive for 8-hour horizon  
4. **No regime awareness**: Strategy ignores volatility/trend shifts

## Lesson
**Architectural complexity ≠ profitability**. Even state-of-the-art time-series representations (TS2Vec + Koopman) cannot compensate for:
- Weak underlying signal-to-noise ratio in EURUSD M15  
- Inadequate target engineering (binary move/no-move)  
- Lack of market context (e.g., news, sessions)

> ⚠️ **Not financial advice**. For research only.

---

**Author**: Vladimir Korneev  
**Telegram**: t.me/realistic_algotrading  
**Repo**: github.com/vger-cell/realistic-algotrading
