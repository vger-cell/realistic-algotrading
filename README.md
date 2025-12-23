# ARPT: Adaptive Realistic Profitability Trader

> **Honest backtest of a level-based ML strategy on EURUSD H4. Spoiler: It loses money.**

## Objective
Test whether engineered features from historical support/resistance levels can predict significant price moves (≥80 pips in 12 hours) on EURUSD H4, using a LightGBM model with weekly online learning on actual trade results.

## Method
- **Data**: 49,173 H4 bars of EURUSD (~22 years)
- **Features**: 14 z-score normalized metrics from 10 nearest price levels
- **Target**: Binary (1 if |move| ≥80 pips in 3 bars, else 0) → only 13.5% positive samples
- **Validation**: Walk-forward with 60-day train / 20-day test folds
- **Adaptation**: Model retrained weekly on last 50 trade PnL outcomes (profit=1, loss=0)
- **Execution**: Fixed 0.1 lot, TP/SL = 80/40 pips (min), no compounding

## Results
- **Total Trades**: 74  
- **Win Rate**: **35.1%** (**below random 50%**)  
- **Net PnL**: **-$213.94** (2.14% loss on $10k account)  
- **Max Drawdown**: 1.32%  
- **Avg PnL per Trade**: -$2.89  

> 🔴 **Critical Insight**: Optimization failed to reach even random baseline. Level-based features—despite engineering—carry no predictive edge. Market regime shifts (e.g., 2008 crisis) invalidate static levels.

## Why This Matters
- **Classical indicators (RSI, MACD) or level-based features used as inputs to ML models do not create "alpha"**—they merely replicate the limitations of the original indicators.
- **Performance is regime-dependent**: a strategy working in ranging markets fails catastrophically in trending or volatile ones.
- **Online learning on PnL does not fix a broken signal**—it only overfits to recent noise.

## Reproduce
```bash
pip install -r requirements.txt
python arpt_levelbased_h4.py
