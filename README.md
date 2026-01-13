# MTF Feature Effectiveness Analysis for EURUSD

**Author**: [Vladimir Korneev](https://t.me/realistic_algotrading)  
**Repository**: [github.com/vger-cell/realistic-algotrading](https://github.com/vger-cell/realistic-algotrading)

## What This Project Is About

Just finished a deep analysis of 15 technical features for predicting EURUSD price movement on the M15 timeframe. The results completely overturn conventional wisdom in retail trading!

## Key Findings

✅ **The strongest feature is `H1_position`** — the relative position of the current price within the last 50-bar H1 range. It improves R² by **+0.0517**, which is a huge leap for short-term forecasting!  
✅ All top 7 features are **multi-timeframe (MTF)**: positions on H1/H4, trends, and distances to moving averages.  
❌ **All classic indicators failed**:  
   • Price Z-Score → R² drop of -0.0016  
   • Volatility (20 bars) → -0.0015  
   • EMA(12)/EMA(26) ratio → -0.0013  

## Why This Matters

This proves that **higher timeframe context** (where price sits inside the H1/H4 range) is far more valuable than oscillators or volatility measures. You’re essentially “seeing” market structure through positional metrics — just like professional traders do.

## What’s Next?

We’re building a live trading strategy based on these features with:  
• Hourly retraining (model updates every hour)  
• Walk-forward validation (no data leakage)  
• Dynamic Stop Loss / Take Profit levels  

## How to Use This Code

1. Install the [MetaTrader 5 terminal](https://www.metatrader5.com/)  
2. Run the script while MT5 is connected  
3. Check the generated `feature_analysis_EURUSD_*.json` for full rankings  

Recommended feature set for your own models:
```python
['log_return', 'high_low_range', 'H1_position', 'H4_dist_ma', 
 'H4_position', 'H4_trend', 'trend_pos_interaction']
```

## Important Disclaimer

This research is for **educational purposes only**. Historical performance does **not** guarantee future results. Always test strategies on a demo account before going live.

> Follow updates: [t.me/realistic_algotrading](https://t.me/realistic_algotrading)  
> Full open-source code: [github.com/vger-cell/realistic-algotrading](https://github.com/vger-cell/realistic-algotrading)
