# EURUSD Advanced Hypotheses Testing System

This project is a scientific experiment to test the limits of predictability in the EUR/USD forex market. It is designed for educational purposes to demonstrate the principles of rigorous backtesting and the reality of market efficiency.

## Objective
To determine if a machine learning model, fed with a rich set of features derived from 7 major currency pairs, can generate statistically significant and economically viable predictions for EUR/USD hourly returns over the 2020-2023 period.

## Key Components
- **Data**: Real or synthetic H1 data for EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD.
- **Features**: 33+ engineered features including cross-asset correlations, volatility ratios, and lagged returns.
- **Models**: Ridge Regression, Gradient Boosting, and a simple ensemble.
- **Validation**: Strict walk-forward analysis with 5 non-overlapping test windows.

## Expected Outcome
Based on the Efficient Market Hypothesis, the expected outcome is that all models will perform at or near the level of random chance. This serves as a critical reminder of the challenges in algorithmic trading.

## Author
Vladimir Korneev

## Disclaimer
This software is for educational and research purposes only. It is not financial advice. Past performance is not indicative of future results.
