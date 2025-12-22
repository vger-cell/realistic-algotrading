# Event-Driven Lead-Lag Analyzer

## 📊 Overview
A Python-based algorithmic trading research tool that identifies statistically significant lead-lag relationships between major FX pairs and gold. Uses event-driven signals (breakouts and mean reversions) with RSI filtering to detect predictive patterns across assets.

## 🔑 Key Features
- **Multi-Asset Analysis**: 8 major FX pairs + gold (XAUUSD)
- **Dual Timeframes**: M5 and M15 for different trading styles
- **Four Signal Types**: Breakout long/short, bounce long/short
- **Statistical Rigor**: Binomial test ensures p < 0.05 significance
- **Realistic Simulation**: 10-pip TP / 20-pip SL with proper pip calculations
- **Visualization**: Heatmaps, distributions, and detailed reports

## 📈 Real Results (180-day backtest)
- **9 significant pairs** identified
- **Best Pair**: EURUSD → XAUUSD (M15): 68.8% hit rate, PF 2.00
- **M15 outperforms M5**: Higher profit factors (1.20-2.00 vs 0.82-1.10)
- **Gold as common follower**: 5/9 pairs lead to XAUUSD movements

## 🚀 Quick Start
1. Install MetaTrader 5 and enable Python integration
2. Install requirements: `pip install -r requirements.txt`
3. Run analysis: `python lead_lag_analyzer.py`
4. Check `event_lead_lag/` folder for results and visualizations

## ⚠️ Important Notes
- **Educational Purpose**: This is a research tool, not production trading software
- **Transaction Costs**: Not included - real trading would reduce profitability
- **Market Conditions**: Results based on last 180 days; relationships may change
- **Risk Management**: Always use proper risk controls in live trading

## 📁 Output Files
- `lead_lag_results_detailed.csv`: All significant pairs with metrics
- `analysis_report.txt`: Summary report
- `plots/`: Heatmaps and distribution charts

## 👨💻 Author
**Vladimir Korneev**  
Telegram: [t.me/realistic_algotrading](https://t.me/realistic_algotrading)  
GitHub: [github.com/vger-cell/realistic-algotrading](https://github.com/vger-cell/realistic-algotrading)

## 📜 License
MIT License - Free for educational and research use