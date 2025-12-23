TS2Vec + Random Forest Trading Strategy
📈 Project Focus & Current Status
This project represents an ongoing development effort to create a robust, machine learning-based trading system for the EURUSD currency pair. The repository has been consolidated to focus on a single, most promising strategy (Strategy 3: TS2Vec + Random Forest Classifier) following extensive backtesting and comparative analysis of multiple approaches.

The core of this strategy is a hybrid model combining:

TS2Vec (LSTM-based encoder): Transforms raw 15-minute price sequences into compact, meaningful 32-dimensional embeddings.

Random Forest Classifier: Predicts price direction 32 bars ahead using the learned embeddings, achieving a consistent ~72.5% accuracy on out-of-sample data.

🛡️ Key Development Principles & Safeguards
A primary focus of this development cycle has been implementing rigorous safeguards against data leakage to ensure model validity and realistic performance estimates:

Temporal Data Splitting: Train (75%) and test (25%) sets are strictly separated by time to prevent future information from contaminating the training process.

Statistic Isolation: Feature normalization (scaling) is performed using only training set statistics (mean, std). The test set is transformed using these pre-computed values.

Sequential Processing: Feature engineering (e.g., indicator calculation) is applied separately to train and test sets to prevent look-ahead bias.

Validation for Calibration: The calibrated classifier is fit on a hold-out validation split from the training data, never on the test set.

These checks are verified in the logs (e.g., [LEAK-CHECK] Train end: 2025-08-13 14:00:00 < [LEAK-CHECK] Test start: 2025-08-13 14:15:00).

🔍 Latest Backtest Results & Analysis
A recent backtest over a 4+ month period (Aug-Dec 2025) yielded the following metrics, highlighting both the model's predictive power and areas for tactical improvement:

Model Accuracy / F1-Score: 72.6% / 0.722 – The core predictive model shows strong and consistent signal.

Strategy Performance: Win Rate: 45% | Profit Factor: 0.90 | Net PnL: -$5.50

Risk Management: Avg Win: $5.50 | Avg Loss: -$5.00 | Risk/Reward: 1.10

🧐 Interpreting the Results
The results reveal a clear disconnect: a high-accuracy predictive model is currently paired with a sub-optimal trading strategy. The positive Risk/Reward (1.10) is a good foundation, but the sub-1.0 Profit Factor and low Win Rate indicate the entry/exit logic needs refinement.

Root Cause Identified: The aggressive fixes applied to improve the previous losing strategy (like raising probability thresholds to 0.6 and adding a trend filter) were too restrictive. They reduced false signals but also filtered out 98.3% of all potential trading opportunities, leaving too few trades (only 20) for the strategy's edge to materialize statistically.

🚀 Next Steps: Strategic Optimization Roadmap
The immediate development priority is systematic parameter optimization to bridge the gap between model accuracy and trading profitability. The focus will be on finding the optimal balance between signal frequency and quality.

The optimization pipeline will target:

Signal Generation Parameters:

Probability thresholds (PROB_THRESHOLD_BUY/SELL)

Minimum probability difference (MIN_PROB_DIFFERENCE)

Trend filter sensitivity (MIN_TREND_STRENGTH)

Trade Management Parameters:

Take-Profit / Stop-Loss levels (TP_PIPS, SL_PIPS)

Position sizing logic

Model Hyperparameters (secondary):

Random Forest depth, number of estimators.

TS2Vec embedding dimension, learning rate.

Optimization will employ walk-forward analysis or cross-validation on sequential data to maintain temporal integrity and prevent overfitting.
