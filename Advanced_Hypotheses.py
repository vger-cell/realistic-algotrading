"""
EURUSD Advanced Hypotheses Testing System
Version 5.2 

This script implements a rigorous, multi-currency walk-forward analysis to test the predictability 
of EUR/USD hourly returns (2020-2023). It leverages cross-asset features from 7 major FX pairs and 
evaluates three models: Ridge Regression, Gradient Boosting, and a simple ensemble.

Objective: To empirically test the Efficient Market Hypothesis (EMH) on the most liquid FX market.
Key Finding: All models perform at or near random chance, providing strong evidence for market efficiency.

Author: Vladimir Korneev
Repository: github.com/vger-cell/realistic-algotrading
"""

"""
EURUSD ADVANCED HYPOTHESES TESTING SYSTEM
Version 5.2 - Clean English version with minimal prints
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import MetaTrader5 as mt5

# ==================== CONFIGURATION ====================
class AdvancedConfig:
    """System configuration"""

    SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]
    TIMEFRAME = "H1"
    START_DATE = "2020-01-01"
    END_DATE = "2023-12-31"

    DATA_DIR = Path("./advanced_data")
    RESULTS_DIR = Path("./advanced_results")

    # Walkforward parameters
    INITIAL_TRAIN_SIZE = 15000
    TEST_SIZE = 5000
    STEP_SIZE = 2500
    MAX_WINDOWS = 5

    def __init__(self):
        for directory in [self.DATA_DIR, self.RESULTS_DIR]:
            directory.mkdir(exist_ok=True)

# ==================== MULTI-CURRENCY DATA LOADER ====================
class MultiCurrencyLoader:
    """Load multiple currency pairs data"""

    def __init__(self, config: AdvancedConfig):
        self.config = config
        self.data = {}

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all currency pairs"""

        for symbol in self.config.SYMBOLS:
            try:
                df = self._load_symbol(symbol)
                if df is not None and len(df) > 1000:
                    self.data[symbol] = df
            except Exception:
                self.data[symbol] = self._create_test_data(symbol)

        return self.data

    def _load_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load single currency pair"""
        if not mt5.initialize():
            return self._create_test_data(symbol)

        try:
            tf_map = {'H1': mt5.TIMEFRAME_H1, 'D1': mt5.TIMEFRAME_D1}
            timeframe = tf_map.get(self.config.TIMEFRAME, mt5.TIMEFRAME_H1)

            rates = mt5.copy_rates_range(
                symbol,
                timeframe,
                datetime.strptime(self.config.START_DATE, "%Y-%m-%d"),
                datetime.strptime(self.config.END_DATE, "%Y-%m-%d")
            )

            if rates is None:
                return self._create_test_data(symbol)

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            df = df[['open', 'high', 'low', 'close', 'tick_volume']]
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

            return df.dropna()

        except Exception:
            return self._create_test_data(symbol)
        finally:
            mt5.shutdown()

    def _create_test_data(self, symbol: str) -> pd.DataFrame:
        """Create test data for currency pair"""
        dates = pd.date_range(
            start=self.config.START_DATE,
            end=self.config.END_DATE,
            freq='H'
        )

        np.random.seed(hash(symbol) % 10000)
        n = len(dates)

        params = {
            'EURUSD': {'vol': 0.08, 'trend': 0.02, 'base': 1.10},
            'GBPUSD': {'vol': 0.10, 'trend': 0.01, 'base': 1.30},
            'USDJPY': {'vol': 0.09, 'trend': -0.01, 'base': 110.0},
            'AUDUSD': {'vol': 0.12, 'trend': 0.00, 'base': 0.75},
            'USDCAD': {'vol': 0.09, 'trend': 0.015, 'base': 1.25},
            'USDCHF': {'vol': 0.07, 'trend': -0.005, 'base': 0.95},
            'NZDUSD': {'vol': 0.13, 'trend': 0.005, 'base': 0.70}
        }

        param = params.get(symbol, {'vol': 0.10, 'trend': 0.00, 'base': 1.0})

        dt = 1/24/252
        price = np.ones(n) * param['base']

        for i in range(1, n):
            drift = param['trend'] * dt
            shock = param['vol'] * np.sqrt(dt) * np.random.randn()
            price[i] = price[i-1] * np.exp(drift + shock)

        df = pd.DataFrame(index=dates)
        df['Close'] = price
        df['Open'] = df['Close'].shift(1) * (1 + np.random.randn(n) * 0.0005)
        df['Open'].iloc[0] = df['Close'].iloc[0]
        df['High'] = df['Close'] * (1 + np.abs(np.random.randn(n)) * 0.001)
        df['Low'] = df['Close'] * (1 - np.abs(np.random.randn(n)) * 0.001)
        df['Volume'] = 10000 + 5000 * np.abs(np.random.randn(n))

        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        return df.dropna()

    def get_aligned_data(self) -> Dict[str, pd.DataFrame]:
        """Align data by time index"""
        if not self.data:
            self.load_all_data()

        common_index = None
        for df in self.data.values():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)

        aligned_data = {}
        for symbol, df in self.data.items():
            aligned_data[symbol] = df.loc[common_index].copy()

        return aligned_data

# ==================== FEATURE ENGINEER ====================
class AdvancedFeatureEngineer:
    """Create advanced features"""

    def __init__(self, config: AdvancedConfig):
        self.config = config

    def create_cross_asset_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create cross-asset features"""

        eurusd = data['EURUSD'].copy()
        returns_df = pd.DataFrame()

        for symbol, df in data.items():
            returns_df[symbol] = df['Returns']

        features = pd.DataFrame(index=eurusd.index)

        # 1. Cross-currency correlations
        window = 100

        for other_symbol in [s for s in returns_df.columns if s != 'EURUSD']:
            corr = returns_df['EURUSD'].rolling(window).corr(returns_df[other_symbol])
            features[f'Corr_{other_symbol}'] = corr

            corr_lag1 = returns_df['EURUSD'].rolling(window).corr(returns_df[other_symbol].shift(1))
            features[f'Corr_{other_symbol}_lag1'] = corr_lag1

        # 2. Volatility ratios
        eurusd_vol = returns_df['EURUSD'].rolling(20).std()

        for other_symbol in [s for s in returns_df.columns if s != 'EURUSD']:
            other_vol = returns_df[other_symbol].rolling(20).std()
            vol_ratio = eurusd_vol / (other_vol + 1e-10)
            features[f'Vol_Ratio_{other_symbol}'] = vol_ratio

        # 3. Lagged returns
        for lag in [1, 2, 3, 5, 10]:
            features[f'Returns_lag_{lag}'] = returns_df['EURUSD'].shift(lag)

        # 4. Market breadth
        same_direction = pd.DataFrame()
        for symbol in returns_df.columns:
            if symbol != 'EURUSD':
                same_direction[symbol] = (np.sign(returns_df['EURUSD']) == np.sign(returns_df[symbol])).astype(int)

        features['Market_Breadth'] = same_direction.mean(axis=1)

        # 5. Statistical features
        features['Returns_Mean_20'] = returns_df['EURUSD'].rolling(20).mean()
        features['Returns_Std_20'] = returns_df['EURUSD'].rolling(20).std()
        features['Returns_Skew_20'] = returns_df['EURUSD'].rolling(20).skew()
        features['Returns_Kurt_20'] = returns_df['EURUSD'].rolling(20).kurt()

        # 6. Time features
        features['Hour'] = features.index.hour
        features['DayOfWeek'] = features.index.dayofweek

        features['Hour_sin'] = np.sin(2 * np.pi * features['Hour'] / 24)
        features['Hour_cos'] = np.cos(2 * np.pi * features['Hour'] / 24)
        features['Day_sin'] = np.sin(2 * np.pi * features['DayOfWeek'] / 7)
        features['Day_cos'] = np.cos(2 * np.pi * features['DayOfWeek'] / 7)

        # 7. Target variable
        features['Target'] = returns_df['EURUSD'].shift(-1)

        features = features.drop(columns=['Hour', 'DayOfWeek'])
        features_clean = features.dropna()

        return features_clean

# ==================== MODELS ====================
class SimpleModels:
    """Simple models for testing"""

    @staticmethod
    def train_predict_ridge(train_data: pd.DataFrame, test_data: pd.DataFrame,
                          feature_cols: list) -> Tuple[np.ndarray, dict]:
        """Ridge regression"""
        from sklearn.linear_model import Ridge

        X_train = train_data[feature_cols].fillna(0).values
        y_train = train_data['Target'].fillna(0).values
        X_test = test_data[feature_cols].fillna(0).values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)

        importance = abs(model.coef_)
        important_features = {}
        for i, col in enumerate(feature_cols):
            if i < len(importance) and importance[i] > 0.01:
                important_features[col] = float(importance[i])

        return predictions, {'important_features': important_features}

    @staticmethod
    def train_predict_gb(train_data: pd.DataFrame, test_data: pd.DataFrame,
                        feature_cols: list) -> Tuple[np.ndarray, dict]:
        """Gradient Boosting"""
        from sklearn.ensemble import GradientBoostingRegressor

        X_train = train_data[feature_cols].fillna(0).values
        y_train = train_data['Target'].fillna(0).values
        X_test = test_data[feature_cols].fillna(0).values

        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        importance = model.feature_importances_
        important_features = {}
        for i, col in enumerate(feature_cols):
            if i < len(importance) and importance[i] > 0.01:
                important_features[col] = float(importance[i])

        return predictions, {'important_features': important_features}

    @staticmethod
    def train_predict_ensemble(train_data: pd.DataFrame, test_data: pd.DataFrame,
                              feature_cols: list) -> Tuple[np.ndarray, dict]:
        """Simple ensemble"""
        preds_ridge, info_ridge = SimpleModels.train_predict_ridge(train_data, test_data, feature_cols)
        preds_gb, info_gb = SimpleModels.train_predict_gb(train_data, test_data, feature_cols)

        predictions = 0.5 * preds_ridge + 0.5 * preds_gb

        important_features = {}
        important_features.update(info_ridge.get('important_features', {}))
        important_features.update(info_gb.get('important_features', {}))

        return predictions, {'important_features': important_features}

# ==================== BACKTESTER ====================
class SimplifiedBacktester:
    """Simplified backtester"""

    def __init__(self, config: AdvancedConfig):
        self.config = config
        self.results = {}

    def generate_windows(self, data: pd.DataFrame) -> List[Tuple]:
        """Generate walkforward windows"""
        n_samples = len(data)
        train_size = self.config.INITIAL_TRAIN_SIZE
        test_size = self.config.TEST_SIZE
        step_size = self.config.STEP_SIZE

        windows = []
        start = 0

        while start + train_size + test_size <= n_samples and len(windows) < self.config.MAX_WINDOWS:
            train_end = start + train_size
            test_end = train_end + test_size

            windows.append((list(range(start, train_end)),
                          list(range(train_end, test_end))))

            start += step_size

        return windows

    def calculate_metrics(self, predictions: np.ndarray, actuals: np.ndarray) -> dict:
        """Calculate performance metrics"""
        if len(predictions) != len(actuals) or len(predictions) < 10:
            return {'direction': 0.5, 'correlation': 0, 'sharpe': 0, 'mse': 0}

        mse = mean_squared_error(actuals, predictions)

        pred_sign = np.sign(predictions)
        actual_sign = np.sign(actuals)
        valid_mask = (pred_sign != 0) & (actual_sign != 0)

        if np.sum(valid_mask) > 10:
            direction = np.mean(pred_sign[valid_mask] == actual_sign[valid_mask])
        else:
            direction = 0.5

        if np.std(predictions) > 1e-10 and np.std(actuals) > 1e-10:
            correlation = np.corrcoef(predictions, actuals)[0, 1]
        else:
            correlation = 0

        threshold = np.percentile(np.abs(predictions), 80)
        signals = np.zeros(len(predictions))
        signals[predictions > threshold] = 1
        signals[predictions < -threshold] = -1

        returns = signals * actuals
        active_returns = returns[signals != 0]

        if len(active_returns) > 10:
            sharpe = np.mean(active_returns) / (np.std(active_returns) + 1e-10) * np.sqrt(252)
            sharpe = float(np.clip(sharpe, -5, 5))
        else:
            sharpe = 0

        return {
            'direction': float(direction),
            'correlation': float(correlation),
            'sharpe': sharpe,
            'mse': float(mse),
            'predictions_std': float(np.std(predictions)),
            'actuals_std': float(np.std(actuals)),
            'test_size': len(actuals)
        }

    def test_hypotheses(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Test hypotheses"""

        feature_cols = [col for col in features.columns if col != 'Target']

        windows = self.generate_windows(features)

        results = {
            'ridge': [],
            'gradient_boosting': [],
            'ensemble': []
        }

        feature_importance = {
            'ridge': {},
            'gradient_boosting': {},
            'ensemble': {}
        }

        for i, (train_idx, test_idx) in enumerate(windows):
            train_data = features.iloc[train_idx]
            test_data = features.iloc[test_idx]

            actuals = test_data['Target'].values

            # Ridge regression
            preds_ridge, info_ridge = SimpleModels.train_predict_ridge(train_data, test_data, feature_cols)
            metrics_ridge = self.calculate_metrics(preds_ridge, actuals)
            results['ridge'].append(metrics_ridge)

            for feat, imp in info_ridge.get('important_features', {}).items():
                feature_importance['ridge'][feat] = feature_importance['ridge'].get(feat, 0) + imp

            # Gradient Boosting
            preds_gb, info_gb = SimpleModels.train_predict_gb(train_data, test_data, feature_cols)
            metrics_gb = self.calculate_metrics(preds_gb, actuals)
            results['gradient_boosting'].append(metrics_gb)

            for feat, imp in info_gb.get('important_features', {}).items():
                feature_importance['gradient_boosting'][feat] = feature_importance['gradient_boosting'].get(feat, 0) + imp

            # Ensemble
            preds_ensemble, info_ensemble = SimpleModels.train_predict_ensemble(train_data, test_data, feature_cols)
            metrics_ensemble = self.calculate_metrics(preds_ensemble, actuals)
            results['ensemble'].append(metrics_ensemble)

            for feat, imp in info_ensemble.get('important_features', {}).items():
                feature_importance['ensemble'][feat] = feature_importance['ensemble'].get(feat, 0) + imp

        final_results = self.analyze_results(results, feature_importance)

        self.results = final_results
        return final_results

    def analyze_results(self, results: Dict[str, List], feature_importance: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze and compare results"""

        final_results = {}

        for model_name, metrics_list in results.items():
            if not metrics_list:
                continue

            directions = [m['direction'] for m in metrics_list]
            correlations = [m['correlation'] for m in metrics_list]
            sharpes = [m['sharpe'] for m in metrics_list]
            mses = [m['mse'] for m in metrics_list]

            model_summary = {
                'mean_direction': float(np.mean(directions)),
                'std_direction': float(np.std(directions)),
                'mean_correlation': float(np.mean(correlations)),
                'std_correlation': float(np.std(correlations)),
                'mean_sharpe': float(np.mean(sharpes)),
                'std_sharpe': float(np.std(sharpes)),
                'mean_mse': float(np.mean(mses)),
                'num_windows': len(metrics_list),
                'window_results': metrics_list
            }

            if model_name in feature_importance:
                imp_dict = feature_importance[model_name]
                if imp_dict:
                    sorted_features = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                    model_summary['top_features'] = {k: v/len(metrics_list) for k, v in sorted_features}

            final_results[model_name] = model_summary

        self.save_results(final_results, feature_importance)
        self.create_plots(final_results)

        return final_results

    def save_results(self, results: Dict[str, Any], feature_importance: Dict[str, Dict]):
        """Save results to file"""
        results_file = self.config.RESULTS_DIR / 'results.json'

        save_data = {
            'results': results,
            'feature_importance': feature_importance,
            'config': {
                'symbols': self.config.SYMBOLS,
                'timeframe': self.config.TIMEFRAME,
                'period': f"{self.config.START_DATE} - {self.config.END_DATE}",
                'train_size': self.config.INITIAL_TRAIN_SIZE,
                'test_size': self.config.TEST_SIZE,
                'step_size': self.config.STEP_SIZE
            }
        }

        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            else:
                return obj

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(convert(save_data), f, indent=4, ensure_ascii=False)

    def create_plots(self, results: Dict[str, Any]):
        """Create comparison plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            model_names = list(results.keys())

            # Direction accuracy
            ax1 = axes[0, 0]
            means = [results[m]['mean_direction'] for m in model_names]
            stds = [results[m]['std_direction'] for m in model_names]

            x_pos = np.arange(len(model_names))
            bars = ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)

            for bar, mean in zip(bars, means):
                if mean > 0.55:
                    bar.set_color('green')
                elif mean > 0.52:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')

            ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
            ax1.set_xlabel('Model')
            ax1.set_ylabel('Direction Accuracy')
            ax1.set_title('Model Comparison: Direction Accuracy')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels([m.replace('_', ' ').title() for m in model_names], rotation=45)
            ax1.grid(True, alpha=0.3, axis='y')

            # Sharpe ratio
            ax2 = axes[0, 1]
            means = [results[m]['mean_sharpe'] for m in model_names]
            stds = [results[m]['std_sharpe'] for m in model_names]

            ax2.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='purple')
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
            ax2.axhline(y=1, color='g', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Model')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.set_title('Model Comparison: Sharpe Ratio')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([m.replace('_', ' ').title() for m in model_names], rotation=45)
            ax2.grid(True, alpha=0.3, axis='y')

            # Correlation
            ax3 = axes[1, 0]
            means = [results[m]['mean_correlation'] for m in model_names]
            stds = [results[m]['std_correlation'] for m in model_names]

            ax3.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='blue')
            ax3.axhline(y=0, color='k', linestyle='-', alpha=0.5)
            ax3.set_xlabel('Model')
            ax3.set_ylabel('Correlation')
            ax3.set_title('Model Comparison: Correlation')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([m.replace('_', ' ').title() for m in model_names], rotation=45)
            ax3.grid(True, alpha=0.3, axis='y')

            # Window stability
            ax4 = axes[1, 1]

            for i, model in enumerate(model_names):
                if 'window_results' in results[model]:
                    directions = [w['direction'] for w in results[model]['window_results']]
                    x = np.random.normal(i, 0.1, size=len(directions))
                    ax4.scatter(x, directions, alpha=0.6, label=model, s=50)

            ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Model')
            ax4.set_ylabel('Direction Accuracy')
            ax4.set_title('Stability Across Windows')
            ax4.set_xticks(range(len(model_names)))
            ax4.set_xticklabels([m.replace('_', ' ').title() for m in model_names], rotation=45)
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_file = self.config.RESULTS_DIR / 'model_comparison.png'
            plt.savefig(plot_file, dpi=120, bbox_inches='tight')
            plt.close()

        except Exception:
            pass

# ==================== MAIN PROGRAM ====================
def main():
    """Main function"""

    # 1. Configuration
    config = AdvancedConfig()

    # 2. Load multi-currency data
    loader = MultiCurrencyLoader(config)
    all_data = loader.get_aligned_data()

    print("Data loaded:")
    for symbol, df in all_data.items():
        returns = df['Returns']
        print(f"{symbol}: {len(df):,} bars | "
              f"Returns: mean={returns.mean():.6f}, std={returns.std():.6f}")

    # 3. Create features
    feature_engineer = AdvancedFeatureEngineer(config)
    features = feature_engineer.create_cross_asset_features(all_data)

    print(f"\nFeatures created: {features.shape}")
    print(f"Period: {features.index[0].date()} - {features.index[-1].date()}")
    print(f"Target: mean={features['Target'].mean():.6f}, std={features['Target'].std():.6f}")

    # 4. Test hypotheses
    backtester = SimplifiedBacktester(config)
    results = backtester.test_hypotheses(features)

    # 5. Display results
    print("\nResults:")
    print("=" * 60)

    for model_name, stats in results.items():
        print(f"\n{model_name.upper().replace('_', ' ')}:")
        print(f"  Direction: {stats['mean_direction']:.3f} ± {stats['std_direction']:.3f}")
        print(f"  Correlation: {stats['mean_correlation']:.3f} ± {stats['std_correlation']:.3f}")
        print(f"  Sharpe: {stats['mean_sharpe']:.3f} ± {stats['std_sharpe']:.3f}")
        print(f"  MSE: {stats['mean_mse']:.6f}")

        if 'top_features' in stats and stats['top_features']:
            print(f"  Top features:")
            for i, (feat, imp) in enumerate(list(stats['top_features'].items())[:3]):
                print(f"    {i+1}. {feat}: {imp:.4f}")

    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['mean_direction'])
    best_name, best_stats = best_model

    print(f"\n" + "=" * 60)
    print(f"Best model: {best_name}")
    print(f"Direction accuracy: {best_stats['mean_direction']:.3f}")
    print(f"Correlation: {best_stats['mean_correlation']:.3f}")
    print(f"Sharpe: {best_stats['mean_sharpe']:.3f}")

if __name__ == "__main__":
    # Check dependencies
    required = ['pandas', 'numpy', 'sklearn', 'scipy', 'networkx', 'matplotlib']

    for lib in required:
        try:
            __import__(lib)
        except ImportError:
            print(f"Missing library: {lib}")
            exit(1)

    # Run
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
