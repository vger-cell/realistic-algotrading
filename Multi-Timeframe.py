"""
Feature Effectiveness Analysis for Multi-Timeframe (MTF) Trading Strategy
Author: Vladimir Korneev
Repository: github.com/vger-cell/realistic-algotrading
Telegram: t.me/realistic_algotrading

Objective:
Identify the most predictive technical features for EURUSD price movement forecasting
using M15 as base timeframe and H1/H4 for multi-timeframe context.

Methodology:
- Loads OHLC data from MetaTrader 5 for M15, H1, H4 (2024–2026)
- Engineers 15 candidate features including position-based MTF metrics
- Evaluates each feature by incremental R² improvement over baseline (log_return + high_low_range)
- Uses time-series split (70% train / 15% val / 15% test) with RandomForestRegressor
- Filters features by ΔR² > 0.001

Key Insight:
Relative price position within H1/H4 ranges (e.g., H1_position) dominates predictive power,
outperforming traditional indicators (EMA, volatility, z-score).

Output:
- JSON report with feature rankings
- Recommended feature set for walk-forward strategy development
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib
import json
import pytz
import MetaTrader5 as mt5


# ==================== CONFIGURATION ====================
class Config:
    SYMBOL = "EURUSD"

    # Timeframe settings
    MT5_TFS = {
        'M15': mt5.TIMEFRAME_M15,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4
    }
    BASE_TIMEFRAME = 'M15'

    # Data period
    START_DATE = datetime(2024, 1, 1, tzinfo=pytz.UTC)
    END_DATE = datetime.now(pytz.UTC)

    # Validation settings (as percentage of data)
    TRAIN_SIZE_PERCENT = 0.7  # 70% for training
    VAL_SIZE_PERCENT = 0.15   # 15% for validation
    TEST_SIZE_PERCENT = 0.15  # 15% for testing

    # Model parameters
    MODEL_TYPE = 'rf'
    N_ESTIMATORS = 100  # Reduced for speed
    MAX_DEPTH = 8
    MIN_SAMPLES_SPLIT = 20

    PERMUTATION_REPEATS = 3
    IMPORTANCE_THRESHOLD = 0.001  # 0.1% improvement

    # Minimum data points required
    MIN_DATA_POINTS = 1000


# ==================== DATA LOADING ====================
class MT5DataFetcher:
    def __init__(self):
        if not mt5.initialize():
            print("❌ MT5 initialization failed")
            raise ConnectionError("Failed to connect to MT5")

        if not mt5.symbol_select(Config.SYMBOL, True):
            print(f"❌ Symbol {Config.SYMBOL} is unavailable")
            mt5.shutdown()
            raise ValueError(f"Symbol {Config.SYMBOL} is unavailable")

    def load_timeframe_data(self, symbol, timeframe, start_date, end_date):
        """Load data for a specific timeframe"""
        try:
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
            if rates is None or len(rates) == 0:
                print(f"  ⚠️ No data via copy_rates_range, trying copy_rates_from...")
                rates = mt5.copy_rates_from(symbol, timeframe, start_date, 10000)

            if rates is None or len(rates) == 0:
                return None

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            df.set_index('time', inplace=True)
            return df[['open', 'high', 'low', 'close', 'tick_volume']].copy()

        except Exception as e:
            print(f"  ✗ Load error: {e}")
            return None

    def load_all_timeframes(self):
        """Load data for all timeframes"""
        print(f"📅 Loading period: {Config.START_DATE} → {Config.END_DATE}")

        data = {}
        for name, tf in Config.MT5_TFS.items():
            print(f"  Loading {name}...", end=" ")
            df = self.load_timeframe_data(Config.SYMBOL, tf, Config.START_DATE, Config.END_DATE)

            if df is not None and len(df) > 0:
                data[name] = df
                print(f"✓ {len(df)} bars")
            else:
                print(f"✗ No data")

        return data

    def get_base_data_for_analysis(self):
        """Get base data for feature analysis"""
        print(f"\n📊 Loading data for feature analysis...")

        # Load all data
        all_data = self.load_all_timeframes()

        if Config.BASE_TIMEFRAME not in all_data:
            print(f"❌ No data for base TF {Config.BASE_TIMEFRAME}")
            return None, None

        # Use M15 as base TF
        base_df = all_data[Config.BASE_TIMEFRAME].copy()

        print(f"\n  ✅ Base TF {Config.BASE_TIMEFRAME}: {len(base_df)} bars")
        print(f"  📅 Period: {base_df.index[0]} - {base_df.index[-1]}")

        # Align data from other TFs
        aligned = {'M15': base_df['close']}
        for tf in ['H1', 'H4']:
            if tf in all_data:
                # Resample to 15-minute TF
                tf_close = all_data[tf]['close'].resample('15min').ffill()
                aligned[tf] = tf_close.reindex(base_df.index, method='ffill')

                print(f"  ✅ {tf}: aligned {aligned[tf].notna().sum()} bars")

        return base_df, aligned

    def __del__(self):
        mt5.shutdown()


# ==================== FEATURE ENGINEERING ====================
class FeatureEngineer:
    @staticmethod
    def create_features_simple(base_df, aligned_data):
        """Simple feature version — only essentials"""
        features_df = pd.DataFrame(index=base_df.index)

        # 1. Base features (mandatory)
        features_df['log_return'] = np.log(base_df['close']) - np.log(base_df['close'].shift(1))
        features_df['high_low_range'] = (base_df['high'] - base_df['low']) / base_df['close']

        # 2. Simple trend features
        features_df['price_zscore_20'] = (base_df['close'] - base_df['close'].rolling(20).mean()) / base_df[
            'close'].rolling(20).std()
        features_df['ema_ratio'] = (base_df['close'].ewm(span=12).mean() / base_df['close'].ewm(span=26).mean()) - 1

        # 3. Volatility
        features_df['volatility_20'] = features_df['log_return'].rolling(20).std()

        # 4. MTF features (main focus)
        for tf in ['H1', 'H4']:
            if tf in aligned_data:
                tf_price = aligned_data[tf]

                # Core feature: relative position
                rolling_min = tf_price.rolling(50).min()
                rolling_max = tf_price.rolling(50).max()
                range_val = rolling_max - rolling_min

                # Safe calculation
                position = np.zeros(len(base_df))
                mask = range_val > 0
                position[mask] = (base_df['close'].values[mask] - rolling_min.values[mask]) / range_val.values[mask]
                position[~mask] = 0.5

                features_df[f'{tf}_position'] = position

                # Simple trend
                features_df[f'{tf}_trend'] = np.sign(tf_price - tf_price.rolling(20).mean())

                # Distance from MA
                features_df[f'{tf}_dist_ma'] = (base_df['close'] - tf_price.rolling(20).mean()) / base_df['close']

        # 5. Lags (minimal)
        features_df['log_return_lag1'] = features_df['log_return'].shift(1)
        features_df['log_return_lag2'] = features_df['log_return'].shift(2)

        # 6. Simple interactions
        if 'H4_position' in features_df.columns:
            features_df['vol_pos_interaction'] = features_df['volatility_20'] * features_df['H4_position']
            features_df['trend_pos_interaction'] = features_df['price_zscore_20'] * features_df['H4_position']

        # Target variable
        features_df['target'] = features_df['log_return'].shift(-1)

        # Remove NaN
        features_df_clean = features_df.dropna()

        if len(features_df_clean) > 0:
            print(f"✅ Features created: {len(features_df_clean.columns) - 1}")
            print(f"📏 Dataset size: {len(features_df_clean)} rows")
        else:
            print("⚠️ No features remain after NaN removal")

        return features_df_clean

    @staticmethod
    def create_features_extended(base_df, aligned_data):
        """Extended version with error protection"""
        try:
            features_df = pd.DataFrame(index=base_df.index)

            # 1. Base features with safeguards
            features_df['log_return'] = np.log(base_df['close']) - np.log(base_df['close'].shift(1))
            features_df['high_low_range'] = (base_df['high'] - base_df['low']) / (base_df['close'] + 1e-10)
            features_df['close_open_ratio'] = base_df['close'] / (base_df['open'] + 1e-10) - 1

            # 2. Trend features
            rolling_mean_20 = base_df['close'].rolling(20, min_periods=10).mean()
            rolling_std_20 = base_df['close'].rolling(20, min_periods=10).std()
            features_df['price_zscore_20'] = np.where(
                rolling_std_20 > 0,
                (base_df['close'] - rolling_mean_20) / rolling_std_20,
                0
            )

            # 3. Volatility
            features_df['volatility_20'] = features_df['log_return'].rolling(20, min_periods=10).std()
            features_df['volatility_50'] = features_df['log_return'].rolling(50, min_periods=20).std()

            # 4. MTF features (core)
            for tf in ['H1', 'H4']:
                if tf in aligned_data:
                    tf_price = aligned_data[tf]

                    # Safe position calculation
                    rolling_min = tf_price.rolling(50, min_periods=25).min()
                    rolling_max = tf_price.rolling(50, min_periods=25).max()
                    range_val = rolling_max - rolling_min

                    position = np.where(
                        range_val > 0,
                        (base_df['close'] - rolling_min) / range_val,
                        0.5
                    )

                    features_df[f'{tf}_position'] = position

                    # Position derivatives
                    position_series = pd.Series(position, index=base_df.index)
                    features_df[f'{tf}_position_diff'] = position_series.diff()
                    features_df[f'{tf}_position_ma'] = position_series.rolling(10, min_periods=5).mean()

                    # Trend
                    features_df[f'{tf}_trend'] = np.where(
                        tf_price.rolling(20, min_periods=10).mean() > 0,
                        np.sign(tf_price - tf_price.rolling(20, min_periods=10).mean()),
                        0
                    )

            # 5. Lags
            for lag in [1, 2, 3, 5, 10]:
                features_df[f'log_return_lag{lag}'] = features_df['log_return'].shift(lag)

            # 6. Interactions
            if 'H4_position' in features_df.columns:
                features_df['volatility_position'] = features_df['volatility_20'] * features_df['H4_position']
                features_df['range_position'] = features_df['high_low_range'] * features_df['H4_position']

                # Extreme zones
                features_df['H4_extreme_high'] = (features_df['H4_position'] > 0.8).astype(float)
                features_df['H4_extreme_low'] = (features_df['H4_position'] < 0.2).astype(float)

            # 7. Combined MTF features
            if 'H1_position' in features_df.columns and 'H4_position' in features_df.columns:
                features_df['position_diff_H4_H1'] = features_df['H4_position'] - features_df['H1_position']
                features_df['position_avg'] = (features_df['H4_position'] + features_df['H1_position']) / 2

            # Target variable
            features_df['target'] = features_df['log_return'].shift(-1)

            # Remove NaN
            features_df_clean = features_df.dropna()

            if len(features_df_clean) > 0:
                print(
                    f"✅ Extended version: {len(features_df_clean.columns) - 1} features, {len(features_df_clean)} rows")
                return features_df_clean
            else:
                print("⚠️ Extended version: no data remains after NaN removal")
                return None

        except Exception as e:
            print(f"❌ Error creating extended features: {e}")
            return None


# ==================== VALIDATION & EVALUATION ====================
class TimeSeriesValidator:
    def __init__(self, train_size=0.7, val_size=0.15, test_size=0.15):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

    def create_splits(self, df):
        """Create train/val/test splits by percentage"""
        if len(df) < Config.MIN_DATA_POINTS:
            raise ValueError(f"Insufficient data: {len(df)} < {Config.MIN_DATA_POINTS}")

        n = len(df)
        train_end = int(n * self.train_size)
        val_end = train_end + int(n * self.val_size)

        # Create masks
        indices = np.arange(n)
        train_mask = indices < train_end
        val_mask = (indices >= train_end) & (indices < val_end)
        test_mask = indices >= val_end

        dates = df.index

        print(f"\n📊 DATA SPLIT:")
        print(f"   Total: {n:,} bars")
        print(f"   Train: {train_mask.sum():,} bars ({train_mask.sum() / n * 100:.1f}%)")
        print(f"        {dates[0]} - {dates[train_end - 1]}")
        print(f"   Val:   {val_mask.sum():,} bars ({val_mask.sum() / n * 100:.1f}%)")
        print(f"        {dates[train_end]} - {dates[val_end - 1]}")
        print(f"   Test:  {test_mask.sum():,} bars ({test_mask.sum() / n * 100:.1f}%)")
        print(f"        {dates[val_end]} - {dates[-1]}")

        return train_mask, val_mask, test_mask


# ==================== MODELING & ANALYSIS ====================
class FeatureAnalyzer:
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.results = {}
        self.improvements_df = None

    def create_model(self):
        """Create optimized model"""
        return RandomForestRegressor(
            n_estimators=self.config.N_ESTIMATORS,
            max_depth=self.config.MAX_DEPTH,
            min_samples_split=self.config.MIN_SAMPLES_SPLIT,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )

    def calculate_metrics(self, y_true, y_pred):
        """Calculate all metrics"""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }

        # Normalized metrics
        y_std = np.std(y_true)
        if y_std > 0:
            metrics['normalized_mse'] = metrics['mse'] / (y_std ** 2)
            metrics['normalized_rmse'] = metrics['rmse'] / y_std

        return metrics

    def evaluate_feature_group(self, X_train, y_train, X_val, y_val, feature_names, group_name):
        """Evaluate feature group effectiveness"""
        print(f"\n{'=' * 60}")
        print(f"📊 EVALUATING GROUP: {group_name}")
        print(f"   Features: {len(feature_names)}")

        if len(X_train) < 50 or len(X_val) < 20:
            print(f"   ⚠️ Insufficient data: train={len(X_train)}, val={len(X_val)}")
            return None, None

        try:
            # Scaling
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Save scaler
            self.scalers[group_name] = scaler

            # Create and train model
            model = self.create_model()
            model.fit(X_train_scaled, y_train)

            # Predictions
            y_pred = model.predict(X_val_scaled)

            # Metrics
            metrics = self.calculate_metrics(y_val, y_pred)

            # Feature importance
            importances = model.feature_importances_

            # Importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances,
                'importance_%': (importances / importances.sum()) * 100
            }).sort_values('importance', ascending=False)

            print(f"\n   📈 VALIDATION METRICS:")
            print(f"      R²:  {metrics['r2']:+.4f}")
            print(f"      RMSE: {metrics['rmse']:.6f}")
            print(f"      MAE:  {metrics['mae']:.6f}")

            print(f"\n   🎯 TOP-3 FEATURES:")
            print(importance_df.head(3).to_string())

            # Save results
            self.results[group_name] = {
                'metrics': metrics,
                'importance_df': importance_df,
                'model': model
            }

            return metrics, importance_df

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None, None

    def analyze_feature_interactions(self, features_df, base_features):
        """Analyze all features"""
        if features_df is None or len(features_df) == 0:
            print("❌ No data for analysis")
            return pd.DataFrame(), []

        print(f"\n{'=' * 60}")
        print("🔬 FEATURE ANALYSIS")
        print('=' * 60)

        try:
            # Create splits
            validator = TimeSeriesValidator(
                Config.TRAIN_SIZE_PERCENT,
                Config.VAL_SIZE_PERCENT,
                Config.TEST_SIZE_PERCENT
            )
            train_mask, val_mask, _ = validator.create_splits(features_df)

            # All features except target
            all_features = [f for f in features_df.columns if f != 'target']

            # Evaluate baseline model
            print(f"\n📊 BASELINE MODEL:")
            X_base = features_df[base_features]
            y = features_df['target']

            X_train_base = X_base[train_mask]
            X_val_base = X_base[val_mask]
            y_train = y[train_mask]
            y_val = y[val_mask]

            base_metrics, _ = self.evaluate_feature_group(
                X_train_base, y_train, X_val_base, y_val,
                base_features, 'base_model'
            )

            if base_metrics is None:
                print("❌ Failed to evaluate baseline model")
                return pd.DataFrame(), []

            # Test each additional feature
            improvements = []
            candidate_features = [f for f in all_features if f not in base_features]

            print(f"\n🔍 TESTING {len(candidate_features)} FEATURES...")

            for feature in tqdm(candidate_features, desc="Analyzing"):
                # Add feature to baseline
                current_features = base_features + [feature]
                X_current = features_df[current_features]

                X_train = X_current[train_mask]
                X_val = X_current[val_mask]

                # Evaluate
                result = self.evaluate_feature_group(
                    X_train, y_train, X_val, y_val,
                    current_features, f'with_{feature}'
                )

                if result[0] is None:
                    continue

                metrics, importance_df = result

                # Compute improvement
                r2_improvement = metrics['r2'] - base_metrics['r2']
                rmse_improvement_pct = ((base_metrics['rmse'] - metrics['rmse']) / base_metrics['rmse']) * 100

                # Importance of added feature
                feature_importance = 0
                if importance_df is not None and feature in importance_df['feature'].values:
                    feature_importance = importance_df.loc[
                        importance_df['feature'] == feature, 'importance_%'
                    ].values[0]

                # Determine category
                is_mtf = any(tf in feature for tf in ['H1', 'H4'])
                is_position = 'position' in feature
                is_interaction = 'interaction' in feature or 'volatility_position' in feature or 'range_position' in feature

                improvements.append({
                    'feature': feature,
                    'r2': metrics['r2'],
                    'rmse': metrics['rmse'],
                    'r2_improvement': r2_improvement,
                    'rmse_improvement_%': rmse_improvement_pct,
                    'feature_importance_%': feature_importance,
                    'is_mtf': is_mtf,
                    'is_position': is_position,
                    'is_interaction': is_interaction
                })

            if not improvements:
                print("❌ Failed to analyze features")
                return pd.DataFrame(), []

            # Create results DataFrame
            self.improvements_df = pd.DataFrame(improvements).sort_values('r2_improvement', ascending=False)

            print(f"\n{'=' * 60}")
            print("📈 FEATURE ANALYSIS RESULTS")
            print('=' * 60)
            print(f"   Baseline R²:  {base_metrics['r2']:+.4f}")
            print(f"   Baseline RMSE: {base_metrics['rmse']:.6f}")

            print(f"\n   🏆 TOP-10 FEATURES BY R² IMPROVEMENT:")
            top_10 = self.improvements_df.head(10)[
                ['feature', 'r2_improvement', 'rmse_improvement_%', 'feature_importance_%']]
            print(top_10.to_string())

            # Statistics
            self._print_statistics()

            # Recommended features
            recommended = self.improvements_df[
                self.improvements_df['r2_improvement'] > self.config.IMPORTANCE_THRESHOLD
                ]['feature'].tolist()

            print(f"\n   ✅ RECOMMENDED ({len(recommended)}):")
            for i, feat in enumerate(recommended[:10], 1):
                row = self.improvements_df[self.improvements_df['feature'] == feat].iloc[0]
                print(f"      {i:2d}. {feat:<25} ΔR²: {row['r2_improvement']:+.4f}")

            return self.improvements_df, recommended

        except Exception as e:
            print(f"❌ Analysis error: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(), []

    def _print_statistics(self):
        """Print statistics"""
        if self.improvements_df is None or self.improvements_df.empty:
            return

        print(f"\n   📊 STATISTICS:")

        # Overall stats
        pos_improvements = self.improvements_df[self.improvements_df['r2_improvement'] > 0]
        neg_improvements = self.improvements_df[self.improvements_df['r2_improvement'] < 0]

        print(f"      Features with positive effect: {len(pos_improvements)}/{len(self.improvements_df)}")
        print(f"      Features with negative effect: {len(neg_improvements)}/{len(self.improvements_df)}")
        print(f"      Average R² improvement: {self.improvements_df['r2_improvement'].mean():+.4f}")

        # By category
        categories = {
            'MTF': self.improvements_df['is_mtf'],
            'Position': self.improvements_df['is_position'],
            'Interaction': self.improvements_df['is_interaction']
        }

        for cat_name, mask in categories.items():
            if mask.any():
                cat_data = self.improvements_df[mask]
                avg_improvement = cat_data['r2_improvement'].mean()
                count = len(cat_data)
                print(f"      {cat_name} ({count}): average ΔR² = {avg_improvement:+.4f}")

        # Best per category
        print(f"\n   🥇 BEST BY CATEGORY:")

        for cat_name, mask in categories.items():
            if mask.any():
                best = self.improvements_df[mask].iloc[0]
                print(f"      {cat_name}: {best['feature']} (ΔR²: {best['r2_improvement']:+.4f})")


# ==================== MAIN PIPELINE ====================
def main():
    print("=" * 60)
    print("🚀 FEATURE EFFECTIVENESS ANALYSIS FOR EURUSD")
    print("=" * 60)

    try:
        # 1. Load data
        print("\n1. 📥 DATA LOADING")
        print("   " + "-" * 50)

        fetcher = MT5DataFetcher()
        base_df, aligned_data = fetcher.get_base_data_for_analysis()

        if base_df is None or len(base_df) == 0:
            print("❌ Failed to load data")
            return

        # 2. Create features
        print("\n2. 🔧 FEATURE CREATION")
        print("   " + "-" * 50)

        engineer = FeatureEngineer()

        print("   Simple version...")
        features_simple = engineer.create_features_simple(base_df, aligned_data)

        print("   Extended version...")
        features_extended = engineer.create_features_extended(base_df, aligned_data)

        # Choose version with data
        if features_simple is not None and len(features_simple) > 0:
            features_df = features_simple
            print(f"   ✅ Using simple version: {len(features_df)} rows")
        elif features_extended is not None and len(features_extended) > 0:
            features_df = features_extended
            print(f"   ✅ Using extended version: {len(features_df)} rows")
        else:
            print("❌ Failed to create features")
            return

        # 3. Analyze features
        print("\n3. 🔬 FEATURE EFFECTIVENESS ANALYSIS")
        print("   " + "-" * 50)

        analyzer = FeatureAnalyzer(Config())

        # Base features (minimal set)
        base_features = ['log_return', 'high_low_range']

        print(f"   Base features: {base_features}")
        print(f"   Total features: {len(features_df.columns) - 1}")

        # Run analysis
        improvements_df, recommended_features = analyzer.analyze_feature_interactions(
            features_df, base_features
        )

        if improvements_df.empty or not recommended_features:
            print("\n❌ No significant features found")
            return

        # 4. Visualization
        print("\n4. 📊 VISUALIZATION")
        print("   " + "-" * 50)

        if not improvements_df.empty:
            # Simple visualization
            plt.figure(figsize=(12, 6))

            # R² improvement bar plot
            top_n = min(15, len(improvements_df))
            top_features = improvements_df.head(top_n)

            colors = []
            for _, row in top_features.iterrows():
                if row['is_mtf']:
                    colors.append('orange')
                elif row['is_position']:
                    colors.append('green')
                elif row['is_interaction']:
                    colors.append('red')
                else:
                    colors.append('blue')

            bars = plt.bar(range(top_n), top_features['r2_improvement'], color=colors)
            plt.xlabel('Feature')
            plt.ylabel('R² Improvement')
            plt.title(f'Top-{top_n} Features by R² Improvement')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xticks(range(top_n), top_features['feature'], rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        # 5. Save results
        print("\n5. 💾 SAVING RESULTS")
        print("   " + "-" * 50)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Compile results
        results = {
            'timestamp': timestamp,
            'symbol': Config.SYMBOL,
            'period': {
                'start': Config.START_DATE.strftime("%Y-%m-%d"),
                'end': Config.END_DATE.strftime("%Y-%m-%d")
            },
            'data_info': {
                'total_rows': len(features_df),
                'features_count': len(features_df.columns) - 1,
                'base_features': base_features
            },
            'analysis_results': {
                'top_5_features': improvements_df.head(5)[['feature', 'r2_improvement']].to_dict('records'),
                'recommended_features': recommended_features[:10],
                'best_improvement': float(improvements_df['r2_improvement'].max()),
                'avg_improvement': float(improvements_df['r2_improvement'].mean())
            },
            'mtf_insights': {
                'mtf_features_in_top_10': sum(
                    1 for f in improvements_df.head(10)['feature'] if any(tf in f for tf in ['H1', 'H4'])),
                'position_features_in_top_10': sum(1 for f in improvements_df.head(10)['feature'] if 'position' in f),
                'best_mtf_feature': improvements_df[improvements_df['is_mtf']].iloc[0]['feature'] if improvements_df[
                    'is_mtf'].any() else None
            }
        }

        filename = f"feature_analysis_{Config.SYMBOL}_{timestamp}"

        # Save JSON
        with open(f"{filename}.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"   ✅ Results saved to {filename}.json")

        # 6. Final recommendations
        print(f"\n6. 💡 FINAL RECOMMENDATIONS")
        print("   " + "-" * 50)

        if not improvements_df.empty:
            best_feature = improvements_df.iloc[0]
            print(f"\n   🏆 MOST EFFECTIVE FEATURE:")
            print(f"      {best_feature['feature']}")
            print(f"      R² improvement: {best_feature['r2_improvement']:+.4f}")
            print(f"      RMSE reduction: {best_feature['rmse_improvement_%']:+.1f}%")

            print(f"\n   🎯 RECOMMENDED FEATURE SET:")
            recommended_set = base_features + recommended_features[:5]
            for i, feat in enumerate(recommended_set, 1):
                print(f"      {i:2d}. {feat}")

            print(f"\n   📊 KEY INSIGHTS:")
            print(
                f"      1. {'MTF' if results['mtf_insights']['mtf_features_in_top_10'] > 0 else 'Non-MTF'} features dominate the top")
            print(f"      2. Position features: {results['mtf_insights']['position_features_in_top_10']} in top-10")
            print(f"      3. Best feature improves model by {best_feature['r2_improvement']:+.4f} R²")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETED")
        print("=" * 60)


if __name__ == "__main__":
    main()
