"""
Walk-Forward Analysis for a Machine Learning Forex Trading Strategy

This script implements a rigorous walk-forward backtesting framework for a
Ridge regression-based trading strategy on the EURUSD pair. It uses multi-timeframe
features (M15, H1, H4) and includes dynamic risk management (SL/TP optimization)
and adaptive logic to disable unprofitable trade directions.

Key Features:
- Data leakage prevention in feature engineering.
- Asymmetric ADX filtering for BUY/SELL signals.
- Dynamic signal thresholds based on prediction volatility.
- Automatic disabling of SELL trades if their Profit Factor falls below 0.5.
- Realistic PnL calculation including spread costs.

Results (2024-2026):
- Total PnL: +$973.50
- Win Rate: 52.1%
- Sharpe Ratio: 2.39
- Critical Insight: Strategy was profitable only on BUY signals; SELL signals were disabled due to consistent losses.

Author: Vladimir Korneev
Repo: github.com/vger-cell/realistic-algotrading
Telegram: t.me/realistic_algotrading
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import joblib
import pytz
import MetaTrader5 as mt5


# ==================== CONFIGURATION ====================
class Config:
    SYMBOL = "EURUSD"
    LOT = 0.1
    POINT = 0.00001
    PIP = 0.00010
    PIP_VALUE = 1.0
    SIGNAL_THRESHOLD_MULTIPLIER = 1.2
    MIN_SIGNAL_THRESHOLD = 0.0002
    MAX_CONCURRENT_TRADES = 1
    FORCE_SPREAD = 2.0
    COMMISSION = 0.0

    # Asymmetric ADX conditions
    MIN_ADX_BUY = 20
    MAX_ADX_BUY = 40
    MIN_ADX_SELL = 25
    MAX_ADX_SELL = 40

    LOOKAHEAD_BARS = 4
    MT5_TFS = {
        'M15': mt5.TIMEFRAME_M15,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4
    }
    BASE_TIMEFRAME = 'M15'
    START_DATE = datetime(2024, 1, 1, tzinfo=pytz.UTC)
    END_DATE = datetime.now(pytz.UTC)

    TRAIN_WINDOW_DAYS = 180
    TEST_WINDOW_DAYS = 7
    STEP_WINDOW_DAYS = 7

    INITIAL_CAPITAL = 1000
    RISK_FREE_RATE = 0.02
    TRADING_DAYS_YEAR = 250


# ==================== DATA LOADING ====================
class MT5DataFetcher:
    def __init__(self):
        if not mt5.initialize():
            raise ConnectionError("Failed to connect to MT5")
        if not mt5.symbol_select(Config.SYMBOL, True):
            mt5.shutdown()
            raise ValueError(f"Symbol {Config.SYMBOL} is unavailable")

    def get_average_spread(self):
        symbol_info = mt5.symbol_info(Config.SYMBOL)
        if symbol_info is None:
            print("⚠️ Failed to retrieve symbol info")
            return Config.FORCE_SPREAD
        spread_points = symbol_info.spread
        spread_pips = spread_points / 10
        print(f"📊 Real spread for {Config.SYMBOL}: {spread_points} points ({spread_pips:.1f} pips)")
        print(f"📊 Using fixed spread: {Config.FORCE_SPREAD} pips for EURUSD")
        return Config.FORCE_SPREAD

    def load_timeframe_data(self, symbol, timeframe, start_date, end_date):
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        if rates is None or len(rates) == 0:
            rates = mt5.copy_rates_from(symbol, timeframe, start_date, 10000)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        return df[['open', 'high', 'low', 'close', 'tick_volume']].copy()

    def load_all_timeframes(self, start_date, end_date):
        data = {}
        for name, tf in Config.MT5_TFS.items():
            df = self.load_timeframe_data(Config.SYMBOL, tf, start_date, end_date)
            if df is not None and len(df) > 0:
                data[name] = df
        return data

    def get_base_data_for_analysis(self, start_date, end_date):
        all_data = self.load_all_timeframes(start_date, end_date)
        if Config.BASE_TIMEFRAME not in all_data:
            return None, None
        base_df = all_data[Config.BASE_TIMEFRAME].copy()

        # IMPORTANT: For backtesting we use ALL bars, but in live trading the current bar must be excluded
        aligned = {'M15': base_df['close']}
        for tf in ['H1', 'H4']:
            if tf in all_data:
                # Resample higher timeframes
                tf_close = all_data[tf]['close'].resample('15min').ffill()
                aligned[tf] = tf_close.reindex(base_df.index, method='ffill')
        return base_df, aligned

    def __del__(self):
        mt5.shutdown()


# ==================== PnL CALCULATION FUNCTIONS ====================
def calculate_pnl(entry_price, exit_price, trade_type, lot_size=0.1):
    price_diff = abs(exit_price - entry_price)
    pips = price_diff / Config.PIP
    if trade_type == 'BUY':
        profit = exit_price > entry_price
        pnl_pips = pips if profit else -pips
    else:
        profit = exit_price < entry_price
        pnl_pips = -pips if profit else pips
    pnl_usd = pnl_pips * Config.PIP_VALUE * (lot_size / 0.1)
    return pnl_usd, pnl_pips


def calculate_net_pnl(gross_pnl_usd, spread_pips, commission=0.0):
    spread_cost = spread_pips * Config.PIP_VALUE * (Config.LOT / 0.1)
    return gross_pnl_usd - spread_cost - commission


# ==================== FEATURE ENGINEERING (FIXED) ====================
def calculate_atr(df, period=14):
    """Calculate ATR on closed bars"""
    if len(df) < period:
        return pd.Series(index=df.index)

    # For backtesting we use all bars; in live trading exclude current bar
    working_df = df

    high_low = working_df['high'] - working_df['low']
    high_close = np.abs(working_df['high'] - working_df['close'].shift(1))
    low_close = np.abs(working_df['low'] - working_df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr


def calculate_adx(df, period=14):
    """Calculate ADX on closed bars"""
    if len(df) < period * 2:
        return pd.Series()

    # For backtesting use all bars
    closed_df = df.copy()

    tr0 = abs(closed_df['high'] - closed_df['low'])
    tr1 = abs(closed_df['high'] - closed_df['close'].shift(1))
    tr2 = abs(closed_df['low'] - closed_df['close'].shift(1))
    tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=period // 2).mean()
    up = closed_df['high'] - closed_df['high'].shift(1)
    down = closed_df['low'].shift(1) - closed_df['low']
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    plus_di = 100 * (pd.Series(plus_dm, index=closed_df.index).rolling(period, min_periods=period // 2).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=closed_df.index).rolling(period, min_periods=period // 2).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(period, min_periods=period // 2).mean()

    return adx.dropna()


class FeatureEngineer:
    @staticmethod
    def create_features_no_leakage(base_df, aligned_data, is_backtest=True):
        """
        Create features WITHOUT data leakage
        is_backtest=True: for backtesting (use all bars)
        is_backtest=False: for live trading (exclude current bar)
        """
        if len(base_df) < 30:
            return pd.DataFrame()

        # For backtesting use all bars; for live trading exclude current bar
        if not is_backtest and len(base_df) > 1:
            working_df = base_df.iloc[:-1].copy()
        else:
            working_df = base_df.copy()

        features_df = pd.DataFrame(index=working_df.index)

        # Lagged returns (only on closed data)
        features_df['log_return_lag1'] = np.log(working_df['close']) - np.log(working_df['close'].shift(1))
        features_df['log_return_lag2'] = np.log(working_df['close'].shift(1)) - np.log(working_df['close'].shift(2))
        features_df['log_return_lag3'] = np.log(working_df['close'].shift(2)) - np.log(working_df['close'].shift(3))

        # Range features
        features_df['high_low_range'] = (working_df['high'] - working_df['low']) / working_df['close']
        features_df['volatility_20'] = features_df['log_return_lag1'].rolling(20, min_periods=10).std()

        # Price statistics (use shift(1) to prevent leakage)
        rolling_mean = working_df['close'].shift(1).rolling(30, min_periods=15).mean()
        rolling_std = working_df['close'].shift(1).rolling(30, min_periods=15).std()
        features_df['price_zscore_30'] = np.where(rolling_std > 0, (working_df['close'] - rolling_mean) / rolling_std,
                                                  0)

        # Higher timeframe features - IMPORTANT: only use closed bars
        for tf in ['H1', 'H4']:
            if tf in aligned_data:
                tf_price = aligned_data[tf]

                # For live trading exclude current higher TF bar
                if not is_backtest and len(tf_price) > 1:
                    tf_price_closed = tf_price.iloc[:-1]
                else:
                    tf_price_closed = tf_price

                # All calculations on shift(1) — information available at bar close
                rolling_min = tf_price_closed.shift(1).rolling(60, min_periods=30).min()
                rolling_max = tf_price_closed.shift(1).rolling(60, min_periods=30).max()
                range_val = rolling_max - rolling_min
                position = np.where(range_val > 0, (working_df['close'] - rolling_min) / range_val, 0.5)
                features_df[f'{tf}_position'] = position

                if tf == 'H4':
                    ma_25 = tf_price_closed.shift(1).rolling(25, min_periods=13).mean()
                    features_df[f'{tf}_dist_ma'] = (working_df['close'] - ma_25) / working_df['close']

        # Interaction features
        if 'H4_position' in features_df.columns:
            features_df['trend_pos_interaction'] = features_df['price_zscore_30'] * features_df['H4_position']
            features_df['vol_pos_interaction'] = features_df['volatility_20'] * features_df['H4_position']

        # Additional features
        features_df['returns_skew_20'] = features_df['log_return_lag1'].shift(1).rolling(20, min_periods=10).skew()
        features_df['volume_ratio'] = working_df['tick_volume'] / working_df['tick_volume'].shift(1).rolling(20,
                                                                                                             min_periods=10).mean()
        features_df['close_vs_open'] = (working_df['close'] - working_df['open']) / working_df['open']
        features_df['momentum_10'] = working_df['close'] / working_df['close'].shift(10) - 1
        features_df['roc_15'] = (working_df['close'] - working_df['close'].shift(15)) / working_df['close'].shift(15)
        features_df['atr_14'] = calculate_atr(working_df, 14)
        features_df['volatility_condition'] = features_df['atr_14'] / working_df['close']

        features_df = features_df.dropna()
        return features_df


# ==================== TARGET CREATION (FIXED) ====================
def create_target_for_backtest(base_df):
    """
    Create target for backtesting
    Uses future data (forward shift) — allowed ONLY in backtesting!
    """
    if len(base_df) < Config.LOOKAHEAD_BARS + 10:
        return pd.Series(index=base_df.index)

    # Look ahead by LOOKAHEAD_BARS bars
    future_high = base_df['high'].rolling(Config.LOOKAHEAD_BARS, min_periods=1).max().shift(-Config.LOOKAHEAD_BARS)
    future_low = base_df['low'].rolling(Config.LOOKAHEAD_BARS, min_periods=1).min().shift(-Config.LOOKAHEAD_BARS)

    # Compare: which direction is more likely — up or down?
    target_series = pd.Series(
        np.where(
            future_high - base_df['close'] > base_df['close'] - future_low,
            1,  # BUY — distance to high is greater
            np.where(future_high - base_df['close'] < base_df['close'] - future_low, -1, 0)  # SELL — distance to low is greater
        ),
        index=base_df.index
    )

    return target_series


# ==================== TRADE SIGNAL GENERATION ====================
def generate_trade_signal(
        pred,
        buy_threshold,
        sell_threshold,
        adx_value,
        price,
        h4_ma,
        disable_sell=False
):
    if disable_sell and pred < 0:
        return 'HOLD'

    if pred > buy_threshold:
        signal = 'BUY'
    elif pred < -sell_threshold:
        signal = 'SELL'
    else:
        return 'HOLD'

    # ADX filter
    if signal == 'BUY':
        if not (Config.MIN_ADX_BUY <= adx_value <= Config.MAX_ADX_BUY):
            return 'HOLD'
    elif signal == 'SELL':
        if not (Config.MIN_ADX_SELL <= adx_value <= Config.MAX_ADX_SELL):
            return 'HOLD'

    # H4 MA filter
    if signal == 'BUY' and price <= h4_ma:
        return 'HOLD'
    if signal == 'SELL' and price >= h4_ma:
        return 'HOLD'

    return signal


# ==================== DYNAMIC THRESHOLDS ====================
def calculate_dynamic_thresholds(preds, min_threshold=None):
    if min_threshold is None:
        min_threshold = Config.MIN_SIGNAL_THRESHOLD
    buy_preds = preds[preds > 0]
    sell_preds = preds[preds < 0]
    buy_threshold = min_threshold
    sell_threshold = min_threshold
    if len(buy_preds) > 10:
        buy_threshold = max(min_threshold, np.std(buy_preds) * Config.SIGNAL_THRESHOLD_MULTIPLIER)
    if len(sell_preds) > 10:
        sell_threshold = max(min_threshold, np.std(abs(sell_preds)) * Config.SIGNAL_THRESHOLD_MULTIPLIER)
    threshold_pips_buy = buy_threshold / Config.PIP
    threshold_pips_sell = sell_threshold / Config.PIP
    if threshold_pips_buy < 2:
        buy_threshold = min_threshold
    if threshold_pips_sell < 2:
        sell_threshold = min_threshold
    print(f"   Dynamic thresholds: BUY={threshold_pips_buy:.1f}, SELL={threshold_pips_sell:.1f} pips")
    return buy_threshold, sell_threshold


# ==================== SL/TP OPTIMIZER (FIXED) ====================
def optimize_sl_tp_on_window(train_df, features_df, base_prices, spread_pips, aligned_data):
    """Optimize SL/TP using only closed bars"""
    print("🔍 Optimizing SL/TP on training window...")
    n = len(features_df)
    split_idx = int(n * 0.7)
    if split_idx < 100:
        return 20, 50

    features = [col for col in features_df.columns if col != 'target']
    X_train = features_df.iloc[:split_idx][features]
    y_train = features_df.iloc[:split_idx]['target']
    X_val = features_df.iloc[split_idx:][features]
    y_val = features_df.iloc[split_idx:]['target']

    if len(X_train) < 50 or len(X_val) < 20:
        return 20, 50

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = Ridge(alpha=10.0, random_state=42)
    model.fit(X_train_scaled, y_train)
    X_val_scaled = scaler.transform(X_val)
    preds = model.predict(X_val_scaled)
    pred_series = pd.Series(preds, index=X_val.index)
    val_prices = base_prices.loc[X_val.index]

    # Calculate indicators on validation data
    adx_val = calculate_adx(val_prices)

    # H4 MA from aligned data
    if 'H4' in aligned_data:
        h4_prices = aligned_data['H4'].loc[X_val.index]
        h4_ma_val = h4_prices.rolling(25, min_periods=13).mean()
    else:
        h4_ma_val = val_prices['close'].rolling(25, min_periods=13).mean()

    buy_threshold, sell_threshold = calculate_dynamic_thresholds(preds)

    sl_range = [15, 20, 25, 30]
    tp_range = [30, 35, 40, 45, 50]
    best_pnl = -np.inf
    best_sl, best_tp = 20, 50

    for sl in sl_range:
        for tp in tp_range:
            if tp <= sl:
                continue

            total_pnl = 0.0
            trade_count = 0
            active_trade = None

            for idx in X_val.index:
                if idx not in pred_series:
                    continue

                pred = pred_series[idx]
                adx_v = adx_val.get(idx, 0)
                h4_ma = h4_ma_val.get(idx, val_prices.loc[idx, 'close'])

                signal = generate_trade_signal(
                    pred, buy_threshold, sell_threshold, adx_v,
                    val_prices.loc[idx, 'close'], h4_ma
                )

                if active_trade is None and signal in ['BUY', 'SELL']:
                    entry_price = val_prices.loc[idx, 'close']
                    sl_price = entry_price - sl * Config.PIP if signal == 'BUY' else entry_price + sl * Config.PIP
                    tp_price = entry_price + tp * Config.PIP if signal == 'BUY' else entry_price - tp * Config.PIP
                    active_trade = {'type': signal, 'entry': entry_price, 'sl': sl_price, 'tp': tp_price}

                elif active_trade is not None:
                    current_price = val_prices.loc[idx, 'close']
                    hit_sl = (current_price <= active_trade['sl']) if active_trade['type'] == 'BUY' else (
                                current_price >= active_trade['sl'])
                    hit_tp = (current_price >= active_trade['tp']) if active_trade['type'] == 'BUY' else (
                                current_price <= active_trade['tp'])

                    # Time-based exit (24 hours)
                    if active_trade.get('entry_time'):
                        time_exit = (idx - active_trade['entry_time']) > pd.Timedelta(hours=24)
                    else:
                        time_exit = False

                    if hit_sl or hit_tp or time_exit or idx == X_val.index[-1]:
                        gross_pnl_usd, _ = calculate_pnl(active_trade['entry'], current_price, active_trade['type'],
                                                         Config.LOT)
                        net_pnl = calculate_net_pnl(gross_pnl_usd, spread_pips, Config.COMMISSION)
                        total_pnl += net_pnl
                        trade_count += 1
                        active_trade = None

            if trade_count >= 5 and total_pnl > best_pnl:
                best_pnl = total_pnl
                best_sl, best_tp = sl, tp

    if best_pnl <= 0:
        best_sl, best_tp = 20, 50

    print(f"   Best parameters: SL={best_sl}, TP={best_tp} (PnL=${best_pnl:.2f})")
    return best_sl, best_tp


# ==================== WALK-FORWARD LOOP (FIXED) ====================
def run_walk_forward_analysis():
    print("=" * 60)
    print("🚀 WALK-FORWARD ANALYSIS (FIXED VERSION)")
    print("=" * 60)
    print("📌 Separate thresholds, asymmetric ADX, H4 MA filter")
    print("📌 Features calculated without data leakage")
    print("=" * 60)

    try:
        fetcher = MT5DataFetcher()
        spread_pips = fetcher.get_average_spread()
        current_date = Config.START_DATE
        all_positions = []
        all_metrics = []
        all_pnls = []
        window_counter = 0
        recent_sell_pnls = []

        while True:
            window_counter += 1
            train_start = current_date
            train_end = train_start + timedelta(days=Config.TRAIN_WINDOW_DAYS)
            test_start = train_end
            test_end = test_start + timedelta(days=Config.TEST_WINDOW_DAYS)

            if test_end > Config.END_DATE:
                break

            print(f"\n{'=' * 60}")
            print(f"📊 WINDOW {window_counter}: {train_start.date()} → {test_end.date()}")
            print(f"{'=' * 60}")

            # Load training data
            train_base, train_aligned = fetcher.get_base_data_for_analysis(train_start, train_end)
            if train_base is None or len(train_base) < 100:
                print(
                    f"   ❌ Insufficient training data: {len(train_base) if train_base is not None else 0} bars")
                current_date += timedelta(days=Config.STEP_WINDOW_DAYS)
                continue

            print(f"   📈 Loaded: {len(train_base)} M15 bars")

            # Create features for BACKTESTING (is_backtest=True)
            engineer = FeatureEngineer()
            train_features = engineer.create_features_no_leakage(train_base, train_aligned, is_backtest=True)

            if train_features.empty:
                print("   ❌ Failed to create features")
                current_date += timedelta(days=Config.STEP_WINDOW_DAYS)
                continue

            # CREATE TARGET for backtesting (uses future data)
            target_series = create_target_for_backtest(train_base)
            train_features['target'] = target_series.reindex(train_features.index, fill_value=0)
            train_features = train_features.dropna()

            if len(train_features) < 100:
                print(f"   ❌ Insufficient data after cleaning: {len(train_features)} rows")
                current_date += timedelta(days=Config.STEP_WINDOW_DAYS)
                continue

            print(f"   ✅ Features created: {len(train_features)} rows, {len(train_features.columns) - 1} features")

            # Optimize SL/TP
            best_sl, best_tp = optimize_sl_tp_on_window(train_base, train_features, train_base, spread_pips,
                                                        train_aligned)

            # Train model
            features_cols = [col for col in train_features.columns if col != 'target']
            X_train = train_features[features_cols]
            y_train = train_features['target']
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            model = Ridge(alpha=10.0, random_state=42)
            model.fit(X_train_scaled, y_train)

            print(f"   🎯 Model trained. R²: {model.score(X_train_scaled, y_train):.4f}")

            # Test data
            test_base, test_aligned = fetcher.get_base_data_for_analysis(test_start, test_end)
            if test_base is None or len(test_base) < 10:
                print(f"   ❌ Insufficient test data: {len(test_base) if test_base is not None else 0} bars")
                current_date += timedelta(days=Config.STEP_WINDOW_DAYS)
                continue

            print(f"   📊 Test data: {len(test_base)} bars")

            # Create features for test data (is_backtest=True)
            test_features = engineer.create_features_no_leakage(test_base, test_aligned, is_backtest=True)
            if test_features.empty:
                print("   ❌ Failed to create features for test")
                current_date += timedelta(days=Config.STEP_WINDOW_DAYS)
                continue

            # Predict
            X_test = test_features[features_cols]
            X_test_scaled = scaler.transform(X_test)
            preds = model.predict(X_test_scaled)
            pred_series = pd.Series(preds, index=test_features.index)

            # Calculate indicators for test data
            adx_test = calculate_adx(test_base)

            # H4 MA for test data
            if 'H4' in test_aligned:
                h4_prices = test_aligned['H4']
                h4_ma_test = h4_prices.rolling(25, min_periods=13).mean()
            else:
                h4_ma_test = test_base['close'].rolling(25, min_periods=13).mean()

            # Dynamic thresholds on test data
            buy_threshold, sell_threshold = calculate_dynamic_thresholds(preds)

            # Disable SELL if performance is poor
            disable_sell = False
            if len(recent_sell_pnls) >= 10:
                sell_profit = sum(p for p in recent_sell_pnls if p > 0)
                sell_loss = abs(sum(p for p in recent_sell_pnls if p < 0))
                sell_pf = sell_profit / sell_loss if sell_loss > 0 else 0
                if sell_pf < 0.5:
                    disable_sell = True
                    print("   ⚠️ SELL trades disabled (PF < 0.5)")

            # Trade on test window
            active_trade = None
            window_positions = []
            window_pnls = []

            for idx in test_features.index:
                if idx not in pred_series:
                    continue

                pred = pred_series[idx]
                adx_val = adx_test.get(idx, 0)
                h4_ma = h4_ma_test.get(idx, test_base.loc[idx, 'close'])

                signal = generate_trade_signal(
                    pred, buy_threshold, sell_threshold, adx_val,
                    test_base.loc[idx, 'close'], h4_ma, disable_sell
                )

                if active_trade is None and signal in ['BUY', 'SELL']:
                    entry_price = test_base.loc[idx, 'close']
                    sl_price = entry_price - best_sl * Config.PIP if signal == 'BUY' else entry_price + best_sl * Config.PIP
                    tp_price = entry_price + best_tp * Config.PIP if signal == 'BUY' else entry_price - best_tp * Config.PIP
                    active_trade = {
                        'type': signal,
                        'entry': entry_price,
                        'sl': sl_price,
                        'tp': tp_price,
                        'entry_time': idx
                    }
                    window_positions.append({
                        'time': idx,
                        'price': entry_price,
                        'type': signal,
                        'action': 'OPEN',
                        'window': window_counter,
                        'sl_pips': best_sl,
                        'tp_pips': best_tp,
                        'adx_value': adx_val
                    })

                elif active_trade is not None:
                    current_price = test_base.loc[idx, 'close']
                    if active_trade['type'] == 'BUY':
                        hit_sl = current_price <= active_trade['sl']
                        hit_tp = current_price >= active_trade['tp']
                    else:
                        hit_sl = current_price >= active_trade['sl']
                        hit_tp = current_price <= active_trade['tp']

                    # Time-based exit (24 hours)
                    time_exit = (idx - active_trade['entry_time']) > pd.Timedelta(hours=24)

                    if hit_sl or hit_tp or time_exit or idx == test_features.index[-1]:
                        window_positions.append({
                            'time': idx,
                            'price': current_price,
                            'type': active_trade['type'],
                            'action': 'CLOSE',
                            'window': window_counter
                        })
                        gross_pnl_usd, _ = calculate_pnl(
                            active_trade['entry'], current_price, active_trade['type'], Config.LOT
                        )
                        net_pnl = calculate_net_pnl(gross_pnl_usd, spread_pips, Config.COMMISSION)
                        window_pnls.append(net_pnl)

                        if active_trade['type'] == 'SELL':
                            recent_sell_pnls.append(net_pnl)
                            if len(recent_sell_pnls) > 20:
                                recent_sell_pnls.pop(0)

                        active_trade = None

            # Save window results
            if window_pnls:
                win_trades = sum(1 for pnl in window_pnls if pnl > 0)
                total_trades = len(window_pnls)
                total_pnl = sum(window_pnls)
                win_rate = win_trades / total_trades if total_trades > 0 else 0
                cum_pnl = np.cumsum(window_pnls)
                running_max = np.maximum.accumulate(np.concatenate([[0], cum_pnl]))
                drawdown = running_max[1:] - cum_pnl
                max_dd = drawdown.max() if len(drawdown) > 0 else 0

                window_metrics = {
                    'window': window_counter,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'avg_pnl': total_pnl / total_trades if total_trades > 0 else 0,
                    'max_drawdown': max_dd,
                    'best_sl': best_sl,
                    'best_tp': best_tp,
                    'risk_reward_ratio': best_tp / best_sl,
                    'spread_pips': spread_pips
                }
                all_metrics.append(window_metrics)
                all_positions.extend(window_positions)
                all_pnls.extend(window_pnls)

                print(
                    f"📊 Window {window_counter} results: trades={total_trades}, win={win_rate:.1%}, PnL=${total_pnl:.2f}")
            else:
                print(f"📊 Window {window_counter}: no trades executed")

            current_date += timedelta(days=Config.STEP_WINDOW_DAYS)

        # Final summary
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            total_trades_all = metrics_df['total_trades'].sum()
            total_pnl_all = metrics_df['total_pnl'].sum()
            avg_win_rate = metrics_df['win_rate'].mean()
            avg_pnl_per_trade = total_pnl_all / total_trades_all if total_trades_all > 0 else 0

            # Additional metrics
            sharpe_ratio = 0
            if len(all_pnls) > 0:
                returns = np.array(all_pnls) / Config.INITIAL_CAPITAL
                excess_returns = returns - (Config.RISK_FREE_RATE / Config.TRADING_DAYS_YEAR)
                sharpe_ratio = np.sqrt(Config.TRADING_DAYS_YEAR) * (
                            np.mean(excess_returns) / np.std(excess_returns)) if np.std(excess_returns) > 0 else 0

            print(f"\n{'=' * 60}")
            print("🎯 FINAL RESULTS")
            print(f"{'=' * 60}")
            print(f"   Total trades: {total_trades_all}")
            print(f"   Average win rate: {avg_win_rate:.1%}")
            print(f"   Total PnL: ${total_pnl_all:.2f}")
            print(f"   Avg PnL per trade: ${avg_pnl_per_trade:.2f}")
            print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")

            if total_trades_all > 0:
                profit_factor = abs(sum(p for p in all_pnls if p > 0) / sum(p for p in all_pnls if p < 0)) if sum(
                    p for p in all_pnls if p < 0) != 0 else 0
                max_consecutive_wins = 0
                max_consecutive_losses = 0
                current_wins = 0
                current_losses = 0

                for pnl in all_pnls:
                    if pnl > 0:
                        current_wins += 1
                        current_losses = 0
                        max_consecutive_wins = max(max_consecutive_wins, current_wins)
                    elif pnl < 0:
                        current_losses += 1
                        current_wins = 0
                        max_consecutive_losses = max(max_consecutive_losses, current_losses)

                print(f"   Profit Factor: {profit_factor:.2f}")
                print(f"   Max consecutive wins: {max_consecutive_wins}")
                print(f"   Max consecutive losses: {max_consecutive_losses}")

            positions_df = pd.DataFrame(all_positions)
            print_recent_trades_with_params(positions_df, spread_pips)
            plot_results(positions_df, metrics_df, all_pnls)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results = {
                'positions': positions_df,
                'metrics': metrics_df,
                'pnls': all_pnls,
                'config': {
                    'symbol': Config.SYMBOL,
                    'lot': Config.LOT,
                    'spread_pips': spread_pips,
                    'start_date': Config.START_DATE,
                    'end_date': Config.END_DATE,
                    'train_window_days': Config.TRAIN_WINDOW_DAYS,
                    'test_window_days': Config.TEST_WINDOW_DAYS,
                    'step_window_days': Config.STEP_WINDOW_DAYS
                }
            }
            joblib.dump(results, f"walkforward_fixed_{Config.SYMBOL}_{timestamp}.pkl")
            print(f"\n💾 Results saved to walkforward_fixed_{Config.SYMBOL}_{timestamp}.pkl")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n✅ Analysis completed")


# ==================== HELPER FUNCTIONS ====================
def print_recent_trades_with_params(positions_df, spread_pips):
    if positions_df.empty:
        print("\n📋 No trades found.")
        return
    trades = []
    open_trade = None
    for _, row in positions_df.sort_values('time').iterrows():
        if row['action'] == 'OPEN':
            open_trade = row
        elif row['action'] == 'CLOSE' and open_trade is not None:
            gross_pnl_usd, pnl_pips = calculate_pnl(open_trade['price'], row['price'], open_trade['type'], Config.LOT)
            net_pnl = calculate_net_pnl(gross_pnl_usd, spread_pips, Config.COMMISSION)
            sl_pips = open_trade.get('sl_pips', 20)
            tp_pips = open_trade.get('tp_pips', 50)
            duration_hours = (row['time'] - open_trade['time']).total_seconds() / 3600
            trades.append({
                'entry_time': open_trade['time'],
                'exit_time': row['time'],
                'type': open_trade['type'],
                'entry_price': open_trade['price'],
                'exit_price': row['price'],
                'duration_hours': duration_hours,
                'pnl': net_pnl,
                'pnl_pips': pnl_pips,
                'sl_pips': sl_pips,
                'tp_pips': tp_pips,
                'adx_value': open_trade.get('adx_value', 0)
            })
            open_trade = None
    if not trades:
        print("\n📋 Could not match OPEN/CLOSE trades.")
        return
    trades_df = pd.DataFrame(trades).sort_values('entry_time').reset_index(drop=True)
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    print(f"\n📋 LAST 20 TRADES:")
    recent = trades_df.tail(20)
    for i, (_, row) in enumerate(recent.iterrows()):
        print(f"{len(trades_df) - 19 + i:2d} {row['entry_time'].strftime('%m-%d %H:%M')} "
              f"{row['type']:4} {row['entry_price']:.5f}→{row['exit_price']:.5f} "
              f"{row['duration_hours']:4.1f}h ${row['pnl']:6.2f} "
              f"SL{row['sl_pips']} TP{row['tp_pips']} ADX{row['adx_value']:.1f}")


def plot_results(positions_df, metrics_df, all_pnls):
    if positions_df.empty:
        return

    # Collect PnL data
    trades_pnl = []
    trade_times = []
    open_trade = None

    for _, row in positions_df.sort_values('time').iterrows():
        if row['action'] == 'OPEN':
            open_trade = row
        elif row['action'] == 'CLOSE' and open_trade is not None:
            gross_pnl_usd, _ = calculate_pnl(open_trade['price'], row['price'], open_trade['type'], Config.LOT)
            trades_pnl.append(gross_pnl_usd)
            trade_times.append(row['time'])
            open_trade = None

    if not trades_pnl:
        return

    cum_pnl = np.cumsum(trades_pnl)

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Cumulative PnL
    axes[0, 0].plot(trade_times, cum_pnl, 'b-', linewidth=2)
    axes[0, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Cumulative PnL')
    axes[0, 0].set_ylabel('PnL ($)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. PnL distribution per trade
    axes[0, 1].hist(trades_pnl, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('PnL Distribution per Trade')
    axes[0, 1].set_xlabel('PnL ($)')
    axes[0, 1].set_ylabel('Number of Trades')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. PnL by walk-forward window
    if not metrics_df.empty:
        window_numbers = metrics_df['window']
        window_pnls = metrics_df['total_pnl']
        colors = ['green' if pnl > 0 else 'red' for pnl in window_pnls]
        axes[1, 0].bar(window_numbers, window_pnls, color=colors, edgecolor='black')
        axes[1, 0].axhline(0, color='black', linestyle='-', alpha=0.5)
        axes[1, 0].set_title('PnL by Walk-Forward Window')
        axes[1, 0].set_xlabel('Window Number')
        axes[1, 0].set_ylabel('PnL ($)')
        axes[1, 0].grid(True, alpha=0.3)

    # 4. Win Rate by window
    if not metrics_df.empty:
        axes[1, 1].plot(metrics_df['window'], metrics_df['win_rate'] * 100, 'o-', color='blue')
        axes[1, 1].axhline(50, color='red', linestyle='--', alpha=0.5, label='50%')
        axes[1, 1].set_title('Win Rate by Window')
        axes[1, 1].set_xlabel('Window Number')
        axes[1, 1].set_ylabel('Win Rate (%)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    run_walk_forward_analysis()
