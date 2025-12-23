"""
ARPT — Adaptive Realistic Profitability Trader  
Author: Vladimir Korneev  
Date: 2023-12-25  

This script implements a walk-forward backtest for a level-based trading strategy on EURUSD H4.  
It uses significant price levels (support/resistance) as engineered features for a LightGBM classifier  
that predicts directional moves >=80 pips over 3 bars. The model incorporates weekly online learning  
using actual trade PnL outcomes. Despite advanced feature engineering and adaptive retraining,  
the strategy fails to outperform random chance, highlighting the limitations of static level-based signals  
in dynamic markets.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# ==================== CONFIGURATION ====================
MIN_PROFIT_PIPS = 80  # Increased threshold for H4
LOT_SIZE = 0.1
SPREAD_PIPS = 2
INITIAL_BALANCE = 10000

# Level-based parameters
LEVEL_WINDOW_DAYS = 300
LEVEL_TOUCH_THRESHOLD = 0.0010  # 10 pips
N_LEVELS_TO_USE = 10
LEVEL_WINDOW_PIPS = 50
FEATURE_WINDOW = 64

SEQ_LENGTH = 64
HORIZON = 3  # 3 H4 bars = 12 hours

TRAIN_DAYS = 60
TEST_DAYS = 20
RETRAIN_EVERY_DAYS = TEST_DAYS

MIN_CONFIDENCE = 0.55
MIN_TRADE_QUALITY = 0.60

if not os.path.exists('outputs_arpt'):
    os.makedirs('outputs_arpt')


# ==================== LEVEL DETECTION ====================
def detect_levels(df, window_days=300, touch_threshold=0.0010):
    """
    Detect support/resistance levels based on closing prices over past N days.
    A level = price rounded to 5 decimals with ≥2 occurrences.
    """
    print(f"[LEVELS] Detecting levels using last {window_days} days...")
    levels_df = df.copy().tail(int(window_days * 6))
    levels_df['rounded_close'] = levels_df['close'].round(5)
    level_counts = levels_df['rounded_close'].value_counts()
    significant_levels = level_counts[level_counts >= 2].index.values
    significant_levels.sort()
    print(f"[LEVELS] Found {len(significant_levels)} significant levels.")
    return significant_levels


# ==================== LEVEL-BASED FEATURE ENGINEERING ====================
def create_level_features(df, levels, n_levels_to_use=N_LEVELS_TO_USE, window=FEATURE_WINDOW):
    """
    Create normalized features based on proximity to historical price levels.
    """
    print(f"[LEVEL FEATURES] Creating improved features based on {n_levels_to_use} nearest levels...")
    features = pd.DataFrame(index=df.index)

    for i, (idx, row) in enumerate(df.iterrows()):
        current_price = row['close']
        distances = np.abs(levels - current_price)
        nearest_indices = np.argsort(distances)[:n_levels_to_use]
        nearest_levels = levels[nearest_indices]
        nearest_distances = distances[nearest_indices]

        for j, (level, dist) in enumerate(zip(nearest_levels, nearest_distances)):
            features.loc[idx, f'level_{j + 1}_dist'] = dist

        is_near_any_level = any(dist <= LEVEL_TOUCH_THRESHOLD for dist in nearest_distances)
        features.loc[idx, 'is_price_near_level'] = int(is_near_any_level)

        if len(nearest_levels) >= 2:
            lower_level = nearest_levels[0]
            upper_level = nearest_levels[1]
            pos_score = (current_price - lower_level) / (upper_level - lower_level) if upper_level != lower_level else 0.5
        else:
            pos_score = 0.5
        features.loc[idx, 'level_position_score'] = pos_score

        level_count_in_window = np.sum(distances <= LEVEL_WINDOW_PIPS / 10000)
        features.loc[idx, 'level_density'] = level_count_in_window

        if i > 0:
            prev_price = df.iloc[i - 1]['close']
            closest_level = nearest_levels[0]
            prev_dist = abs(prev_price - closest_level)
            curr_dist = abs(current_price - closest_level)
            approach_speed = (prev_dist - curr_dist) / (prev_dist + 1e-8)
        else:
            approach_speed = 0
        features.loc[idx, 'price_approaching_level'] = approach_speed

    features = features.fillna(0)

    # Normalize via rolling z-score
    for col in features.columns:
        features[f'{col}_zscore'] = features[col].rolling(window=window, min_periods=1).apply(
            lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8), raw=False
        )

    features = features[[col for col in features.columns if col.endswith('_zscore')]]
    features = features.fillna(0).replace([np.inf, -np.inf], 0)
    print(f"[LEVEL FEATURES] Created {len(features.columns)} improved level-based features")
    return features


# ==================== TARGET ENGINEERING ====================
def create_targets(df, min_pips=80, horizon=3):
    """
    Binary target: 1 if |price move| >= min_pips in `horizon` bars, else 0.
    Direction label: 1 = up, -1 = down, 0 = no move.
    """
    print(f"[TARGET] Creating binary target (|movement| >= {min_pips} pips in {horizon} bars)...")
    prices = df['close'].values
    targets_move = []
    targets_dir = []

    for i in range(len(prices) - horizon):
        current = prices[i]
        future = prices[i + horizon]
        move_pips = (future - current) * 10000
        if abs(move_pips) >= min_pips:
            targets_move.append(1)
            targets_dir.append(1 if move_pips > 0 else -1)
        else:
            targets_move.append(0)
            targets_dir.append(0)

    targets_move.extend([0] * horizon)
    targets_dir.extend([0] * horizon)

    move_series = pd.Series(targets_move, index=df.index)
    dir_series = pd.Series(targets_dir, index=df.index)
    print(f"[TARGET] Movement distribution: {move_series.value_counts().to_dict()}")
    return move_series, dir_series


# ==================== WALK-FORWARD VALIDATOR ====================
class WalkForwardValidator:
    def __init__(self, train_days=60, test_days=20, retrain_every=20):
        self.train_days = train_days
        self.test_days = test_days
        self.retrain_every = retrain_every

    def generate_folds(self, df, start_date=None, end_date=None):
        if start_date is None:
            start_date = df.index[0]
        if end_date is None:
            end_date = df.index[-1]

        folds = []
        current = start_date
        while True:
            train_end = current + timedelta(days=self.train_days)
            test_end = train_end + timedelta(days=self.test_days)
            if test_end > end_date:
                break
            folds.append({
                'train_start': current,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end,
                'fold_num': len(folds) + 1
            })
            current += timedelta(days=self.retrain_every)
        print(f"[WALK-FORWARD] Generated {len(folds)} non-overlapping folds")
        return folds


# ==================== ADAPTIVE SIGNAL MODEL ====================
class AdaptiveSignalModel:
    def __init__(self, min_profit_pips=80, horizon=3):
        self.min_profit_pips = min_profit_pips
        self.horizon = horizon
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        )
        self.is_trained = False
        self.feature_columns = None

    def train(self, X_train, y_train):
        print(f"[TRAIN] Training on {len(X_train)} samples...")
        self.feature_columns = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def partial_fit(self, X_new, y_new):
        if not self.is_trained:
            print("[PARTIAL_FIT] Model not trained yet, calling full train.")
            self.train(X_new, y_new)
            return
        print(f"[PARTIAL_FIT] Updating model with {len(X_new)} new samples...")
        self.model.fit(X_new, y_new, init_model=self.model)

    def predict_profitability(self, X):
        if not self.is_trained:
            return np.zeros(len(X)), np.zeros(len(X))
        pred_proba = self.model.predict_proba(X)
        classes = self.model.classes_
        profit_class_idx = np.where(classes == 1)[0]
        prob_profit = pred_proba[:, profit_class_idx[0]] if len(profit_class_idx) > 0 else np.zeros(len(X))
        pred_class = self.model.predict(X)
        return pred_class, prob_profit


# ==================== TRADING STRATEGY ====================
class AdaptiveTradingStrategy:
    def __init__(self, initial_balance=10000):
        self.balance = initial_balance
        self.equity = [initial_balance]
        self.trades = []
        self.active_trades = []
        self.trade_history = []

    def execute_trade(self, signal_type, entry_price, current_time, tp_pips, sl_pips):
        if signal_type == 'buy':
            tp = entry_price + tp_pips * 0.0001
            sl = entry_price - sl_pips * 0.0001
        else:
            tp = entry_price - tp_pips * 0.0001
            sl = entry_price + sl_pips * 0.0001

        trade = {
            'entry_time': current_time,
            'entry_price': entry_price,
            'position': signal_type,
            'tp': tp,
            'sl': sl,
            'tp_pips': tp_pips,
            'sl_pips': sl_pips,
            'status': 'active'
        }
        self.active_trades.append(trade)

    def check_trades(self, current_price, current_time):
        new_active = []
        closed_trades = []
        for trade in self.active_trades:
            pnl = 0
            exit_reason = None
            exit_price = current_price
            duration_bars = 1

            if trade['position'] == 'buy':
                if current_price >= trade['tp']:
                    exit_price = trade['tp']; exit_reason = 'TP'
                    pnl = (exit_price - trade['entry_price']) * 10000 * LOT_SIZE
                elif current_price <= trade['sl']:
                    exit_price = trade['sl']; exit_reason = 'SL'
                    pnl = (exit_price - trade['entry_price']) * 10000 * LOT_SIZE
            else:
                if current_price <= trade['tp']:
                    exit_price = trade['tp']; exit_reason = 'TP'
                    pnl = (trade['entry_price'] - exit_price) * 10000 * LOT_SIZE
                elif current_price >= trade['sl']:
                    exit_price = trade['sl']; exit_reason = 'SL'
                    pnl = (trade['entry_price'] - exit_price) * 10000 * LOT_SIZE

            if exit_reason:
                pnl -= SPREAD_PIPS * 10 * LOT_SIZE
                self.balance += pnl
                closed_trade = {**trade, 'exit_time': current_time, 'exit_price': exit_price,
                                'exit_reason': exit_reason, 'pnl': pnl, 'balance': self.balance,
                                'duration_bars': duration_bars, 'pnl_pips': pnl / (LOT_SIZE * 10)}
                self.trades.append(closed_trade)
                closed_trades.append(closed_trade)
            else:
                new_active.append(trade)
        self.active_trades = new_active
        self.equity.append(self.balance)
        return closed_trades


# ==================== DATA LOADING ====================
def load_data_from_mt5(symbol, timeframe, n_bars=50000):
    print(f"[DATA] Loading {n_bars} bars of {symbol} @ {timeframe}...")
    if not mt5.initialize():
        raise Exception("MT5 init failed")
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    mt5.shutdown()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df['returns'] = df['close'].pct_change()
    print(f"[DATA] Loaded {len(df):,} bars")
    return df


# ==================== MAIN PIPELINE ====================
def main():
    print("=" * 80)
    print("ARPT — IMPROVED LEVEL-BASED FEATURES (H4) & ONLINE LEARNING ON TRADE RESULTS")
    print("=" * 80)

    df = load_data_from_mt5("EURUSD", mt5.TIMEFRAME_H4, n_bars=50000)
    significant_levels = detect_levels(df, window_days=LEVEL_WINDOW_DAYS, touch_threshold=LEVEL_TOUCH_THRESHOLD)
    features = create_level_features(df, significant_levels, n_levels_to_use=N_LEVELS_TO_USE, window=FEATURE_WINDOW)
    y_move, y_dir = create_targets(df, min_pips=MIN_PROFIT_PIPS, horizon=HORIZON)
    y_combined = y_dir.copy()
    y_combined[y_move == 0] = 0

    validator = WalkForwardValidator(train_days=TRAIN_DAYS, test_days=TEST_DAYS, retrain_every=RETRAIN_EVERY_DAYS)
    folds = validator.generate_folds(df)

    strategy = AdaptiveTradingStrategy(initial_balance=INITIAL_BALANCE)
    signal_model = AdaptiveSignalModel(min_profit_pips=MIN_PROFIT_PIPS, horizon=HORIZON)

    for fold in folds:
        train_mask = (df.index >= fold['train_start']) & (df.index < fold['train_end'])
        X_train = features[train_mask]
        y_train = y_combined[train_mask]
        train_valid_mask = y_train != 0
        X_train_filtered = X_train[train_valid_mask]
        y_train_filtered = y_train[train_valid_mask]

        if len(X_train_filtered) < 100:
            continue

        signal_model.train(X_train_filtered, y_train_filtered)

        test_mask = (df.index >= fold['test_start']) & (df.index < fold['test_end'])
        X_test = features[test_mask]
        test_index = features[test_mask].index
        if len(X_test) == 0:
            continue

        directions, confidences = signal_model.predict_profitability(X_test)
        pred_dict = {idx: (dir_val, conf_val) for idx, (dir_val, conf_val) in
                     zip(test_index, zip(directions, confidences))}

        last_retrain_date = None
        for i, idx in enumerate(test_index):
            current_price = df.loc[idx, 'close'] if idx in df.index else None
            if current_price is None:
                continue

            hour = idx.hour
            if hour < 7 or hour > 19:
                continue

            if idx not in pred_dict:
                continue

            pred_dir, net_conf = pred_dict[idx]
            if pred_dir == 1:
                signal_type = 'buy'
            elif pred_dir == -1:
                signal_type = 'sell'
            else:
                continue

            if net_conf < MIN_CONFIDENCE:
                continue

            current_atr = df.loc[idx, 'close'] * 0.0008
            tp_pips = max(min(int(current_atr * 10000 * 2.5), 300), 80)
            sl_pips = max(min(int(current_atr * 10000 * 1.2), 150), 40)

            closed_trades = strategy.check_trades(current_price, idx)
            for trade in closed_trades:
                entry_features = features.loc[trade['entry_time']] if trade['entry_time'] in features.index else None
                if entry_features is not None:
                    trade_result = {
                        'features': entry_features.values,
                        'action': 1 if trade['position'] == 'buy' else -1,
                        'result_profit': 1 if trade['pnl'] > 0 else 0,
                        'pnl_pips': trade['pnl_pips'],
                        'duration': trade['duration_bars']
                    }
                    strategy.trade_history.append(trade_result)

            if len(strategy.active_trades) == 0:
                strategy.execute_trade(signal_type, current_price, idx, tp_pips, sl_pips)
                print(f"[TRADE] {idx} {signal_type.upper()} @ {current_price:.5f} "
                      f"| Conf: {net_conf:.2f} | TP: {tp_pips} SL: {sl_pips}")

            if idx.weekday() == 0 and (last_retrain_date is None or (idx - last_retrain_date).days >= 7):
                if len(strategy.trade_history) >= 10:
                    print(f"[ONLINE LEARNING] Retraining on trade results at {idx}...")
                    hist = strategy.trade_history[-50:]
                    X_hist = pd.DataFrame([h['features'] for h in hist])
                    y_hist_result = pd.Series([h['result_profit'] for h in hist])
                    signal_model.partial_fit(X_hist, y_hist_result)
                    last_retrain_date = idx

    # Close remaining trades
    if strategy.active_trades:
        last_price = df['close'].iloc[-1]
        for trade in strategy.active_trades:
            pnl = (last_price - trade['entry_price']) * 10000 * LOT_SIZE if trade['position'] == 'buy' else \
                  (trade['entry_price'] - last_price) * 10000 * LOT_SIZE
            pnl -= SPREAD_PIPS * 10 * LOT_SIZE
            strategy.balance += pnl

            entry_features = features.loc[trade['entry_time']] if trade['entry_time'] in features.index else None
            if entry_features is not None:
                strategy.trade_history.append({
                    'features': entry_features.values,
                    'action': 1 if trade['position'] == 'buy' else -1,
                    'result_profit': 1 if pnl > 0 else 0,
                    'pnl_pips': pnl / (LOT_SIZE * 10),
                    'duration': 1
                })
            strategy.trades.append({**trade, 'exit_time': df.index[-1], 'exit_price': last_price,
                                    'exit_reason': 'END', 'pnl': pnl, 'balance': strategy.balance,
                                    'duration_bars': 1, 'pnl_pips': pnl / (LOT_SIZE * 10)})

    # Results
    if strategy.trades:
        trades_df = pd.DataFrame(strategy.trades)
        total_trades = len(trades_df)
        win_rate = (trades_df['pnl'] > 0).mean() * 100
        net_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        max_dd = 0
        peak = strategy.equity[0]
        for value in strategy.equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        print(f"\n{'=' * 60}")
        print(f"TOTAL TRADES: {total_trades}")
        print(f"WIN RATE: {win_rate:.1f}%")
        print(f"NET PnL: ${net_pnl:.2f}")
        print(f"AVERAGE PnL: ${avg_pnl:.2f}")
        print(f"RETURN: {(net_pnl / INITIAL_BALANCE) * 100:.2f}%")
        print(f"MAX DRAWDOWN: {max_dd:.2f}%")

        trades_df.to_csv('outputs_arpt/trades_arpt.csv', index=False)

        plt.figure(figsize=(12, 6))
        plt.plot(strategy.equity, label='Equity')
        plt.title(f'ARPT Equity Curve | Trades: {total_trades}, WR: {win_rate:.1f}%')
        plt.grid(True, alpha=0.3)
        plt.savefig('outputs_arpt/equity_arpt.png', dpi=150, bbox_inches='tight')
        plt.close()
    else:
        print("[WARNING] No trades executed!")


if __name__ == "__main__":
    main()
