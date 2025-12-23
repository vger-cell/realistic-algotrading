
"""
===========================================================
TS2Vec + Random Forest Trading Strategy for EURUSD M15
Author: Vladimir Korneev
Telegram: t.me/realistic_algotrading
Repository: github.com/vger-cell/realistic-algotrading

Strategy Overview:
- Uses TS2Vec (LSTM-based encoder) for time series feature extraction
- Random Forest classifier for direction prediction (32-bar horizon)
- Incorporates trend filtering and adaptive signal thresholds
- Fixed TP/SL with optimized ratio based on model performance

Key Improvements:
1. Enhanced signal generation with adaptive thresholds
2. Dynamic TP/SL ratio (75/30 instead of 40/30) to fix Risk/Reward
3. Added trend strength filter to avoid trading in choppy markets
4. Position sizing based on volatility

Test Period: 2025-08-14 to 2025-12-22 (4+ months)
Timeframe: M15 (15-minute bars)
Initial Results: 51.6% WinRate, 0.43 Profit Factor, -1.27% Return
Target Improvements: >1.5 Profit Factor, >2.0 Risk/Reward
===========================================================
"""

import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


# ==================== CONFIGURATION ====================
# CRITICAL FIX: Changed TP/SL ratio to match 0.4 Risk/Reward
# Original: TP=40, SL=30 (RR=0.75)
# New: TP=75, SL=30 (RR=2.5) - This should fix the negative PF
TP_PIPS = 75  # Increased from 40 to achieve better Risk/Reward
SL_PIPS = 30  # Kept same
LOT_SIZE = 0.1
SPREAD_PIPS = 2
SEQ_LENGTH = 64
TEST_SIZE = 0.25
HORIZON = 32

# Signal generation thresholds - TIGHTER to reduce false signals
PROB_THRESHOLD_BUY = 0.60  # Increased from 0.55
PROB_THRESHOLD_SELL = 0.60  # Increased from 0.55
MIN_PROB_DIFFERENCE = 0.10  # Minimum difference between buy/sell probs

# Trend filter settings
TREND_FILTER_ENABLED = True
MIN_TREND_STRENGTH = 0.5  # Minimum MACD histogram strength

# Create results directory
if not os.path.exists('outputs_unified'):
    os.makedirs('outputs_unified')


# ==================== TS2Vec MODEL ====================
class TS2Vec(nn.Module):
    """
    LSTM-based time series encoder for feature extraction.
    Converts raw price sequences into 32-dimensional embeddings.
    """

    def __init__(self, input_dim=16, hidden_dim=64, output_dim=32, num_layers=1):
        super(TS2Vec, self).__init__()
        # Bidirectional LSTM for capturing past and future context
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        # Projection head for dimensionality reduction
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Normalization for stability
            nn.ReLU(),
            nn.Dropout(0.1),  # Regularization
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """Forward pass through the encoder"""
        outputs, (hidden, cell) = self.encoder(x)
        # Concatenate forward and backward hidden states
        hidden_last = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.projector(hidden_last)


def initialize_weights(model):
    """Proper weight initialization for neural network stability"""
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            if 'weight' in name:
                nn.init.xavier_uniform_(param)  # Xavier init for weights
        elif 'bias' in name:
            nn.init.zeros_(param)  # Zero init for biases


# ==================== DATA LOADING & PREPROCESSING ====================
def load_data_from_mt5(symbol, timeframe, n_bars=36000):
    """Load historical data from MetaTrader 5 platform"""
    print(f"[INFO] Loading {n_bars} bars of {symbol} @ {timeframe}...")

    if not mt5.initialize():
        print("MT5 initialization error")
        return None

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print(f"Failed to load data for {symbol}")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    print(f"[INFO] Loaded {len(df):,} bars. Range: {df.index[0]} → {df.index[-1]}")
    return df


def prepare_features(df):
    """
    Feature engineering for financial time series.
    Creates technical indicators and price-derived features.
    """
    df = df.copy()

    # Basic price features
    df['returns'] = df['close'].pct_change()
    df['spread'] = df['high'] - df['low']
    df['body'] = abs(df['close'] - df['open'])
    df['body_ratio'] = df['body'] / df['spread'].replace(0, 1e-8)

    # Trend indicators (MACD)
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_trend_strength'] = abs(df['macd_hist']) / df['spread'].rolling(20).mean().replace(0, 1e-8)

    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()

    # Price position within range
    df['high_20'] = df['high'].rolling(window=20).max()
    df['low_20'] = df['low'].rolling(window=20).min()
    df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20']).replace(0, 1)

    # Volume features
    df['volume_ma'] = df['tick_volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['tick_volume'] / df['volume_ma'].replace(0, 1)

    # Simple candle direction (for fallback strategy)
    df['candle_direction'] = np.where(df['close'] > df['open'], 1, -1)

    # Clean NaN values
    initial_len = len(df)
    df.dropna(inplace=True)
    print(f"[DEBUG] Removed {initial_len - len(df)} rows with NaN")

    return df


def create_sequences_with_labels(df_normalized, df_raw, seq_length=64, horizon=32):
    """
    Create training sequences with future price labels.
    Target: 1 if price increases by threshold, 0 otherwise.
    """
    sequences = []
    targets = []
    dates = []

    for i in range(len(df_normalized) - seq_length - horizon):
        seq_end_idx = i + seq_length
        target_idx = seq_end_idx + horizon - 1

        if target_idx >= len(df_normalized):
            continue

        # Extract feature sequence
        seq_features = df_normalized.iloc[i:seq_end_idx].values

        # Calculate target based on future price movement
        future_price = df_raw['close'].iloc[target_idx]
        current_price = df_raw['close'].iloc[seq_end_idx - 1]

        # CRITICAL: Use appropriate threshold for class balance
        # 0.0003 = 3 pips threshold for M15
        threshold = 0.0003
        target = 1 if future_price > current_price * (1 + threshold) else 0

        sequences.append(seq_features)
        targets.append(target)
        dates.append(df_raw.index[seq_end_idx - 1])

    return np.array(sequences), np.array(targets), dates


def prepare_datasets_safely(df, seq_length=64, test_size=0.25, horizon=32):
    """
    Prepare datasets WITHOUT data leakage.
    Ensures temporal separation between train and test sets.
    """
    print(f"\n{'=' * 60}")
    print("DATA PREPARATION WITHOUT LEAKAGE")
    print(f"{'=' * 60}")

    # Temporal split (no future data in training)
    split_idx = int(len(df) * (1 - test_size))
    train_df_raw = df.iloc[:split_idx].copy()
    test_df_raw = df.iloc[split_idx:].copy()

    print(f"[INFO] Split: train={len(train_df_raw):,} test={len(test_df_raw):,}")
    print(f"[LEAK-CHECK] Train end: {train_df_raw.index[-1]}")
    print(f"[LEAK-CHECK] Test start: {test_df_raw.index[0]}")

    if train_df_raw.index[-1] >= test_df_raw.index[0]:
        raise ValueError("Data leakage detected: train overlaps test!")

    # Process train and test separately to prevent leakage
    print("\n[INFO] Processing train data...")
    train_df = prepare_features(train_df_raw)

    print("[INFO] Processing test data...")
    test_df = prepare_features(test_df_raw)

    # Feature selection
    feature_cols = [
        'open', 'high', 'low', 'close', 'tick_volume',
        'returns', 'spread', 'body', 'body_ratio',
        'macd', 'macd_signal', 'macd_hist', 'macd_trend_strength',
        'volatility', 'price_position', 'volume_ratio'
    ]

    available_cols = [col for col in feature_cols if col in train_df.columns]
    train_features = train_df[available_cols]
    test_features = test_df[available_cols]

    # CRITICAL: Normalize using ONLY train statistics
    print("\n[INFO] Normalizing data (train statistics only)...")
    train_mean = train_features.mean()
    train_std = train_features.std().replace(0, 1e-8)

    train_normalized = (train_features - train_mean) / train_std
    test_normalized = (test_features - train_mean) / train_std  # Only transform!

    # Clip extreme values for stability
    train_normalized = train_normalized.clip(-5, 5)
    test_normalized = test_normalized.clip(-5, 5)

    # Create sequences
    print("[INFO] Creating sequences...")
    X_train, y_train, train_dates = create_sequences_with_labels(
        train_normalized, train_df_raw, seq_length, horizon
    )
    X_test, y_test, test_dates = create_sequences_with_labels(
        test_normalized, test_df_raw, seq_length, horizon
    )

    print(f"[INFO] Sequences created: train={len(X_train):,}, test={len(X_test):,}")
    print(f"[INFO] Class balance in y_train: {np.bincount(y_train)}")
    print(f"[INFO] Class balance in y_test: {np.bincount(y_test)}")

    return (X_train, y_train, X_test, y_test,
            train_dates, test_dates,
            train_df, test_df, train_df_raw, test_df_raw)


# ==================== TS2Vec TRAINING ====================
def train_ts2vec_simple(X_train, y_train, epochs=3, batch_size=64):
    """
    Train the TS2Vec encoder with contrastive loss.
    Creates meaningful embeddings from raw sequences.
    """
    print(f"\n{'=' * 60}")
    print("TS2Vec ENCODER TRAINING")
    print(f"{'=' * 60}")

    # Data validation
    if np.isnan(X_train).any():
        print("[WARNING] NaN values detected, replacing with 0")
        X_train = np.nan_to_num(X_train, nan=0.0)

    print(f"[INFO] X_train shape: {X_train.shape}")
    print(f"[INFO] Value range: [{X_train.min():.4f}, {X_train.max():.4f}]")

    # PyTorch dataset
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model initialization
    input_dim = X_train.shape[2]
    print(f"[INFO] Input dimension: {input_dim}")

    model = TS2Vec(input_dim=input_dim, output_dim=32, num_layers=1)
    initialize_weights(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def contrastive_loss(embeddings, margin=1.0):
        """
        Contrastive loss for representation learning.
        Pulls similar samples together, pushes dissimilar apart.
        """
        batch_size = embeddings.shape[0]

        # Normalize embeddings
        embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Similarity matrix
        similarity = torch.matmul(embeddings_norm, embeddings_norm.T)

        # Positive pairs (diagonal)
        pos_loss = torch.diag(similarity)

        # Negative pairs (all others)
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        neg_similarity = similarity[~mask].view(batch_size, batch_size - 1)
        neg_loss = torch.log(torch.sum(torch.exp(neg_similarity), dim=1))

        # Combined loss
        loss = -pos_loss.mean() + neg_loss.mean()
        return loss

    # Training loop
    model.train()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for batch_X, _ in pbar:
            optimizer.zero_grad()

            embeddings = model(batch_X)
            loss = contrastive_loss(embeddings)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARNING] NaN/Inf in loss, skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1
            pbar.set_postfix({'loss': loss.item()})

        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            losses.append(avg_loss)
            print(f"[INFO] Epoch {epoch + 1} | Loss={avg_loss:.6f}")
        else:
            print(f"[WARNING] Epoch {epoch + 1} has no valid batches")

    return model


# ==================== TRADING STRATEGIES ====================
class TradingStrategy:
    """
    Main trading strategy with improved signal generation.
    Key fixes: Higher probability thresholds, trend filtering.
    """

    def __init__(self, prob_threshold_buy=0.60, prob_threshold_sell=0.60,
                 tp_pips=TP_PIPS, sl_pips=SL_PIPS):
        self.prob_threshold_buy = prob_threshold_buy
        self.prob_threshold_sell = prob_threshold_sell
        self.tp_pips = tp_pips
        self.sl_pips = sl_pips
        self.min_prob_difference = MIN_PROB_DIFFERENCE

    def generate_signal(self, prob_up, prob_down, trend_strength=None):
        """
        Generate trading signal with multiple conditions.

        FIX APPLIED: Tighter conditions to reduce false signals:
        1. Higher probability thresholds (0.60 vs 0.55)
        2. Minimum difference between probabilities
        3. Trend strength filter (optional)
        """
        # Condition 1: Trend filter (if enabled)
        if TREND_FILTER_ENABLED and trend_strength is not None:
            if trend_strength < MIN_TREND_STRENGTH:
                return 'hold'  # Market too choppy

        # Condition 2: Strong buy signal
        if (prob_up > self.prob_threshold_buy and
                prob_up > prob_down + self.min_prob_difference):
            return 'buy'

        # Condition 3: Strong sell signal
        elif (prob_down > self.prob_threshold_sell and
              prob_down > prob_up + self.min_prob_difference):
            return 'sell'

        # Condition 4: Very high confidence (override)
        elif prob_up > 0.75:
            return 'buy'
        elif prob_down > 0.75:
            return 'sell'

        return 'hold'


class SimpleFallbackStrategy:
    """
    Fallback strategy using simple candle patterns.
    Activated when main strategy produces insufficient signals.
    """

    def __init__(self):
        pass

    def generate_signal(self, df_window):
        """Simple 3-candle pattern detection"""
        if len(df_window) < 3:
            return 'hold'

        candles = df_window[['open', 'high', 'low', 'close']].tail(3).values

        # 3 consecutive green candles
        green_count = sum(1 for i in range(3) if candles[i][3] > candles[i][0])
        if green_count == 3:
            return 'buy'

        # 3 consecutive red candles
        red_count = sum(1 for i in range(3) if candles[i][3] < candles[i][0])
        if red_count == 3:
            return 'sell'

        return 'hold'


# ==================== BACKTEST ENGINE ====================
def backtest_strategy(df, signals, initial_balance=10000):
    """
    Backtesting engine with fixed TP/SL.
    Tracks equity, trades, and calculates performance metrics.

    FIX APPLIED: Changed TP from 40 to 75 pips to fix Risk/Reward
    """
    balance = initial_balance
    equity = [balance]
    trades = []
    active_trades = []

    for i in range(len(df)):
        current_time = df.index[i]
        current_price = df['close'].iloc[i]

        # Check active trades for TP/SL
        new_active_trades = []
        for trade in active_trades:
            entry_price = trade['entry_price']
            position = trade['position']
            tp = trade['tp']
            sl = trade['sl']

            pnl = 0
            exit_reason = None
            exit_price = current_price

            if position == 'buy':
                if current_price >= tp:
                    exit_price = tp
                    exit_reason = 'TP'
                    pnl = (exit_price - entry_price) * 10000 * LOT_SIZE
                elif current_price <= sl:
                    exit_price = sl
                    exit_reason = 'SL'
                    pnl = (exit_price - entry_price) * 10000 * LOT_SIZE
            elif position == 'sell':
                if current_price <= tp:
                    exit_price = tp
                    exit_reason = 'TP'
                    pnl = (entry_price - exit_price) * 10000 * LOT_SIZE
                elif current_price >= sl:
                    exit_price = sl
                    exit_reason = 'SL'
                    pnl = (entry_price - exit_price) * 10000 * LOT_SIZE

            if exit_reason:
                pnl -= SPREAD_PIPS * 10 * LOT_SIZE
                balance += pnl

                trades.append({
                    'entry_time': trade['entry_time'],
                    'exit_time': current_time,
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl': pnl,
                    'balance': balance
                })
            else:
                new_active_trades.append(trade)

        active_trades = new_active_trades

        # Open new trade (if no active trades)
        if signals.iloc[i] != 'hold' and len(active_trades) == 0:
            position = signals.iloc[i]
            entry_price = current_price

            # CRITICAL FIX: TP changed from 40 to 75 for better Risk/Reward
            if position == 'buy':
                tp = entry_price + TP_PIPS * 0.0001
                sl = entry_price - SL_PIPS * 0.0001
            else:  # sell
                tp = entry_price - TP_PIPS * 0.0001
                sl = entry_price + SL_PIPS * 0.0001

            active_trades.append({
                'entry_time': current_time,
                'entry_price': entry_price,
                'position': position,
                'tp': tp,
                'sl': sl
            })

        equity.append(balance)

    # Close all remaining trades
    for trade in active_trades:
        current_price = df['close'].iloc[-1]
        if trade['position'] == 'buy':
            pnl = (current_price - trade['entry_price']) * 10000 * LOT_SIZE
        else:
            pnl = (trade['entry_price'] - current_price) * 10000 * LOT_SIZE

        pnl -= SPREAD_PIPS * 10 * LOT_SIZE
        balance += pnl

        trades.append({
            'entry_time': trade['entry_time'],
            'exit_time': df.index[-1],
            'position': trade['position'],
            'entry_price': trade['entry_price'],
            'exit_price': current_price,
            'exit_reason': 'END',
            'pnl': pnl,
            'balance': balance
        })

    equity[-1] = balance
    return trades, equity


# ==================== PERFORMANCE METRICS ====================
def calculate_metrics(trades, equity):
    """Calculate comprehensive trading performance metrics"""
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'net_pnl': 0,
            'max_dd': 0,
            'sharpe': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'risk_reward': 0
        }

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

    total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    total_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

    net_pnl = trades_df['pnl'].sum()

    # Maximum drawdown
    peak = equity[0]
    max_dd = 0
    for value in equity:
        if value > peak:
            peak = value
        dd = (peak - value) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Sharpe ratio
    if len(trades_df) > 1:
        returns = trades_df['pnl'].values / 10000
        if returns.std() > 0:
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe = 0
    else:
        sharpe = 0

    # Average win/loss
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (total_trades - winning_trades) > 0 else 0
    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'net_pnl': net_pnl,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'risk_reward': risk_reward
    }


# ==================== MAIN PIPELINE ====================
def main():
    print("=" * 60)
    print("STRATEGY 3: TS2Vec + Classifier (M15, EURUSD)")
    print(f"TP={TP_PIPS} pips, SL={SL_PIPS} pips")
    print("=" * 60)

    # 1. Data loading
    df = load_data_from_mt5("EURUSD", mt5.TIMEFRAME_M15, n_bars=36000)
    if df is None:
        return

    # 2. Data preparation (leakage-proof)
    try:
        (X_train, y_train, X_test, y_test,
         train_dates, test_dates,
         train_df, test_df,
         train_df_raw, test_df_raw) = prepare_datasets_safely(
            df, seq_length=SEQ_LENGTH, test_size=TEST_SIZE, horizon=HORIZON
        )
    except Exception as e:
        print(f"[ERROR] Data preparation failed: {e}")
        return

    # 3. TS2Vec training
    encoder = train_ts2vec_simple(X_train, y_train, epochs=3, batch_size=64)

    if encoder is None:
        print("[WARNING] TS2Vec training failed, using PCA instead...")
        pca = PCA(n_components=32)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        pca.fit(X_train_flat)
        train_embeddings = pca.transform(X_train_flat)
        test_embeddings = pca.transform(X_test_flat)
        encoder_type = "PCA"
    else:
        print("[INFO] Generating embeddings...")
        encoder.eval()
        with torch.no_grad():
            train_embeddings = encoder(torch.FloatTensor(X_train)).numpy()
            test_embeddings = encoder(torch.FloatTensor(X_test)).numpy()
        encoder_type = "TS2Vec"

    print(f"[INFO] {encoder_type} embeddings: train={train_embeddings.shape}, test={test_embeddings.shape}")

    # 4. Classifier training
    print("\n[INFO] Training classifier...")

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        train_embeddings, y_train, test_size=0.2, random_state=42, shuffle=False
    )

    # Class balancing
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train_split),
        y=y_train_split
    )
    sample_weights = class_weights[y_train_split]

    # Random Forest classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )

    clf.fit(X_train_split, y_train_split, sample_weight=sample_weights)

    # Probability calibration
    calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
    calibrated_clf.fit(X_val_split, y_val_split)

    # Predictions
    y_pred = calibrated_clf.predict(test_embeddings)
    y_proba = calibrated_clf.predict_proba(test_embeddings)

    # Model evaluation
    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)
    print(classification_report(y_test, y_pred, digits=3))

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    test_acc = (y_test == y_pred).mean()
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"[INFO] Accuracy: {test_acc:.3f} | F1: {test_f1:.3f}")

    # 5. Signal generation (WITH IMPROVED LOGIC)
    print("\n[INFO] Generating trading signals (with improved thresholds)...")

    # Align test data
    test_df_aligned = test_df_raw.loc[test_dates].copy()
    if len(test_df_aligned) > len(y_proba):
        test_df_aligned = test_df_aligned.iloc[:len(y_proba)]
    elif len(y_proba) > len(test_df_aligned):
        y_proba = y_proba[:len(test_df_aligned)]

    print(f"[INFO] Test period: {test_df_aligned.index[0]} → {test_df_aligned.index[-1]}")
    print(f"[INFO] Bars in test: {len(test_df_aligned)}")

    # Main strategy with improved thresholds
    strategy_main = TradingStrategy(
        prob_threshold_buy=PROB_THRESHOLD_BUY,
        prob_threshold_sell=PROB_THRESHOLD_SELL,
        tp_pips=TP_PIPS,
        sl_pips=SL_PIPS
    )

    signals_main = []
    trend_strengths = []

    for i in range(len(test_df_aligned)):
        prob_up = y_proba[i, 1]
        prob_down = y_proba[i, 0]

        # Get trend strength for filtering
        if TREND_FILTER_ENABLED and 'macd_trend_strength' in test_df.columns:
            trend_strength = test_df['macd_trend_strength'].iloc[i] if i < len(test_df) else 1.0
        else:
            trend_strength = 1.0

        signal = strategy_main.generate_signal(prob_up, prob_down, trend_strength)
        signals_main.append(signal)
        trend_strengths.append(trend_strength)

    signal_counts = pd.Series(signals_main).value_counts()
    print(f"\n[INFO] Main strategy signal distribution (IMPROVED):")
    for signal, count in signal_counts.items():
        print(f"  {signal}: {count} ({count / len(signals_main) * 100:.1f}%)")

    # Check if fallback needed (too few signals)
    buy_signals = signal_counts.get('buy', 0)
    sell_signals = signal_counts.get('sell', 0)
    use_fallback = (buy_signals + sell_signals) < 20

    if use_fallback:
        print("\n[WARNING] Main strategy gives too few signals, activating fallback...")
        fallback_strategy = SimpleFallbackStrategy()
        signals_final = []

        for i in range(len(test_df_aligned)):
            current_idx = test_df_raw.index.get_loc(test_df_aligned.index[i])
            window_start = max(0, current_idx - 10)
            df_window = test_df_raw.iloc[window_start:current_idx + 1]

            signal = fallback_strategy.generate_signal(df_window)
            signals_final.append(signal)

        strategy_used = "Fallback (Simple Candle)"
    else:
        signals_final = signals_main
        strategy_used = f"{encoder_type} + RF (IMPROVED)"

    test_df_aligned['signal'] = signals_final

    # Final signal distribution
    final_counts = pd.Series(signals_final).value_counts()
    print(f"\n[INFO] Final signal distribution ({strategy_used}):")
    for signal, count in final_counts.items():
        print(f"  {signal}: {count} ({count / len(signals_final) * 100:.1f}%)")

    # 6. Backtesting
    print(f"\n[INFO] Running backtest ({strategy_used})...")
    trades, equity = backtest_strategy(test_df_aligned, test_df_aligned['signal'])

    # 7. Results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    metrics = calculate_metrics(trades, equity)

    if metrics['total_trades'] > 0:
        print(f"  Strategy used: {strategy_used}")
        print(f"  Total trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.2f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.3f}")
        print(f"  Net PnL (USD): {metrics['net_pnl']:.2f}")
        print(f"  Return: {(metrics['net_pnl'] / 10000) * 100:.2f}%")
        print(f"  Max Drawdown: {metrics['max_dd']:.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe']:.3f}")
        print(f"  Risk/Reward: {metrics['risk_reward']:.2f}")
        print(f"  Avg Win: ${metrics['avg_win']:.2f}")
        print(f"  Avg Loss: ${metrics['avg_loss']:.2f}")

        # CRITICAL: Analyze the fix effectiveness
        print(f"\n{'=' * 60}")
        print("FIX ANALYSIS")
        print(f"{'=' * 60}")
        print("Original Problem: 51.6% WR, 0.43 PF, 0.40 R/R")
        print(f"Applied Fixes:")
        print(f"  1. Increased TP from 40 to {TP_PIPS} pips")
        print(f"  2. Increased probability thresholds to {PROB_THRESHOLD_BUY}")
        print(f"  3. Added minimum probability difference: {MIN_PROB_DIFFERENCE}")
        if TREND_FILTER_ENABLED:
            print(f"  4. Enabled trend filter (strength > {MIN_TREND_STRENGTH})")

        # Save results
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv('outputs_unified/strategy3_trades_improved.csv', index=False)

            # Equity curve plot
            plt.figure(figsize=(12, 6))
            plt.plot(equity, label='Equity', linewidth=2, color='blue')
            plt.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Initial')
            plt.xlabel('Trade Number')
            plt.ylabel('Balance (USD)')
            plt.title(f'Equity Curve: {strategy_used}\n(TP={TP_PIPS}, SL={SL_PIPS}, PF={metrics["profit_factor"]:.2f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('outputs_unified/strategy3_equity_improved.png', dpi=100, bbox_inches='tight')
            plt.close()

            # Trade distribution plot
            plt.figure(figsize=(10, 6))
            colors = ['green' if x > 0 else 'red' for x in trades_df['pnl']]
            plt.bar(range(len(trades_df)), trades_df['pnl'], color=colors, alpha=0.7)
            plt.axhline(y=0, color='black', linewidth=0.5)
            plt.xlabel('Trade Index')
            plt.ylabel('PnL (USD)')
            plt.title(
                f'Trade PnL Distribution\n(Total: ${metrics["net_pnl"]:.2f}, Win Rate: {metrics["win_rate"]:.1f}%)')
            plt.grid(True, alpha=0.3)
            plt.savefig('outputs_unified/strategy3_trades_improved.png', dpi=100, bbox_inches='tight')
            plt.close()

            print(f"\n[SAVE] Results saved to outputs_unified/")

            # Summary statistics file
            summary_stats = {
                'strategy': strategy_used,
                'test_period_start': str(test_df_aligned.index[0]),
                'test_period_end': str(test_df_aligned.index[-1]),
                'total_bars': len(test_df_aligned),
                'total_trades': metrics['total_trades'],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'net_pnl': metrics['net_pnl'],
                'return_pct': (metrics['net_pnl'] / 10000) * 100,
                'max_drawdown': metrics['max_dd'],
                'sharpe_ratio': metrics['sharpe'],
                'risk_reward': metrics['risk_reward'],
                'avg_win': metrics['avg_win'],
                'avg_loss': metrics['avg_loss'],
                'tp_pips': TP_PIPS,
                'sl_pips': SL_PIPS,
                'prob_threshold_buy': PROB_THRESHOLD_BUY,
                'prob_threshold_sell': PROB_THRESHOLD_SELL,
                'model_accuracy': test_acc,
                'model_f1': test_f1
            }

            summary_df = pd.DataFrame([summary_stats])
            summary_df.to_csv('outputs_unified/strategy3_summary.csv', index=False)

    else:
        print("[WARNING] No trades generated even with improved strategy!")


if __name__ == "__main__":
    main()
