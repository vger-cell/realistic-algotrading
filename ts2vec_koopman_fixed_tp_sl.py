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

"""
===========================================================
TS2Vec + Koopman Coordinates — FIXED TP/SL (8-HOUR HORIZON)
Author: Vladimir Korneev
Key fixes:
1. HORIZON = 32 bars (8 hours) — kept as is
2. TP/SL = 120/60 — used directly (no auto-calculation)
3. No data leakage — Koopman trained only on train set
===========================================================
"""

# ==================== CONFIGURATION ====================
# 🔥 CRITICAL: 8-hour horizon (32 M15 bars)
HORIZON = 32

# 🔥 FIXED TP/SL for 8-hour moves (not auto-calculated!)
TP_PIPS = 40  # Take Profit in pips
SL_PIPS = 30  # Stop Loss in pips

LOT_SIZE = 0.1
SPREAD_PIPS = 2
SEQ_LENGTH = 64
TEST_SIZE = 0.25

# Signal thresholds (can be tuned)
PROB_THRESHOLD_BUY = 0.55#0.60
PROB_THRESHOLD_SELL = 0.55#0.60
MIN_PROB_DIFFERENCE = 0.10

TREND_FILTER_ENABLED = False#True
MIN_TREND_STRENGTH = 0.5

KOOPMAN_SEQ_LEN = 50
KOOPMAN_LATENT_DIM = 6
KOOPMAN_EPOCHS = 300

if not os.path.exists('outputs_fixed'):
    os.makedirs('outputs_fixed')


# ==================== PIKA MODEL (EMBEDDED) ====================
class PIKA(nn.Module):
    def __init__(self, input_dim=50, latent_dim=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.K = nn.Linear(latent_dim, latent_dim, bias=False)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        with torch.no_grad():
            self.K.weight.copy_(torch.eye(latent_dim) * 0.99)

    def forward(self, x):
        z = self.encoder(x)
        return z


def train_pika_on_prices(prices, seq_len=50, latent_dim=6, epochs=500):
    from sklearn.preprocessing import StandardScaler
    from numpy.lib.stride_tricks import sliding_window_view
    log_returns = np.diff(np.log(prices))
    log_returns = log_returns[~np.isnan(log_returns)]
    if len(log_returns) < seq_len + 10:
        raise ValueError("Not enough data for PIKA")
    scaler = StandardScaler()
    log_returns_scaled = scaler.fit_transform(log_returns.reshape(-1, 1)).flatten()
    X = sliding_window_view(log_returns_scaled, window_shape=seq_len)
    X = torch.tensor(X, dtype=torch.float32)
    device = torch.device('cpu')
    model = PIKA(input_dim=seq_len, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(X), 64):
            batch = X[i:i + 64].to(device)
            optimizer.zero_grad()
            z = model(batch)
            x_rec = model.decoder(z)
            loss = nn.MSELoss()(x_rec, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return model, scaler


def get_koopman_coords_from_model(prices, model, scaler, seq_len=50):
    from numpy.lib.stride_tricks import sliding_window_view
    log_returns = np.diff(np.log(prices))
    log_returns = log_returns[~np.isnan(log_returns)]
    log_returns_scaled = scaler.transform(log_returns.reshape(-1, 1)).flatten()
    X = sliding_window_view(log_returns_scaled, window_shape=seq_len)
    X = torch.tensor(X, dtype=torch.float32)
    model.eval()
    coords = []
    with torch.no_grad():
        for i in range(len(X)):
            z = model.encoder(X[i:i + 1])
            coords.append(z.numpy().flatten())
    return np.array(coords)


# ==================== TS2Vec MODEL ====================
class TS2Vec(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, output_dim=32, num_layers=1):
        super(TS2Vec, self).__init__()
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        outputs, (hidden, cell) = self.encoder(x)
        hidden_last = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.projector(hidden_last)


def initialize_weights(model):
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)


# ==================== DATA LOADING & PREPROCESSING ====================
def load_data_from_mt5(symbol, timeframe, n_bars=36000):
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
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['spread'] = df['high'] - df['low']
    df['body'] = abs(df['close'] - df['open'])
    df['body_ratio'] = df['body'] / df['spread'].replace(0, 1e-8)
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_trend_strength'] = abs(df['macd_hist']) / df['spread'].rolling(20).mean().replace(0, 1e-8)
    df['volatility'] = df['returns'].rolling(window=20).std()
    df['high_20'] = df['high'].rolling(window=20).max()
    df['low_20'] = df['low'].rolling(window=20).min()
    df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20']).replace(0, 1)
    df['volume_ma'] = df['tick_volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['tick_volume'] / df['volume_ma'].replace(0, 1)
    df['candle_direction'] = np.where(df['close'] > df['open'], 1, -1)
    initial_len = len(df)
    df.dropna(inplace=True)
    print(f"[DEBUG] Removed {initial_len - len(df)} rows with NaN")
    return df


def create_sequences_with_labels(df_normalized, df_raw, seq_length=64, horizon=32):
    sequences = []
    targets = []
    dates = []
    for i in range(len(df_normalized) - seq_length - horizon):
        seq_end_idx = i + seq_length
        target_idx = seq_end_idx + horizon - 1
        if target_idx >= len(df_normalized):
            continue
        seq_features = df_normalized.iloc[i:seq_end_idx].values
        future_price = df_raw['close'].iloc[target_idx]
        current_price = df_raw['close'].iloc[seq_end_idx - 1]
        threshold = 0.0050  # 3 pips
        target = 1 if future_price > current_price * (1 + threshold) else 0
        sequences.append(seq_features)
        targets.append(target)
        dates.append(df_raw.index[seq_end_idx - 1])
    return np.array(sequences), np.array(targets), dates


def prepare_datasets_with_koopman(df, seq_length=64, test_size=0.25, horizon=32):
    print(f"\n{'=' * 60}")
    print("DATA PREPARATION WITH KOOPMAN (8-HOUR HORIZON)")
    print(f"{'=' * 60}")
    split_idx = int(len(df) * (1 - test_size))
    train_df_raw = df.iloc[:split_idx].copy()
    test_df_raw = df.iloc[split_idx:].copy()
    print(f"[INFO] Split: train={len(train_df_raw):,} test={len(test_df_raw):,}")
    if train_df_raw.index[-1] >= test_df_raw.index[0]:
        raise ValueError("Data leakage detected!")
    print("[INFO] Training PIKA on TRAIN data only...")
    pika_model, pika_scaler = train_pika_on_prices(
        train_df_raw['close'].values,
        seq_len=KOOPMAN_SEQ_LEN,
        latent_dim=KOOPMAN_LATENT_DIM,
        epochs=KOOPMAN_EPOCHS
    )
    print("[INFO] Extracting Koopman coordinates...")
    koopman_train = get_koopman_coords_from_model(
        train_df_raw['close'].values, pika_model, pika_scaler, seq_len=KOOPMAN_SEQ_LEN
    )
    koopman_test = get_koopman_coords_from_model(
        test_df_raw['close'].values, pika_model, pika_scaler, seq_len=KOOPMAN_SEQ_LEN
    )
    print(f"[INFO] Koopman coords: train={koopman_train.shape}, test={koopman_test.shape}")
    train_df = prepare_features(train_df_raw)
    test_df = prepare_features(test_df_raw)
    feature_cols = [
        'open', 'high', 'low', 'close', 'tick_volume',
        'returns', 'spread', 'body', 'body_ratio',
        'macd', 'macd_signal', 'macd_hist', 'macd_trend_strength',
        'volatility', 'price_position', 'volume_ratio'
    ]
    available_cols = [col for col in feature_cols if col in train_df.columns]
    train_features = train_df[available_cols]
    test_features = test_df[available_cols]
    train_mean = train_features.mean()
    train_std = train_features.std().replace(0, 1e-8)
    train_normalized = (train_features - train_mean) / train_std
    test_normalized = (test_features - train_mean) / train_std
    train_normalized = train_normalized.clip(-5, 5)
    test_normalized = test_normalized.clip(-5, 5)
    X_train, y_train, train_dates = create_sequences_with_labels(
        train_normalized, train_df_raw, seq_length, horizon
    )
    X_test, y_test, test_dates = create_sequences_with_labels(
        test_normalized, test_df_raw, seq_length, horizon
    )
    print(f"[INFO] Base sequences: train={len(X_train):,}, test={len(X_test):,}")
    koopman_offset = KOOPMAN_SEQ_LEN - 1
    min_train = min(len(X_train), len(koopman_train))
    min_test = min(len(X_test), len(koopman_test))
    X_train = X_train[-min_train:]
    y_train = y_train[-min_train:]
    train_dates = train_dates[-min_train:]
    koopman_train = koopman_train[-min_train:]
    X_test = X_test[-min_test:]
    y_test = y_test[-min_test:]
    test_dates = test_dates[-min_test:]
    koopman_test = koopman_test[-min_test:]
    koopman_dim = koopman_train.shape[1]
    koopman_train_exp = np.repeat(koopman_train[:, np.newaxis, :], X_train.shape[1], axis=1)
    koopman_test_exp = np.repeat(koopman_test[:, np.newaxis, :], X_test.shape[1], axis=1)
    X_train = np.concatenate([X_train, koopman_train_exp], axis=2)
    X_test = np.concatenate([X_test, koopman_test_exp], axis=2)
    print(f"[INFO] Final shapes with Koopman: X_train={X_train.shape}, X_test={X_test.shape}")
    return (X_train, y_train, X_test, y_test,
            train_dates, test_dates,
            train_df, test_df, train_df_raw, test_df_raw)


# ==================== TS2Vec TRAINING ====================
def train_ts2vec_simple(X_train, y_train, epochs=3, batch_size=64):
    print(f"\n{'=' * 60}")
    print("TS2Vec ENCODER TRAINING")
    print(f"{'=' * 60}")
    if np.isnan(X_train).any():
        X_train = np.nan_to_num(X_train, nan=0.0)
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input_dim = X_train.shape[2]
    model = TS2Vec(input_dim=input_dim, output_dim=32, num_layers=1)
    initialize_weights(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def contrastive_loss(embeddings, margin=1.0):
        batch_size = embeddings.shape[0]
        embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        similarity = torch.matmul(embeddings_norm, embeddings_norm.T)
        pos_loss = torch.diag(similarity)
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        neg_similarity = similarity[~mask].view(batch_size, batch_size - 1)
        neg_loss = torch.log(torch.sum(torch.exp(neg_similarity), dim=1))
        loss = -pos_loss.mean() + neg_loss.mean()
        return loss

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for batch_X, _ in pbar:
            optimizer.zero_grad()
            embeddings = model(batch_X)
            loss = contrastive_loss(embeddings)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        if batch_count > 0:
            print(f"[INFO] Epoch {epoch + 1} | Loss={epoch_loss / batch_count:.6f}")
    return model


# ==================== TRADING STRATEGY ====================
class TradingStrategy:
    def __init__(self, prob_threshold_buy=0.60, prob_threshold_sell=0.60,
                 tp_pips=120, sl_pips=60):
        self.prob_threshold_buy = prob_threshold_buy
        self.prob_threshold_sell = prob_threshold_sell
        self.tp_pips = tp_pips
        self.sl_pips = sl_pips
        self.min_prob_difference = MIN_PROB_DIFFERENCE

    def generate_signal(self, prob_up, prob_down, trend_strength=None):
        if TREND_FILTER_ENABLED and trend_strength is not None:
            if trend_strength < MIN_TREND_STRENGTH:
                return 'hold'

        if (prob_up > self.prob_threshold_buy and
                prob_up > prob_down + self.min_prob_difference):
            return 'buy'
        elif (prob_down > self.prob_threshold_sell and
              prob_down > prob_up + self.min_prob_difference):
            return 'sell'
        elif prob_up > 0.75:
            return 'buy'
        elif prob_down > 0.75:
            return 'sell'
        return 'hold'


# ==================== BACKTEST ENGINE ====================
def backtest_strategy(df, signals, initial_balance=10000, tp_pips=120, sl_pips=60):
    # 🔒 Safety check
    if tp_pips <= 0 or sl_pips <= 0:
        raise ValueError(f"Invalid TP/SL: TP={tp_pips}, SL={sl_pips}")

    balance = initial_balance
    equity = [balance]
    trades = []
    active_trades = []

    for i in range(len(df)):
        current_time = df.index[i]
        current_price = df['close'].iloc[i]

        # Check active trades
        new_active = []
        for trade in active_trades:
            entry_price = trade['entry_price']
            position = trade['position']
            tp = trade['tp']
            sl = trade['sl']

            pnl = 0;
            exit_reason = None;
            exit_price = current_price
            if position == 'buy':
                if current_price >= tp:
                    exit_price, exit_reason = tp, 'TP'
                elif current_price <= sl:
                    exit_price, exit_reason = sl, 'SL'
            else:
                if current_price <= tp:
                    exit_price, exit_reason = tp, 'TP'
                elif current_price >= sl:
                    exit_price, exit_reason = sl, 'SL'

            if exit_reason:
                pnl = (exit_price - entry_price) * 10000 * LOT_SIZE if position == 'buy' else (
                                                                                                          entry_price - exit_price) * 10000 * LOT_SIZE
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
                new_active.append(trade)
        active_trades = new_active

        # Open new trade
        if signals.iloc[i] != 'hold' and len(active_trades) == 0:
            position = signals.iloc[i]
            entry_price = current_price
            if position == 'buy':
                tp = entry_price + tp_pips * 0.0001
                sl = entry_price - sl_pips * 0.0001
            else:
                tp = entry_price - tp_pips * 0.0001
                sl = entry_price + sl_pips * 0.0001
            active_trades.append({
                'entry_time': current_time,
                'entry_price': entry_price,
                'position': position,
                'tp': tp,
                'sl': sl
            })

        equity.append(balance)

    # Close leftovers
    for trade in active_trades:
        current_price = df['close'].iloc[-1]
        pnl = (current_price - trade['entry_price']) * 10000 * LOT_SIZE if trade['position'] == 'buy' else (trade[
                                                                                                                'entry_price'] - current_price) * 10000 * LOT_SIZE
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
    if not trades:
        return {k: 0 for k in
                ['total_trades', 'win_rate', 'profit_factor', 'net_pnl', 'max_dd', 'sharpe', 'avg_win', 'avg_loss',
                 'risk_reward']}
    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    win_rate = winning_trades / total_trades * 100
    total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    total_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    net_pnl = trades_df['pnl'].sum()
    peak = equity[0];
    max_dd = 0
    for value in equity:
        if value > peak: peak = value
        dd = (peak - value) / peak * 100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd
    returns = trades_df['pnl'].values / 10000
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 and returns.std() > 0 else 0
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
    print("STRATEGY: TS2Vec + Koopman (8-HOUR HORIZON, FIXED TP/SL)")
    print(f"HORIZON = {HORIZON} bars | TP = {TP_PIPS} pips | SL = {SL_PIPS} pips")
    print("=" * 60)

    df = load_data_from_mt5("EURUSD", mt5.TIMEFRAME_M15, n_bars=36000)
    if df is None:
        return

    try:
        (X_train, y_train, X_test, y_test,
         train_dates, test_dates,
         train_df, test_df,
         train_df_raw, test_df_raw) = prepare_datasets_with_koopman(
            df, seq_length=SEQ_LENGTH, test_size=TEST_SIZE, horizon=HORIZON
        )
    except Exception as e:
        print(f"[ERROR] Data preparation failed: {e}")
        return

    encoder = train_ts2vec_simple(X_train, y_train, epochs=3, batch_size=64)
    print("[INFO] Generating embeddings...")
    encoder.eval()
    with torch.no_grad():
        train_embeddings = encoder(torch.FloatTensor(X_train)).numpy()
        test_embeddings = encoder(torch.FloatTensor(X_test)).numpy()

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        train_embeddings, y_train, test_size=0.2, random_state=42, shuffle=False
    )
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_split), y=y_train_split)
    sample_weights = class_weights[y_train_split]

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2,
        random_state=42, class_weight='balanced', n_jobs=-1
    )
    clf.fit(X_train_split, y_train_split, sample_weight=sample_weights)

    calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
    calibrated_clf.fit(X_val_split, y_val_split)

    y_pred = calibrated_clf.predict(test_embeddings)
    y_proba = calibrated_clf.predict_proba(test_embeddings)

    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)
    print(classification_report(y_test, y_pred, digits=3))

    test_acc = (y_test == y_pred).mean()
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"[INFO] Accuracy: {test_acc:.3f} | F1: {test_f1:.3f}")

    # Generate signals
    test_df_aligned = test_df_raw.loc[test_dates].copy()
    if len(test_df_aligned) > len(y_proba):
        test_df_aligned = test_df_aligned.iloc[:len(y_proba)]
    elif len(y_proba) > len(test_df_aligned):
        y_proba = y_proba[:len(test_df_aligned)]

    # ✅ USE FIXED TP/SL — NO AUTO-CALCULATION
    strategy = TradingStrategy(
        prob_threshold_buy=PROB_THRESHOLD_BUY,
        prob_threshold_sell=PROB_THRESHOLD_SELL,
        tp_pips=TP_PIPS,
        sl_pips=SL_PIPS
    )

    signals_final = []
    for i in range(len(test_df_aligned)):
        prob_up = y_proba[i, 1]
        prob_down = y_proba[i, 0]
        trend_strength = test_df['macd_trend_strength'].iloc[i] if 'macd_trend_strength' in test_df.columns and i < len(
            test_df) else 1.0
        signal = strategy.generate_signal(prob_up, prob_down, trend_strength)
        signals_final.append(signal)

    test_df_aligned['signal'] = signals_final
    final_counts = pd.Series(signals_final).value_counts()
    print(f"\n[INFO] Final signal distribution:")
    for signal, count in final_counts.items():
        print(f"  {signal}: {count} ({count / len(signals_final) * 100:.1f}%)")

    # ✅ BACKTEST WITH FIXED TP/SL
    trades, equity = backtest_strategy(
        test_df_aligned,
        test_df_aligned['signal'],
        tp_pips=TP_PIPS,
        sl_pips=SL_PIPS
    )
    metrics = calculate_metrics(trades, equity)

    print("\n" + "=" * 60)
    print("FINAL RESULTS (8-HOUR HORIZON, FIXED TP/SL)")
    print(f"TP = {TP_PIPS} pips, SL = {SL_PIPS} pips")
    print("=" * 60)
    if metrics['total_trades'] > 0:
        print(f"  Total trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.2f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.3f}")
        print(f"  Net PnL: ${metrics['net_pnl']:.2f}")
        print(f"  Return: {(metrics['net_pnl'] / 10000) * 100:.2f}%")
        print(f"  Max Drawdown: {metrics['max_dd']:.2f}%")
        print(f"  Risk/Reward: {metrics['risk_reward']:.2f}")

        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv('outputs_fixed/trades_fixed.csv', index=False)
            plt.figure(figsize=(12, 6))
            plt.plot(equity, label='Equity', linewidth=2, color='blue')
            plt.axhline(y=10000, color='gray', linestyle='--', alpha=0.5)
            plt.title(
                f'Equity Curve (8-Hour Horizon)\nPF={metrics["profit_factor"]:.2f}, RR={metrics["risk_reward"]:.2f}')
            plt.legend();
            plt.grid(True, alpha=0.3)
            plt.savefig('outputs_fixed/equity_fixed.png', dpi=100, bbox_inches='tight')
            plt.close()

            summary_stats = {
                'total_trades': metrics['total_trades'],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'net_pnl': metrics['net_pnl'],
                'max_dd': metrics['max_dd'],
                'risk_reward': metrics['risk_reward'],
                'tp_pips': TP_PIPS,
                'sl_pips': SL_PIPS,
                'horizon_bars': HORIZON,
                'model_accuracy': test_acc,
                'model_f1': test_f1
            }
            pd.DataFrame([summary_stats]).to_csv('outputs_fixed/summary_fixed.csv', index=False)
    else:
        print("[WARNING] No trades generated!")


if __name__ == "__main__":
    main()
