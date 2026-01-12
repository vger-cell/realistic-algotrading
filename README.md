"""
M1 Randomness Trader - Walk-Forward Backtest with Pip-based Forecast
Author: Vladimir Korneev, 2026
Telegram: t.me/realistic_algotrading
Repo: github.com/vger-cell/realistic-algotrading

A realistic evaluation of short-term predictability on EURUSD M1 data.
Uses CNN with Gaussian uncertainty estimation to detect tradable edges.
Walk-forward validation reveals no statistically significant signals,
demonstrating market efficiency at minute timeframes.

Key findings:
- Real 10-bar volatility: ~2.6 pips
- Model uncertainty: ~2.8-3.3 pips (well-calibrated)
- No signals meet edge criteria (|mean|/σ > 1.5)
- Honest assessment: M1 EURUSD shows no exploitable patterns

Educational value: Proper backtesting methodology with uncertainty
quantification prevents false positive discoveries.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import linregress
import tensorflow as tf
import os, warnings, traceback
warnings.filterwarnings('ignore')

# ==============================================================================
# ПАРАМЕТРЫ
# ==============================================================================

SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
WINDOW_SIZE = 80
HORIZONS = [10, 15]          # баров вперёд
PIP_SIZE = 0.0001            # для EURUSD
CONFIDENCE_STD_THRESHOLD = 8.0  # торговать только если σ < 8 пипсов
MIN_EDGE = 1.5               # минимальное |mean| / σ для сигнала
MODEL_PATH = "m1_pip_model.h5"
CACHE_FILE = "mt5_eurusd_m1_50k.csv"

# ==============================================================================
# MT5 + ДАННЫЕ
# ==============================================================================

def init_mt5():
    if not mt5.initialize():
        raise RuntimeError("❌ MT5 init failed")
    print("✅ MT5 initialized")

def fetch_or_load_data(n_bars=50000):
    if os.path.exists(CACHE_FILE):
        print("💾 Loading cached data...")
        df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
    else:
        print("📥 Fetching data from MT5...")
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, n_bars)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df[['close', 'tick_volume']].rename(columns={'close': 'Close', 'tick_volume': 'Volume'})
        df.to_csv(CACHE_FILE)
    return df

# ==============================================================================
# ФИЧИ И ЦЕЛЬ В ПИПСАХ
# ==============================================================================

def normalize_window(close, volume):
    close_norm = close / close[0] - 1.0
    vol_norm = volume / (np.mean(volume) + 1e-8)
    return np.stack([close_norm, vol_norm], axis=-1).astype(np.float32)

def compute_regression_features(prices):
    x = np.arange(len(prices))
    try:
        slope, _, r_value, _, _ = linregress(x, prices)
        return np.array([slope, r_value**2], dtype=np.float32)
    except:
        return np.array([0.0, 0.0])

def prepare_dataset(df, window=80, horizons=[10, 15]):
    prices = df['Close'].values
    volumes = df['Volume'].values
    X_img, X_reg, Y, timestamps = [], [], [], []

    for i in range(window, len(df) - max(horizons)):
        win_close = prices[i - window:i]
        win_vol = volumes[i - window:i]

        img = normalize_window(win_close, win_vol)
        reg = compute_regression_features(win_close)

        current = prices[i]
        # Цель в ПИПСАХ!
        targets = [(prices[i + h] - current) / PIP_SIZE for h in horizons]

        X_img.append(img)
        X_reg.append(reg)
        Y.append(targets)
        timestamps.append(df.index[i])  # время входа

    return (
        np.array(X_img),
        np.array(X_reg),
        np.array(Y, dtype=np.float32),
        np.array(timestamps)
    )

# ==============================================================================
# МОДЕЛЬ И ПОТЕРЯ
# ==============================================================================

def gaussian_nll_loss(y_true, y_pred):
    means = y_pred[:, ::2]
    log_vars = y_pred[:, 1::2]
    nll = 0.5 * (log_vars + tf.square(y_true - means) / (tf.exp(log_vars) + 1e-8))
    return tf.reduce_mean(nll)

def build_model(input_shape=(80, 2), reg_dim=2, n_horizons=2):
    img_input = tf.keras.Input(shape=input_shape, name='img_input')
    reg_input = tf.keras.Input(shape=(reg_dim,), name='reg_input')

    x = tf.keras.layers.Conv1D(16, 5, activation='relu', padding='same')(img_input)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)

    combined = tf.keras.layers.concatenate([x, reg_input])
    dense = tf.keras.layers.Dense(64, activation='relu')(combined)
    output = tf.keras.layers.Dense(2 * n_horizons, name='distribution_output')(dense)

    model = tf.keras.Model(inputs=[img_input, reg_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=gaussian_nll_loss)
    return model

# ==============================================================================
# WALK-FORWARD BACKTEST
# ==============================================================================

def walk_forward_backtest(df, test_days=10):
    """
    Обучаем на всех данных до дня D, тестируем на день D.
    Последние 10% — hold-out валидация.
    """
    df = df.sort_index()
    total_days = (df.index[-1] - df.index[0]).days
    if total_days < 20:
        raise ValueError("Need at least 20 days of data")

    # Hold-out: последние 10%
    holdout_start = df.index[int(0.9 * len(df))]
    df_trainval = df[df.index < holdout_start]
    df_holdout = df[df.index >= holdout_start]

    # Walk-forward: последние `test_days` дней как тест
    test_start = df_trainval.index[-1] - timedelta(days=test_days)
    df_train = df_trainval[df_trainval.index < test_start]
    df_test = df_trainval[df_trainval.index >= test_start]

    print(f"📅 Train: {df_train.index[0].date()} → {df_train.index[-1].date()}")
    print(f"🧪 Test (walk-forward): {df_test.index[0].date()} → {df_test.index[-1].date()}")
    print(f"🔒 Hold-out: {df_holdout.index[0].date()} → {df_holdout.index[-1].date()}")

    # Обучение на df_train
    X_img, X_reg, Y, _ = prepare_dataset(df_train, WINDOW_SIZE, HORIZONS)
    model = build_model(n_horizons=len(HORIZONS))

    split = int(0.85 * len(X_img))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]

    model.fit(
        {'img_input': X_img[:split], 'reg_input': X_reg[:split]},
        Y[:split],
        validation_data=({'img_input': X_img[split:], 'reg_input': X_reg[split:]}, Y[split:]),
        epochs=30, batch_size=128, callbacks=callbacks, verbose=0
    )

    # Тест на walk-forward
    X_img_test, X_reg_test, Y_test, ts_test = prepare_dataset(df_test, WINDOW_SIZE, HORIZONS)
    pred_test = model.predict({'img_input': X_img_test, 'reg_input': X_reg_test}, verbose=0)

    # Hold-out
    X_img_ho, X_reg_ho, Y_ho, ts_ho = prepare_dataset(df_holdout, WINDOW_SIZE, HORIZONS)
    pred_ho = model.predict({'img_input': X_img_ho, 'reg_input': X_reg_ho}, verbose=0)

    # Оценка
    def evaluate_predictions(Y_true, Y_pred, name):
        results = {}
        for i, h in enumerate(HORIZONS):
            mean = Y_pred[:, 2*i]
            std = np.sqrt(np.exp(Y_pred[:, 2*i + 1]))
            actual = Y_true[:, i]

            edge = np.abs(mean) / (std + 1e-6)
            signal = (std < CONFIDENCE_STD_THRESHOLD) & (edge > MIN_EDGE)

            returns = np.where(mean > 0, actual, -actual)  # long/short по прогнозу
            strategy_returns = returns[signal]

            if len(strategy_returns) > 0:
                pf = max(1e-6, strategy_returns[strategy_returns > 0].sum()) / \
                     max(1e-6, -strategy_returns[strategy_returns < 0].sum())
                results[h] = {
                    'signals': len(strategy_returns),
                    'mean_return': np.mean(strategy_returns),
                    'sharpe': np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-6),
                    'profit_factor': pf,
                    'win_rate': np.mean(strategy_returns > 0)
                }
            else:
                results[h] = {'signals': 0}
        return results

    test_results = evaluate_predictions(Y_test, pred_test, "Test")
    ho_results = evaluate_predictions(Y_ho, pred_ho, "Hold-out")

    print("\n📊 WALK-FORWARD TEST RESULTS:")
    for h in HORIZONS:
        r = test_results[h]
        if r['signals'] > 0:
            print(f"  Horizon {h}: {r['signals']} signals | "
                  f"Mean: {r['mean_return']:.2f} pip | "
                  f"PF: {r['profit_factor']:.2f} | "
                  f"Win%: {r['win_rate']*100:.1f}%")
        else:
            print(f"  Horizon {h}: no valid signals")

    print("\n🔒 HOLD-OUT VALIDATION:")
    for h in HORIZONS:
        r = ho_results[h]
        if r['signals'] > 0:
            print(f"  Horizon {h}: {r['signals']} signals | "
                  f"Mean: {r['mean_return']:.2f} pip | "
                  f"PF: {r['profit_factor']:.2f}")
        else:
            print(f"  Horizon {h}: no valid signals")

    # Сохраняем модель
    model.save(MODEL_PATH)
    print(f"\n💾 Model saved to {MODEL_PATH}")
    return model

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("🚀 M1 Randomness Trader – Walk-Forward Backtest")
    init_mt5()

    try:
        df = fetch_or_load_data(50000)
        print(f"📈 Loaded {len(df)} bars ({df.index[0]} → {df.index[-1]})")

        # Реальная волатильность
        ret10 = df['Close'].pct_change(10).dropna()
        real_std_10 = ret10.std() / PIP_SIZE
        print(f"🔍 Real volatility (10 bars): {real_std_10:.1f} pips")

        model = walk_forward_backtest(df, test_days=7)

        # Пример инференса
        X_img, X_reg, Y, ts = prepare_dataset(df.tail(200), WINDOW_SIZE, HORIZONS)
        if len(X_img) > 0:
            pred = model.predict({'img_input': X_img[-1:], 'reg_input': X_reg[-1:]}, verbose=0)[0]
            print("\n🔮 Latest prediction:")
            for i, h in enumerate(HORIZONS):
                mean = pred[2*i]
                std = np.sqrt(np.exp(pred[2*i+1]))
                print(f"  Horizon {h}: {mean:+.1f} ± {std:.1f} pips")

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
    finally:
        mt5.shutdown()
        print("\n🔌 MT5 shutdown")

if __name__ == "__main__":
    main()
