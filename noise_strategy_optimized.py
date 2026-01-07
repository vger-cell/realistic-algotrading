# noise_strategy_optimized.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ==============================
# PARAMETERS
# ==============================
CSV_FILE = "EURUSDM1.csv"
LOT = 0.1
TIMEOUT_MIN = 3
POINT = 0.00001
PIPS_TO_USD = 10.0  # for EURUSD: 1 pip = $1 with lot 0.1

# Optimization parameters
REG_WINDOW_CANDIDATES = [30, 50, 70, 100]
QUANTILE_CANDIDATES = [0.90, 0.93, 0.95, 0.97]
VOLUME_WINDOW = 1000  # for volume median

# ==============================
# LOAD CSV (UTF-16 LE BOM)
# ==============================
def load_csv_data(filepath):
    df = pd.read_csv(
        filepath,
        encoding='utf-16',
        sep=',',
        header=None,
        names=["datetime", "open", "high", "low", "close", "volume", "spread"]
    )
    df['datetime'] = pd.to_datetime(df['datetime'], format="%Y.%m.%d %H:%M")
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    print(f"✅ Loaded {len(df)} bars")
    print(f"   Period: {df.index[0]} → {df.index[-1]}")
    return df[['close', 'volume', 'spread']].copy()

# ==============================
# COMPUTE NOISE + DYNAMIC THRESHOLD
# ==============================
def generate_signals_optimized(df, reg_window, quantile, vol_window=1000):
    df = df.copy()
    # 1. Noise via regression
    noise = pd.Series(index=df.index, dtype='float64')
    x = np.arange(reg_window)
    for i in range(reg_window - 1, len(df)):
        y_vals = df['close'].iloc[i - reg_window + 1: i + 1].values
        if len(y_vals) < reg_window:
            continue
        A = np.vstack([x, np.ones(reg_window)]).T
        slope, intercept = np.linalg.lstsq(A, y_vals, rcond=None)[0]
        trend_val = slope * (reg_window - 1) + intercept
        noise.iloc[i] = df['close'].iloc[i] - trend_val
    df['noise'] = noise

    # 2. Dynamic threshold (quantile of absolute noise)
    df['noise_abs'] = df['noise'].abs()
    df['threshold'] = df['noise_abs'].rolling(window=2000, min_periods=100).quantile(quantile)

    # 3. Volume filter
    df['vol_median'] = df['volume'].rolling(window=vol_window, min_periods=1).median()
    df['high_volume'] = df['volume'] > df['vol_median']

    # 4. Signals
    df['signal'] = 0
    buy_cond = (df['noise'] < -df['threshold']) & df['high_volume']
    sell_cond = (df['noise'] > df['threshold']) & df['high_volume']
    df.loc[buy_cond, 'signal'] = 1
    df.loc[sell_cond, 'signal'] = -1

    return df

# ==============================
# BACKTEST WITH SPREAD
# ==============================
def backtest_with_spread(df, lot=0.1, timeout_min=3):
    equity = 0.0
    equity_curve = []
    timestamps = []
    position = 0
    entry_price = None
    entry_spread = 0
    next_trade_time = df.index[0]

    for i in range(len(df)):
        current_time = df.index[i]
        row = df.iloc[i]

        if current_time < next_trade_time:
            signal = 0
        else:
            signal = row['signal']

        # Close position
        if position != 0:
            # On exit: sell at bid (close), bought at ask (entry_price = close + spread)
            # Spread cost is accounted for at entry, but kept symmetric here:
            if position == 1:
                pnl_pips = (row['close'] - (entry_price + entry_spread * POINT)) / POINT
            else:
                pnl_pips = ((entry_price - entry_spread * POINT) - row['close']) / POINT

            pnl_usd = pnl_pips * (lot * PIPS_TO_USD)
            equity += pnl_usd
            equity_curve.append(equity)
            timestamps.append(current_time)
            position = 0

            if pnl_usd < 0:
                next_trade_time = current_time + timedelta(minutes=timeout_min)

        # Open new position
        if signal != 0 and position == 0:
            position = signal
            entry_price = row['close']
            entry_spread = row['spread']  # spread in points

    return pd.Series(equity_curve, index=timestamps) if equity_curve else pd.Series()

# ==============================
# OPTIMIZE ON WF
# ==============================
def optimize_parameters(wf_data):
    best_pnl = -np.inf
    best_params = (50, 0.95)

    print("🔍 Optimizing parameters on WF period...")
    for reg_w in REG_WINDOW_CANDIDATES:
        for q in QUANTILE_CANDIDATES:
            try:
                df_signals = generate_signals_optimized(wf_data, reg_w, q, VOLUME_WINDOW)
                equity = backtest_with_spread(df_signals, LOT, TIMEOUT_MIN)
                pnl = equity.iloc[-1] if not equity.empty else -np.inf
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_params = (reg_w, q)
                print(f"  reg_window={reg_w}, quantile={q:.2f} → PnL={pnl:.1f}")
            except Exception as e:
                print(f"  Error with reg_window={reg_w}, quantile={q}: {e}")

    print(f"✅ Best parameters: reg_window={best_params[0]}, quantile={best_params[1]:.2f}")
    return best_params

# ==============================
# CALCULATE METRICS
# ==============================
def calculate_metrics(equity_series):
    if equity_series.empty:
        return {'trades': 0, 'pnl': 0, 'win_rate': 0, 'max_dd': 0}
    returns = equity_series.diff().dropna()
    pnl = equity_series.iloc[-1]
    win_rate = (returns > 0).mean() if len(returns) > 0 else 0
    cum = equity_series
    dd = (cum.cummax() - cum).max()
    return {
        'trades': len(equity_series),
        'pnl': round(pnl, 2),
        'win_rate': round(win_rate * 100, 2),
        'max_dd': round(dd, 2)
    }

# ==============================
# MAIN
# ==============================
def main():
    print("🚀 Optimized noise-based strategy (with spread and volatility adjustment)")

    data = load_csv_data(CSV_FILE)
    if len(data) < 2000:
        print("❌ Insufficient data")
        return

    # Split data
    oos_start_idx = int(len(data) * 0.75)
    wf_data = data.iloc[:oos_start_idx].copy()
    oos_data = data.iloc[oos_start_idx:].copy()

    print(f"\nSplit:")
    print(f"WF period : {wf_data.index[0]} → {wf_data.index[-1]} ({len(wf_data)} bars)")
    print(f"OOS test  : {oos_data.index[0]} → {oos_data.index[-1]} ({len(oos_data)} bars)")

    # Optimize on WF
    best_reg_window, best_quantile = optimize_parameters(wf_data)

    # Final test on WF with best parameters
    wf_final = generate_signals_optimized(wf_data, best_reg_window, best_quantile, VOLUME_WINDOW)
    wf_equity = backtest_with_spread(wf_final, LOT, TIMEOUT_MIN)

    # OOS test with same parameters
    oos_final = generate_signals_optimized(oos_data, best_reg_window, best_quantile, VOLUME_WINDOW)
    oos_equity = backtest_with_spread(oos_final, LOT, TIMEOUT_MIN)

    # Metrics
    wf_metrics = calculate_metrics(wf_equity)
    oos_metrics = calculate_metrics(oos_equity)

    print("\n" + "=" * 60)
    print("FINAL RESULTS (with spread accounted for)")
    print("=" * 60)
    print(
        f"WF period → Trades: {wf_metrics['trades']}, PnL: {wf_metrics['pnl']} USD, Win%: {wf_metrics['win_rate']}, MaxDD: {wf_metrics['max_dd']} USD")
    print(
        f"OOS test  → Trades: {oos_metrics['trades']}, PnL: {oos_metrics['pnl']} USD, Win%: {oos_metrics['win_rate']}, MaxDD: {oos_metrics['max_dd']} USD")

    # Buy & Hold (OOS)
    bh_pips = (oos_data['close'].iloc[-1] - oos_data['close'].iloc[0]) / POINT
    bh_usd = bh_pips * (LOT * PIPS_TO_USD)
    print(f"\nBuy & Hold (OOS): {bh_usd:.2f} USD")

    # Visualization
    plt.figure(figsize=(12, 6))
    if not wf_equity.empty:
        plt.plot(wf_equity.index, wf_equity.values, label='WF Equity', color='steelblue', alpha=0.8)
    if not oos_equity.empty:
        plt.plot(oos_equity.index, oos_equity.values, label='OOS Equity', color='darkgreen', linewidth=2)
    plt.axvline(oos_data.index[0], color='red', linestyle='--', label='OOS Start')
    plt.title(f'Equity Curve (reg_window={best_reg_window}, quantile={best_quantile:.2f})')
    plt.xlabel('Time')
    plt.ylabel('Equity (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
