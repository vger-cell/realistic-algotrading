# -*- coding: utf-8 -*-
"""
Unified Strategy Runner: S1 (Per-Symbol Optimizer) + S2 (Aggressive EURUSD) + S3 (TS2Vec+Classifier)
Author: Vladimir Korneev
Telegram: t.me/realistic_algotrading
Repository: github.com/vger-cell/realistic-algotrading

Description:
    Executes three distinct algorithmic trading strategies in sequence:
    1. S1 - Per-symbol parameter optimization for EURUSD and AUDUSD (H1 timeframe)
    2. S2 - Aggressive EURUSD trading with BOCPD, regime switching, EVT, and bandit logic (H1)
    3. S3 - TS2Vec embeddings with calibrated classifier for EURUSD (M15 timeframe)

    Each strategy runs a complete backtest with training/validation/testing split.
    All trades are aggregated into a unified portfolio with consolidated metrics.
    Results are saved to CSV and plotted as equity curves.

Period:
    Approximately May 2024 - December 2025 (varies by symbol and data availability)

Markets:
    Forex: EURUSD, AUDUSD, GBPUSD (depending on strategy configuration)

Outputs:
    - outputs_unified/all_trades.csv: All trades from all strategies
    - outputs_unified/equity_all.png: Portfolio and individual strategy equity curves

Dependencies:
    MetaTrader5, pandas, numpy, scikit-learn, matplotlib, torch, xgboost

Important Notes:
    - This is a BACKTESTING framework only, not a live trading system
    - Results show mixed performance: only S1 was profitable in test period
    - All strategies have significant limitations and should not be used for live trading
    - Educational purposes only - demonstrates multi-strategy portfolio construction
"""

import warnings

warnings.filterwarnings("ignore")

import os
import math
import csv
import json
import random
from dataclasses import dataclass
from collections import deque, defaultdict
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import MetaTrader5 as MT5

mt5 = MT5  # alias for S3 block

# ============================================================================
# ============================  SHARED / UNIFIED  ============================
# ============================================================================

UNIFIED_OUT_DIR = "outputs_unified"
UNIFIED_TRADES_CSV = os.path.join(UNIFIED_OUT_DIR, "all_trades.csv")
UNIFIED_EQUITY_PNG = os.path.join(UNIFIED_OUT_DIR, "equity_all.png")

RISK_FREE = 0.0
LOT = 0.1  # Fixed lot size for USD calculations
S3_PIP_VALUE_PER_LOT = 10.0  # $/pip for 1.0 lot; at LOT=0.1 → $1/pip
DEFAULT_SPREAD_PIPS = {"EURUSD": 1.5, "AUDUSD": 1.8, "GBPUSD": 2.0}
MARGIN_RATE = 0.02  # ~2% margin, for estimating peak total margin
CONTRACT_SIZE = 100000  # Standard forex contract size


def ensure_dir(path: str):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def pip_size(symbol: str) -> float:
    """Get pip size for a symbol (0.01 for JPY pairs, 0.0001 for others)"""
    return 0.01 if "JPY" in symbol else 0.0001


def pips_between(a: float, b: float, symbol: str) -> float:
    """Calculate pips between two prices"""
    return (b - a) / pip_size(symbol)


def money_per_pip_per_lot(symbol: str) -> float:
    """Get money value per pip per lot (simplified to $10 for USD-quoted)"""
    return 10.0


def _strip_tz_series(s: pd.Series) -> pd.Series:
    """
    Convert to timezone-naive (remove tz) without breaking order.
    """
    s = pd.to_datetime(s, errors='coerce')
    try:
        # If series is tz-aware
        return s.dt.tz_convert(None)
    except Exception:
        # If already tz-naive
        return s.dt.tz_localize(None)


def normalize_times_df(df: pd.DataFrame, entry_col="entry_time", exit_col="exit_time") -> pd.DataFrame:
    """Normalize time columns to timezone-naive format"""
    if entry_col in df.columns:
        df[entry_col] = _strip_tz_series(df[entry_col])
    if exit_col in df.columns:
        df[exit_col] = _strip_tz_series(df[exit_col])
    return df


def normalize_eq_points(eq_points: List[tuple]) -> List[tuple]:
    """Normalize equity curve timestamps to timezone-naive"""
    out = []
    for t, v in eq_points:
        tt = pd.to_datetime(t, errors='coerce')
        try:
            tt = tt.tz_convert(None)
        except Exception:
            tt = tt.tz_localize(None)
        out.append((tt, float(v)))
    return out


def calc_metrics_usd(trades_df: pd.DataFrame) -> dict:
    """Calculate universal metrics from trades DataFrame with columns ['exit_time','pnl_usd']"""
    if trades_df.empty:
        return dict(trades=0)
    df = trades_df.copy()
    df = normalize_times_df(df, "entry_time", "exit_time")
    df = df.sort_values("exit_time").reset_index(drop=True)
    total_trades = len(df)
    wins = (df["pnl_usd"] > 0).sum()
    winrate = wins / total_trades if total_trades else 0.0
    gp = df.loc[df["pnl_usd"] > 0, "pnl_usd"].sum()
    gl = -df.loc[df["pnl_usd"] < 0, "pnl_usd"].sum()
    pf = (gp / gl) if gl > 0 else np.inf
    net = df["pnl_usd"].sum()

    eq = df["pnl_usd"].cumsum().values
    peaks = np.maximum.accumulate(eq) if len(eq) else np.array([0.0])
    dd = (peaks - eq) if len(eq) else np.array([0.0])
    maxdd = dd.max() if len(dd) > 0 else 0.0
    ret = np.diff(np.insert(eq, 0, 0.0))
    sharpe = (ret.mean() - RISK_FREE) / (ret.std() + 1e-9)
    return dict(trades=total_trades, winrate=winrate, profit_factor=pf,
                net_money=net, max_dd=maxdd, sharpe=sharpe)


def print_metrics(title: str, metrics: dict):
    """Print formatted metrics"""
    print(f"\n[{title}]")
    if not metrics or metrics.get("trades", 0) == 0:
        print("  No trades.")
        return
    print(f"  Trades: {metrics['trades']}")
    print(f"  WinRate: {metrics['winrate'] * 100:.2f}%")
    print(f"  Profit Factor: {metrics['profit_factor']:.3f}")
    print(f"  Net PnL (USD): {metrics['net_money']:.2f}")
    print(f"  Max Drawdown (USD): {metrics['max_dd']:.2f}")
    print(f"  Sharpe: {metrics['sharpe']:.3f}")


def compute_peak_margin(all_rows: List[dict]) -> float:
    """
    Approximate peak total margin: sum margin for concurrent positions
    margin ≈ CONTRACT_SIZE * LOT * entry_price * MARGIN_RATE (in quote currency; USD for USD-quoted)
    """
    events = []
    for r in all_rows:
        sym = r.get("symbol", "")
        if "USD" not in sym:  # Simplification for USD-quoted pairs
            continue
        ent = pd.to_datetime(r["entry_time"], errors='coerce')
        ex = pd.to_datetime(r["exit_time"], errors='coerce')
        try:
            ent = ent.tz_convert(None)
        except Exception:
            ent = ent.tz_localize(None)
        try:
            ex = ex.tz_convert(None)
        except Exception:
            ex = ex.tz_localize(None)
        price = float(r.get("entry_price", r.get("entry_exec_price", r.get("entry_px", 1.0))))
        margin = CONTRACT_SIZE * LOT * price * MARGIN_RATE
        events.append((ent, +margin))
        events.append((ex, -margin))
    if not events:
        return 0.0
    events.sort(key=lambda x: (x[0], -x[1]))
    cur = 0.0;
    peak = 0.0
    for _, delta in events:
        cur += delta
        if cur > peak:
            peak = cur
    return peak


def plot_unified_equity(curves: dict, title: str, path: str):
    """
    Plot equity curves for portfolio and individual strategies
    curves: {name: list[(time, equity_usd)]}
    """
    ensure_dir(os.path.dirname(path))
    plt.figure(figsize=(11, 4.5))
    for name, seq in curves.items():
        if not seq:
            continue
        seq = normalize_eq_points(seq)
        ts = pd.DataFrame(seq, columns=["time", "equity"]).dropna().sort_values("time")
        plt.plot(ts["time"], ts["equity"], label=name, linewidth=1.2)
    plt.legend(loc="best", frameon=False)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("PNL (USD)")
    plt.grid(True, linewidth=0.3, alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


# ============================================================================
# ===================== MAGIC / LIVE CONTROL HELPERS =========================
# ============================================================================
# UNIQUE magic numbers for each strategy. Use these in a real bot
# for entry/modification/close operations. This file doesn't apply them -
# functions below are provided for integration into realtime scripts.
MAGIC_NUMBERS = {
    "STRAT1": 100001,
    "STRAT2": 100002,
    "STRAT3": 100003,
}


def mt5_point(symbol: str) -> float:
    """Get MT5 point size for a symbol"""
    si = MT5.symbol_info(symbol)
    return si.point if si else 0.0001


def mt5_get_tick(symbol: str):
    """Get current tick for a symbol"""
    if not MT5.symbol_select(symbol, True):
        return None
    return MT5.symbol_info_tick(symbol)


def mt5_positions_by_magic(magic: int, symbol: Optional[str] = None):
    """Get positions by magic number"""
    pos = MT5.positions_get(symbol=symbol)
    if pos is None:
        return []
    return [p for p in pos if int(getattr(p, "magic", 0)) == int(magic)]


def mt5_orders_by_magic(magic: int, symbol: Optional[str] = None):
    """Get orders by magic number"""
    orders = MT5.orders_get(symbol=symbol)
    if orders is None:
        return []
    return [o for o in orders if int(getattr(o, "magic", 0)) == int(magic)]


def mt5_open_trade_with_magic(symbol: str, side: str, volume: float, sl_price: Optional[float] = None,
                              tp_price: Optional[float] = None, magic: int = 0, comment: str = "",
                              deviation: int = 20):
    """Open a trade with magic number"""
    if not MT5.symbol_select(symbol, True):
        return {"retcode": -1, "comment": "symbol_select failed"}
    tick = MT5.symbol_info_tick(symbol)
    if tick is None:
        return {"retcode": -2, "comment": "no tick"}
    price = tick.ask if side.upper() == "BUY" else tick.bid
    req = {
        "action": MT5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "type": MT5.ORDER_TYPE_BUY if side.upper() == "BUY" else MT5.ORDER_TYPE_SELL,
        "price": price,
        "volume": float(volume),
        "deviation": int(deviation),
        "magic": int(magic),
        "comment": comment or f"{side.upper()}-{magic}",
        "type_filling": MT5.ORDER_FILLING_IOC,
    }
    if sl_price is not None:
        req["sl"] = float(sl_price)
    if tp_price is not None:
        req["tp"] = float(tp_price)
    return MT5.order_send(req)


def mt5_modify_position_sl_tp(position_ticket: int, symbol: str, sl_price: Optional[float], tp_price: Optional[float],
                              deviation: int = 20):
    """Modify SL/TP for an existing position"""
    req = {
        "action": MT5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": int(position_ticket),
        "sl": float(sl_price) if sl_price is not None else 0.0,
        "tp": float(tp_price) if tp_price is not None else 0.0,
        "deviation": int(deviation),
    }
    return MT5.order_send(req)


def mt5_close_position_by_ticket(position_ticket: int, symbol: str, deviation: int = 20):
    """Close position by ticket number"""
    pos = MT5.positions_get(ticket=position_ticket)
    if not pos:
        return {"retcode": -3, "comment": "position not found"}
    p = pos[0]
    tick = MT5.symbol_info_tick(symbol)
    if tick is None:
        return {"retcode": -2, "comment": "no tick"}
    close_type = MT5.ORDER_TYPE_SELL if p.type == MT5.POSITION_TYPE_BUY else MT5.ORDER_TYPE_BUY
    price = tick.bid if close_type == MT5.ORDER_TYPE_SELL else tick.ask
    req = {
        "action": MT5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "type": close_type,
        "position": int(position_ticket),
        "price": price,
        "volume": float(p.volume),
        "deviation": int(deviation),
        "magic": int(getattr(p, "magic", 0)),
        "comment": f"CLOSE-{int(getattr(p, 'magic', 0))}",
        "type_filling": MT5.ORDER_FILLING_IOC,
    }
    return MT5.order_send(req)


# Example "strategic" helpers (not used inside this file):
def s1_open_buy(symbol: str, volume: float, sl_price: Optional[float], tp_price: Optional[float]):
    return mt5_open_trade_with_magic(symbol, "BUY", volume, sl_price, tp_price, MAGIC_NUMBERS["STRAT1"], "S1")


def s2_open_buy(symbol: str, volume: float, sl_price: Optional[float], tp_price: Optional[float]):
    return mt5_open_trade_with_magic(symbol, "BUY", volume, sl_price, tp_price, MAGIC_NUMBERS["STRAT2"], "S2")


def s3_open_buy(symbol: str, volume: float, sl_price: Optional[float], tp_price: Optional[float]):
    return mt5_open_trade_with_magic(symbol, "BUY", volume, sl_price, tp_price, MAGIC_NUMBERS["STRAT3"], "S3")


def s1_positions(symbol: Optional[str] = None):
    return mt5_positions_by_magic(MAGIC_NUMBERS["STRAT1"], symbol)


def s2_positions(symbol: Optional[str] = None):
    return mt5_positions_by_magic(MAGIC_NUMBERS["STRAT2"], symbol)


def s3_positions(symbol: Optional[str] = None):
    return mt5_positions_by_magic(MAGIC_NUMBERS["STRAT3"], symbol)


# ============================================================================
# ================================  S1  ======================================
# ===================== Per-Symbol Optimizer (original) ======================
# ============================================================================

# --- S1 Constants and grids ---
s1_SYMBOLS = ["EURUSD", "AUDUSD"]
s1_TIMEFRAME = MT5.TIMEFRAME_H1
s1_BARS_HISTORY = 10000
s1_LOT = LOT
s1_MIN_SL_PIPS = 35
s1_TEST_RATIO = 0.30
s1_VAL_RATIO_WITHIN_TRAINVAL = 0.20
s1_DEFAULT_SPREAD_PIPS = {"EURUSD": 1.5, "GBPUSD": 2.0, "USDJPY": 1.6, "AUDUSD": 1.8, "USDCAD": 2.0}
s1_COMMISSION_PER_LOT_USD = 7.0
s1_GRID = {
    "horizon": [16, 24, 32],
    "tp_pips": [60, 80, 100],
    "sl_pips": [35, 40, 50],
    "prob_threshold": [0.58, 0.62, 0.66],
    "expect_trades_window": [30, 35, 40],
}
s1_FEATURES = ["r1", "rng", "gap", "run_len", "rng_surprise", "sign_entropy", "shape_rare", "z_gap", "z_rng"]


# --- S1 helper functions (with s1_ prefix) ---
def s1_pip_size(symbol: str) -> float:
    return 0.01 if "JPY" in symbol else 0.0001


def s1_pips_between(a: float, b: float, symbol: str) -> float:
    return (b - a) / s1_pip_size(symbol)


def s1_money_per_pip_per_lot(symbol: str) -> float:
    return 10.0


def s1_init_mt5():
    return True


def s1_fetch_symbol_history(symbol: str, timeframe, bars: int) -> pd.DataFrame:
    """Fetch historical data from MT5"""
    if not MT5.symbol_select(symbol, True):
        raise RuntimeError(f"Failed to select symbol {symbol}")
    rates = MT5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        raise RuntimeError(f"Failed to get data for {symbol}: {MT5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "tick_volume": "Volume"},
              inplace=True)
    df = df[["time", "Open", "High", "Low", "Close", "Volume"]].sort_values("time").reset_index(drop=True)
    return df


def s1_permutation_entropy(series, order=3, delay=1):
    """Calculate permutation entropy"""
    n = len(series)
    if n < order * delay: return np.nan
    patterns = {}
    for i in range(n - delay * (order - 1)):
        window = series[i:i + delay * order:delay]
        key = tuple(np.argsort(window))
        patterns[key] = patterns.get(key, 0) + 1
    probs = np.array(list(patterns.values()), dtype=float)
    probs /= probs.sum()
    pe = -np.sum(probs * np.log2(probs))
    return pe / np.log2(math.factorial(order))


def s1_build_features(df: pd.DataFrame, symbol: str, w: int = 48) -> pd.DataFrame:
    """Build technical features for S1 strategy"""
    out = df.copy()
    out["mid"] = (out["High"] + out["Low"]) / 2.0
    out["r1"] = out["Close"].pct_change()
    out["rng"] = (out["High"] - out["Low"]) / (out["mid"].replace(0, np.nan))
    out["gap"] = out["Open"].pct_change()

    r_sign = np.sign(out["Close"].diff().fillna(0.0).values)
    rl = np.zeros_like(r_sign)
    cur = 0
    for i in range(1, len(r_sign)):
        if r_sign[i] != 0 and r_sign[i] == r_sign[i - 1]:
            cur += 1
        else:
            cur = 0
        rl[i] = cur
    out["run_len"] = rl

    out["rng_med"] = out["rng"].rolling(w).median()
    out["rng_surprise"] = (out["rng"] - out["rng_med"]) / (out["rng_med"] + 1e-8)

    sr = pd.Series(np.sign(out["Close"].diff().fillna(0.0).values))
    out["sign_entropy"] = sr.rolling(w).apply(lambda s: s1_permutation_entropy(s.values, order=3, delay=1), raw=False)

    def rarity(vec):
        ranks = pd.Series(vec).rank().astype(int).astype(str).values
        code = "".join(ranks)
        return hash(code) % 1000

    out["shape_code"] = out["Close"].rolling(w).apply(rarity, raw=True)
    out["shape_rare"] = out["shape_code"].rolling(w * 2).apply(
        lambda s: 1.0 / (pd.Series(s).value_counts().reindex([s.iloc[-1]], fill_value=0).values[0] + 1e-6)
        if len(s) > 0 else np.nan, raw=False
    )

    out["z_gap"] = (out["gap"] - out["gap"].rolling(w).mean()) / (out["gap"].rolling(w).std() + 1e-9)
    out["z_rng"] = (out["rng"] - out["rng"].rolling(w).mean()) / (out["rng"].rolling(w).std() + 1e-9)
    out["symbol"] = symbol
    return out


def s1_meta_label_tp_sl(df: pd.DataFrame, symbol: str, horizon: int, tp_pips: int, sl_pips: int):
    """Create meta-labels based on TP/SL hitting within horizon"""
    ps = s1_pip_size(symbol)
    tp = tp_pips * ps
    sl = sl_pips * ps
    high = df["High"].values
    low = df["Low"].values
    openp = df["Open"].values

    labels = np.full(len(df), np.nan)
    for i in range(len(df) - horizon - 1):
        entry = openp[i + 1]
        tp_level = entry + tp
        sl_level = entry - sl
        hit_tp, hit_sl = None, None
        for k in range(1, horizon + 1):
            h = high[i + k];
            l = low[i + k]
            if hit_tp is None and h >= tp_level: hit_tp = i + k
            if hit_sl is None and l <= sl_level: hit_sl = i + k
            if hit_tp is not None or hit_sl is not None: break
        labels[i] = 1.0 if (hit_tp is not None and (hit_sl is None or hit_tp <= hit_sl)) else 0.0 if (
                    hit_sl is not None) else 0.0
    return labels


def s1_make_time_splits(df: pd.DataFrame):
    """Split data into train/validation/test periods"""
    N = len(df)
    test_len = int(N * s1_TEST_RATIO)
    test_start = max(N - test_len, 0)
    trainval_end = test_start
    val_len = int(trainval_end * s1_VAL_RATIO_WITHIN_TRAINVAL)
    train_end = max(trainval_end - val_len, 0)
    t0 = df["time"].iloc[0]
    t_train_end = df["time"].iloc[train_end - 1] if train_end > 0 else None
    t_val_end = df["time"].iloc[trainval_end - 1] if trainval_end > 0 else None
    t_test_end = df["time"].iloc[-1]
    print(
        f"      Periods: TRAIN [{t0} → {t_train_end}], VAL [{df['time'].iloc[train_end] if train_end < len(df) else None} → {t_val_end}], TEST [{df['time'].iloc[test_start]} → {t_test_end}]")
    return train_end, trainval_end, test_start


def s1_safe_slice_with_horizon(df: pd.DataFrame, start: int, end: int, H: int) -> pd.DataFrame:
    """Slice data with horizon buffer to avoid lookahead bias"""
    if end - start <= H + 2:
        return df.iloc[0:0].copy()
    return df.iloc[start: end - H - 1].copy()


def s1_assert_no_label_leakage(seg_df: pd.DataFrame, full_end_index: int, H: int, name: str):
    """Check for label leakage (educational)"""
    if len(seg_df) == 0: return
    print(f"      [LEAK-CHECK] {name}: samples={len(seg_df)} (H={H}) — OK")


@dataclass
class s1_Trade:
    """Trade data structure for S1"""
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int  # 1=long
    entry_price: float
    exit_price: float
    pips: float
    pnl_money: float
    reason: str


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def s1_fit_model(X, y):
    """Fit logistic regression model with scaling"""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    model.fit(X, y)
    return model


def s1_backtest_segment(test_df: pd.DataFrame, symbol: str, model, params: dict):
    """Backtest S1 strategy on a data segment"""
    H = params["horizon"];
    TP = params["tp_pips"];
    SL = params["sl_pips"]
    P0 = params["prob_threshold"];
    Nexp = params["expect_trades_window"]
    ps = s1_pip_size(symbol)
    pip_val_1lot = s1_money_per_pip_per_lot(symbol)
    spread = s1_DEFAULT_SPREAD_PIPS.get(symbol, 2.0)
    comm = s1_COMMISSION_PER_LOT_USD * s1_LOT

    proba = model.predict_proba(test_df[s1_FEATURES].values)[:, 1]
    df = test_df.copy()
    df["p_succ"] = proba

    open_position = None
    last_trades = deque(maxlen=Nexp)
    trades, equity_points = [], []
    for i in range(len(df) - H - 2):
        if len(last_trades) >= max(5, Nexp // 2):
            exp_pips = np.mean([t.pips for t in last_trades])
            adj = np.tanh(exp_pips / 100.0) * 0.05
        else:
            adj = 0.0
        thr = np.clip(P0 - adj, 0.05, 0.95)

        if open_position is not None:
            j = i
            h = df.iloc[j]["High"];
            l = df.iloc[j]["Low"]
            entry = open_position["entry_price"]
            tp_level = entry + TP * ps
            sl_level = entry - SL * ps

            exit_now, reason, exit_price = False, None, None
            if h >= tp_level:
                exit_price = tp_level;
                exit_now = True;
                reason = "TP"
            elif l <= sl_level:
                exit_price = sl_level;
                exit_now = True;
                reason = "SL"
            elif (j - open_position["entry_index"]) >= H:
                exit_price = df.iloc[j]["Close"];
                exit_now = True;
                reason = "TIMEOUT"

            if exit_now:
                raw_pips = s1_pips_between(open_position["entry_exec_price"], exit_price, symbol)
                net_pips = raw_pips - spread
                pnl_money = net_pips * pip_val_1lot * s1_LOT - comm
                trade = s1_Trade(
                    symbol=symbol,
                    entry_time=open_position["entry_time"],
                    exit_time=df.iloc[j]["time"],
                    direction=1,
                    entry_price=open_position["entry_exec_price"],
                    exit_price=exit_price,
                    pips=net_pips,
                    pnl_money=pnl_money,
                    reason=reason
                )
                trades.append(trade)
                last_trades.append(trade)
                equity_points.append((trade.exit_time, (equity_points[-1][1] if equity_points else 0.0) + pnl_money))
                open_position = None

        if open_position is None and df.iloc[i]["p_succ"] >= thr:
            entry_index = i + 1
            if entry_index < len(df):
                entry_time = df.iloc[entry_index]["time"]
                entry_price = df.iloc[entry_index]["Open"]
                entry_exec_price = entry_price + (spread * ps) / 2.0
                open_position = dict(
                    entry_index=entry_index,
                    entry_time=entry_time,
                    entry_price=entry_price,
                    entry_exec_price=entry_exec_price
                )

    if open_position is not None:
        last_idx = len(df) - 1
        last_close = df.iloc[last_idx]["Close"]
        raw_pips = s1_pips_between(open_position["entry_exec_price"], last_close, symbol)
        net_pips = raw_pips - spread
        pnl_money = net_pips * pip_val_1lot * s1_LOT - comm
        trade = s1_Trade(
            symbol=symbol,
            entry_time=open_position["entry_time"],
            exit_time=df.iloc[last_idx]["time"],
            direction=1,
            entry_price=open_position["entry_exec_price"],
            exit_price=last_close,
            pips=net_pips,
            pnl_money=pnl_money,
            reason="FORCE_CLOSE"
        )
        trades.append(trade)
        equity_points.append((trade.exit_time, (equity_points[-1][1] if equity_points else 0.0) + pnl_money))

    return trades, equity_points


def s1_calc_metrics(trades: list):
    """Calculate performance metrics for S1 trades"""
    if len(trades) == 0:
        return dict(trades=0)
    df = pd.DataFrame([t.__dict__ for t in trades]).sort_values("exit_time").reset_index(drop=True)
    total_trades = len(df)
    wins = (df["pips"] > 0).sum()
    winrate = wins / total_trades if total_trades else 0.0
    gross_profit = df.loc[df["pips"] > 0, "pnl_money"].sum()
    gross_loss = -df.loc[df["pips"] < 0, "pnl_money"].sum()
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf
    net = df["pnl_money"].sum()

    eq = df["pnl_money"].cumsum().values
    peaks = np.maximum.accumulate(eq) if len(eq) else np.array([0.0])
    dd = (peaks - eq) if len(eq) else np.array([0.0])
    maxdd = dd.max() if len(dd) > 0 else 0.0
    ret = np.diff(np.insert(eq, 0, 0.0))
    std = ret.std() + 1e-9
    sharpe = (ret.mean() - RISK_FREE) / std

    return dict(
        trades=total_trades,
        winrate=winrate,
        profit_factor=profit_factor,
        net_money=net,
        max_dd=maxdd,
        sharpe=sharpe,
        df=df
    )


def s1_print_symbol_result(symbol, params, metrics, split_info):
    """Print S1 optimization results for a symbol"""
    print(f"\n>>> {symbol} — BEST PARAMETERS (on VAL): {params}")
    print(
        f"    Periods: TRAIN {split_info['train_period']} | VAL {split_info['val_period']} | TEST {split_info['test_period']}")
    if metrics.get("trades", 0) == 0:
        print("    No trades on VAL.")
        return
    print(f"    VAL: Trades: {metrics['trades']}, WinRate: {metrics['winrate'] * 100:.2f}%, "
          f"PF: {metrics['profit_factor']:.3f}, Net: {metrics['net_money']:.2f}, "
          f"MaxDD: {metrics['max_dd']:.2f}, Sharpe: {metrics['sharpe']:.3f}")


def s1_optimize_symbol(df_all: pd.DataFrame, symbol: str):
    """Optimize S1 parameters for a symbol using grid search"""
    best_params, best_metrics = None, None
    tried = 0
    total = (len(s1_GRID["horizon"]) * len(s1_GRID["tp_pips"]) * len(s1_GRID["sl_pips"]) *
             len(s1_GRID["prob_threshold"]) * len(s1_GRID["expect_trades_window"]))
    print(f"\n[OPT:{symbol}] Combinations: {total}")

    train_end, trainval_end, test_start = s1_make_time_splits(df_all)
    split_info = dict(
        train_period=(df_all["time"].iloc[0], df_all["time"].iloc[train_end - 1] if train_end > 0 else None),
        val_period=(df_all["time"].iloc[train_end] if train_end < len(df_all) else None,
                    df_all["time"].iloc[trainval_end - 1] if trainval_end > 0 else None),
        test_period=(df_all["time"].iloc[test_start], df_all["time"].iloc[-1])
    )

    for H in s1_GRID["horizon"]:
        train_df_raw = s1_safe_slice_with_horizon(df_all, 0, train_end, H)
        val_df_raw = s1_safe_slice_with_horizon(df_all, train_end, trainval_end, H)

        s1_assert_no_label_leakage(train_df_raw, train_end, H, "TRAIN")
        s1_assert_no_label_leakage(val_df_raw, trainval_end, H, "VAL")

        if len(train_df_raw) == 0 or len(val_df_raw) == 0:
            continue

        for TP in s1_GRID["tp_pips"]:
            for SL in s1_GRID["sl_pips"]:
                if SL < s1_MIN_SL_PIPS:
                    continue
                y_full = s1_meta_label_tp_sl(df_all, symbol, H, TP, SL)
                df_lab = df_all.copy()
                df_lab["label"] = y_full

                train_df = train_df_raw.copy()
                train_df = train_df.dropna(subset=s1_FEATURES)
                train_df = train_df.assign(label=df_lab.loc[train_df.index, "label"].values)
                train_df = train_df.dropna(subset=["label"])
                y_train = train_df["label"].values
                X_train = train_df[s1_FEATURES].values
                if len(X_train) == 0 or np.unique(y_train).size < 2:
                    continue
                model = s1_fit_model(X_train, y_train)

                for P in s1_GRID["prob_threshold"]:
                    for Nexp in s1_GRID["expect_trades_window"]:
                        tried += 1
                        params = dict(horizon=H, tp_pips=TP, sl_pips=SL,
                                      prob_threshold=P, expect_trades_window=Nexp)

                        val_df = val_df_raw.copy().dropna(subset=s1_FEATURES)
                        trades_val, _ = s1_backtest_segment(val_df, symbol, model, params)
                        m = s1_calc_metrics(trades_val)

                        score = (m.get("net_money", -1e9), m.get("profit_factor", 0.0))
                        if (best_metrics is None) or (score > (best_metrics.get("net_money", -1e9),
                                                               best_metrics.get("profit_factor", 0.0))):
                            best_params = params
                            best_metrics = m

                        if tried % 20 == 0:
                            print(f"[OPT:{symbol}] Progress {tried}/{total} ...")

    s1_print_symbol_result(symbol, best_params, best_metrics if best_metrics else {}, split_info)
    return best_params, split_info


def s1_portfolio_backtest(data_by_symbol: dict, best_params_by_symbol: dict):
    """Run portfolio backtest for S1 with best parameters"""
    all_trades = []
    equity_by_symbol = {}
    print("\n[PORTFOLIO:S1] Backtesting on TEST with best parameters per pair...")

    for symbol, df_all in data_by_symbol.items():
        if symbol not in best_params_by_symbol or best_params_by_symbol[symbol] is None:
            print(f"[PORTFOLIO:S1] Skipping {symbol}: no best parameters.")
            continue
        params = best_params_by_symbol[symbol]
        H = params["horizon"]

        train_end, trainval_end, test_start = s1_make_time_splits(df_all)

        trainval_df = s1_safe_slice_with_horizon(df_all, 0, trainval_end, H).dropna(subset=s1_FEATURES)
        y_full = s1_meta_label_tp_sl(df_all, symbol, H, params["tp_pips"], params["sl_pips"])
        df_lab = df_all.copy()
        df_lab["label"] = y_full
        trainval_df = trainval_df.assign(label=df_lab.loc[trainval_df.index, "label"].values).dropna(subset=["label"])

        if len(trainval_df) == 0 or np.unique(trainval_df["label"].values).size < 2:
            print(f"[PORTFOLIO:S1] {symbol}: insufficient data for training on TRAIN+VAL.")
            continue

        model = s1_fit_model(trainval_df[s1_FEATURES].values, trainval_df["label"].values)

        test_df = df_all.iloc[test_start:].copy().dropna(subset=s1_FEATURES)
        trades, eq_points = s1_backtest_segment(test_df, symbol, model, params)
        all_trades.extend(trades)
        equity_by_symbol[symbol] = eq_points

    return all_trades, equity_by_symbol


def run_strategy1_return_unified() -> Tuple[pd.DataFrame, List[tuple]]:
    """
    Run S1 strategy and return unified format results

    Returns:
    - df_unified_s1: DataFrame with S1 trades in unified format
    - equity_curve_s1: list of (time, equity_usd) for S1 cumulative equity
    """
    print("\n=== STRATEGY 1: Per-Symbol Optimizer ===")
    data_by_symbol = {}
    for sym in s1_SYMBOLS:
        print(f"\n[MT5] Loading {sym}...")
        df = s1_fetch_symbol_history(sym, s1_TIMEFRAME, s1_BARS_HISTORY)
        df = s1_build_features(df, sym, w=48)
        data_by_symbol[sym] = df
    best_params_by_symbol = {}
    for sym in s1_SYMBOLS:
        best_params, _ = s1_optimize_symbol(data_by_symbol[sym], sym)
        best_params_by_symbol[sym] = best_params

    all_trades, eq_by_sym = s1_portfolio_backtest(data_by_symbol, best_params_by_symbol)

    # Convert to unified DF
    rows = []
    eq_points = []
    for t in all_trades:
        rows.append(dict(
            strategy="STRAT1",
            magic=MAGIC_NUMBERS["STRAT1"],
            comment="S1",
            symbol=t.symbol,
            entry_time=t.entry_time,
            exit_time=t.exit_time,
            direction=("BUY" if t.direction == 1 else "SELL"),
            entry_price=t.entry_price,
            exit_price=t.exit_price,
            pnl_pips=t.pips,  # pips already NET including spread
            pnl_usd=t.pnl_money,
            reason=t.reason
        ))
    # Cumulative equity (by time)
    if eq_by_sym:
        all_times = sorted({tt for sym in eq_by_sym for (tt, _) in eq_by_sym[sym]})
        for t in all_times:
            total = 0.0
            for sym, seq in eq_by_sym.items():
                vals = [v for (tt, v) in seq if tt <= t]
                total += (vals[-1] if vals else 0.0)
            eq_points.append((t, total))
    df_s1 = pd.DataFrame(rows)
    df_s1 = normalize_times_df(df_s1)  # Important time normalization
    eq_points = normalize_eq_points(eq_points)
    m = calc_metrics_usd(df_s1) if not df_s1.empty else {}
    print_metrics("STRAT1", m)
    return df_s1, eq_points


# ============================================================================
# ================================  S2  ======================================
# =========== Aggressive EURUSD (BOCPD + Switching + EVT + Bandit) ==========
# ============================================================================

s2_ENABLE_GBPUSD = False
s2_SYMBOLS = ["EURUSD"] + (["GBPUSD"] if s2_ENABLE_GBPUSD else [])
s2_TIMEFRAME = MT5.TIMEFRAME_H1
s2_BARS_HISTORY = 10000
s2_LOT = LOT
s2_MIN_SL_PIPS = 35
s2_RISK_FREE = 0.0
s2_TEST_RATIO = 0.30
s2_VAL_RATIO_WITHIN_TRAINVAL = 0.20
s2_DEFAULT_SPREAD_PIPS = {"EURUSD": 1.5, "GBPUSD": 2.0}
s2_COMMISSION_PER_LOT_USD = 7.0
s2_RATIO_BUCKETS = [1.2, 1.5, 2.0]
s2_EXPERT_SET = ["E1", "E2", "E3", "E4"]
s2_RECENCY_WEIGHT_HALF_LIFE_MAP = {"EURUSD": 50, "GBPUSD": 90}
s2_DECAY_TO_PRIOR = 0.990
s2_MSM_PERSISTENCE = 0.97
s2_MSM_LEARNING = 0.005
s2_PATTERN_WINDOW = 40
s2_COLD_STREAK = 3
s2_HOT_STREAK = 3
s2_AGGR_EURUSD = True
s2_BANDIT_MIN_PULLS_FOR_CONF = 4 if s2_AGGR_EURUSD else 6
s2_BANDIT_CONF_LOWER = 0.45 if s2_AGGR_EURUSD else 0.48
s2_BANDIT_CONF_K = 0.60 if s2_AGGR_EURUSD else 0.67
s2_EPSILON_BASE = 0.06
s2_ROLL_N_BASE = 28 if s2_AGGR_EURUSD else 30
s2_ROLL_PF_MIN_BASE = 0.98 if s2_AGGR_EURUSD else 1.0
s2_DAILY_STOP_USD_BASE = -150.0
s2_FEATS = ["r1", "rng", "gap", "run_len", "rv", "bpv", "rv_bpv_ratio", "sign_entropy", "shape_rare", "z_gap", "z_rng"]


def s2_pip_size(symbol: str) -> float:
    return 0.01 if "JPY" in symbol else 0.0001


def s2_pips_between(a: float, b: float, symbol: str) -> float:
    return (b - a) / s2_pip_size(symbol)


def s2_money_per_pip_per_lot(symbol: str) -> float:
    return 10.0


def s2_fetch_symbol_history(symbol: str, timeframe, bars: int) -> pd.DataFrame:
    """Fetch historical data for S2"""
    if not MT5.symbol_select(symbol, True):
        raise RuntimeError(f"Failed to select {symbol}")
    rates = MT5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        raise RuntimeError(f"Failed to get data for {symbol}: {MT5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "tick_volume": "Volume"},
              inplace=True)
    df = df[["time", "Open", "High", "Low", "Close", "Volume"]].sort_values("time").reset_index(drop=True)
    return df


def s2_permutation_entropy(series, order=3, delay=1):
    """Calculate permutation entropy for S2"""
    n = len(series)
    if n < order * delay: return np.nan
    patterns = {}
    for i in range(n - delay * (order - 1)):
        window = series[i:i + delay * order:delay]
        key = tuple(np.argsort(window))
        patterns[key] = patterns.get(key, 0) + 1
    probs = np.array(list(patterns.values()), dtype=float)
    probs /= probs.sum()
    pe = -np.sum(probs * np.log2(probs))
    return pe / np.log2(math.factorial(order))


def s2_build_features(df: pd.DataFrame, symbol: str, w: int = 48) -> pd.DataFrame:
    """Build technical features for S2 strategy"""
    out = df.copy()
    out["mid"] = (out["High"] + out["Low"]) / 2.0
    out["r1"] = out["Close"].pct_change().fillna(0.0)
    out["rng"] = ((out["High"] - out["Low"]) / out["mid"].replace(0, np.nan)).fillna(0.0)
    out["gap"] = out["Open"].pct_change().fillna(0.0)

    r_sign = np.sign(out["Close"].diff().fillna(0.0).values)
    rl = np.zeros_like(r_sign);
    cur = 0
    for i in range(1, len(r_sign)):
        if r_sign[i] != 0 and r_sign[i] == r_sign[i - 1]:
            cur += 1
        else:
            cur = 0
        rl[i] = cur
    out["run_len"] = rl

    out["abs_r"] = out["r1"].abs()
    out["rv"] = out["abs_r"].rolling(w).sum()
    out["bpv"] = (out["abs_r"] * out["abs_r"].shift(1)).rolling(w).sum()
    out["rv_bpv_ratio"] = (out["rv"] / (out["bpv"] + 1e-12)).fillna(1.0)
    sr = pd.Series(np.sign(out["Close"].diff().fillna(0.0).values))
    out["sign_entropy"] = sr.rolling(w).apply(lambda s: s2_permutation_entropy(s.values, 3, 1), raw=False)

    def rarity(vec):
        ranks = pd.Series(vec).rank().astype(int).astype(str).values
        code = "".join(ranks)
        return hash(code) % 1000

    out["shape_code"] = out["Close"].rolling(w).apply(rarity, raw=True)
    out["shape_rare"] = out["shape_code"].rolling(w * 2).apply(
        lambda s: 1.0 / (pd.Series(s).value_counts().reindex([s.iloc[-1]], fill_value=0).values[0] + 1e-6)
        if len(s) > 0 else np.nan, raw=False
    )

    out["z_gap"] = (out["gap"] - out["gap"].rolling(w).mean()) / (out["gap"].rolling(w).std() + 1e-9)
    out["z_rng"] = (out["rng"] - out["rng"].rolling(w).mean()) / (out["rng"].rolling(w).std() + 1e-9)
    out["symbol"] = symbol
    return out


class s2_BOCPDLight:
    """Bayesian Online Change Point Detection (lightweight)"""

    def __init__(self, hazard=1 / 200.0, decay=0.98):
        self.hazard = hazard;
        self.decay = decay
        self.mu = 0.0;
        self.var = 1e-4;
        self.p_change = 0.0

    def update(self, r):
        pmu, pvar = self.mu, self.var
        self.mu = self.decay * self.mu + (1 - self.decay) * r
        self.var = self.decay * self.var + (1 - self.decay) * (r - self.mu) ** 2 + 1e-8
        like_old = np.exp(-0.5 * ((r - pmu) ** 2) / (pvar + 1e-8)) / np.sqrt(2 * np.pi * (pvar + 1e-8))
        like_new = np.exp(-0.5 * ((r - 0.0) ** 2) / (10 * pvar + 1e-8)) / np.sqrt(2 * np.pi * (10 * pvar + 1e-8))
        pC = self.hazard
        num = pC * like_new;
        den = num + (1 - pC) * like_old + 1e-30
        self.p_change = num / den
        return self.p_change


def s2_emission_regime_probs(row, p_change):
    """Calculate regime emission probabilities"""
    jumpy = row["rv_bpv_ratio"] if not np.isnan(row["rv_bpv_ratio"]) else 1.0
    ent = row["sign_entropy"] if not np.isnan(row["sign_entropy"]) else 0.5
    rl = row["run_len"] if not np.isnan(row["run_len"]) else 0.0
    pturb = np.clip(0.5 * jumpy + 0.5 * p_change * 5.0, 0.0, 1.0)
    ptr = np.clip((rl / 20.0) * (1.0 - ent), 0.0, 1.0)
    pcalm = np.clip(1.0 - max(pturb, ptr), 0.0, 1.0)
    s = pcalm + ptr + pturb + 1e-12
    return np.array([pcalm / s, ptr / s, pturb / s])


class s2_MarkovSwitcher:
    """Markov Switching Model"""

    def __init__(self, persistence=s2_MSM_PERSISTENCE, lr=s2_MSM_LEARNING):
        self.A = np.full((3, 3), (1 - persistence) / 2.0)
        np.fill_diagonal(self.A, persistence)
        self.p = np.array([1 / 3, 1 / 3, 1 / 3])
        self.lr = lr

    def update(self, emission, observed=None):
        pred = self.p @ self.A
        post = pred * emission
        post = post / (post.sum() + 1e-12)
        self.p = post
        if observed is not None:
            i = np.argmax(observed);
            j = np.argmax(self.p)
            adj = np.zeros_like(self.A);
            adj[i, j] = 1.0
            self.A = (1 - self.lr) * self.A + self.lr * adj
            self.A = (self.A.T / (self.A.sum(axis=1) + 1e-12)).T
        return self.p

    def regime_name(self):
        return ["calm", "trend", "turbulent"][int(np.argmax(self.p))]


class s2_EVTPOTLight:
    """Extreme Value Theory - Peak Over Threshold (lightweight)"""

    def __init__(self, N=500, floor_pips=s2_MIN_SL_PIPS):
        self.N = N;
        self.floor = floor_pips
        self.neg = deque(maxlen=N)

    def update(self, r):
        if r < 0: self.neg.append(abs(r))

    @staticmethod
    def _hill_q(xs, q=0.995):
        xs = np.array(xs)
        if len(xs) < 30: return np.nan
        xs = np.sort(xs);
        n = len(xs)
        k = max(10, n // 5);
        tail = xs[-k:];
        u = tail[0]
        logs = np.log(tail + 1e-12) - np.log(u + 1e-12)
        alpha = 1.0 / (logs.mean() + 1e-12)
        return max(u * ((k / (max(1e-6, 1.0 - q) * n)) ** (1.0 / alpha)), u)

    def sl_pips(self, symbol, ps):
        q = self._hill_q(self.neg, 0.995)
        if np.isnan(q): return float(self.floor)
        pips_approx = 1.15 * ((q / ps) if ps < 0.001 else (q / ps))
        return float(max(self.floor, min(300.0, pips_approx)))


def s2_regime_params(symbol: str, regime: str, pattern: str, aggressive: bool):
    """Get dynamic parameters based on regime and pattern"""
    if symbol == "EURUSD":
        eps_base = s2_EPSILON_BASE
        if regime == "trend":
            eps = eps_base * (1.8 if aggressive else 1.5)
        elif regime == "calm":
            eps = eps_base * (0.7 if aggressive else 0.5)
        else:
            eps = eps_base

        if regime == "trend":
            roll_n = 22 if aggressive else 24
            pf_min = 0.92 if aggressive else 0.95
        elif regime == "calm":
            roll_n = 24 if aggressive else 30
            pf_min = 0.95 if aggressive else 1.00
        else:
            roll_n = 36
            pf_min = 1.05

        day_stop = s2_DAILY_STOP_USD_BASE
        if pattern == "cold":
            pf_min += 0.05
            day_stop = s2_DAILY_STOP_USD_BASE * 1.2

        spread_limit = 0.90 if aggressive else 0.80

        if regime == "trend":
            thr = dict(
                E1=dict(ratio_max=1.06 if aggressive else 1.05, ent_max=0.41 if aggressive else 0.40, gap_pos=True),
                E2=dict(ratio_min=1.20 if aggressive else 1.22, z_gap_abs_max=1.05 if aggressive else 1.0,
                        gap_nonneg=True),
                E3=dict(ratio_min=1.55, z_rng_min=1.1, gap_neg=True),
                E4=dict(ent_min=0.70, run_len_min=5, z_rng_max=0.50),
            )
        elif regime == "turbulent":
            thr = dict(
                E1=dict(ratio_max=1.02, ent_max=0.35, gap_pos=True),
                E2=dict(ratio_min=1.27, z_gap_abs_max=0.8, gap_nonneg=True),
                E3=dict(ratio_min=1.60, z_rng_min=1.2, gap_neg=True),
                E4=dict(ent_min=0.76, run_len_min=6, z_rng_max=0.40),
            )
        else:
            thr = dict(
                E1=dict(ratio_max=1.04 if aggressive else 1.03, ent_max=0.40 if aggressive else 0.38, gap_pos=True),
                E2=dict(ratio_min=1.22 if aggressive else 1.23, z_gap_abs_max=1.0 if aggressive else 0.9,
                        gap_nonneg=True),
                E3=dict(ratio_min=1.55, z_rng_min=1.1, gap_neg=True),
                E4=dict(ent_min=0.70 if aggressive else 0.72, run_len_min=5, z_rng_max=0.50 if aggressive else 0.45),
            )
    else:
        eps = s2_EPSILON_BASE * 0.5
        if regime == "trend":
            roll_n = 36;
            pf_min = 1.05
        elif regime == "turbulent":
            roll_n = 42;
            pf_min = 1.10
        else:
            roll_n = 32;
            pf_min = 1.05
        day_stop = s2_DAILY_STOP_USD_BASE * 1.2
        spread_limit = 0.70
        thr = dict(
            E1=dict(ratio_max=1.02, ent_max=0.35, gap_pos=True),
            E2=dict(ratio_min=1.26, z_gap_abs_max=0.8, gap_nonneg=True),
            E3=dict(ratio_min=1.62, z_rng_min=1.2, gap_neg=True),
            E4=dict(ent_min=0.76, run_len_min=6, z_rng_max=0.40),
        )

    if pattern == "hot":
        pf_min = max(0.9, pf_min - (0.06 if aggressive else 0.04))
        if "E2" in thr:
            thr["E2"]["ratio_min"] = max(1.16, thr["E2"]["ratio_min"] - (0.04 if aggressive else 0.02))
        if "E4" in thr:
            thr["E4"]["ent_min"] = max(0.66, thr["E4"]["ent_min"] - (0.03 if aggressive else 0.02))
    elif pattern == "cold":
        pf_min += (0.06 if aggressive else 0.05)
        if "E2" in thr: thr["E2"]["ratio_min"] += (0.03 if aggressive else 0.02)
        if "E3" in thr: thr["E3"]["ratio_min"] += (0.02 if aggressive else 0.02)
        if "E1" in thr: thr["E1"]["ent_max"] = min(thr["E1"]["ent_max"], 0.38)

    return dict(expert_thr=thr, epsilon=eps, roll_n=roll_n, roll_pf_min=pf_min,
                day_stop=day_stop, spread_limit=spread_limit)


def s2_expert_signal_dynamic(expert, row, TH):
    """Check if expert signal conditions are met"""
    ratio = row["rv_bpv_ratio"];
    ent = row["sign_entropy"]
    z_rng = row["z_rng"];
    z_gap = row["z_gap"];
    rl = row["run_len"];
    gap = row["gap"]
    t = TH[expert]
    if expert == "E1":
        cond = (ratio < t["ratio_max"]) and (ent < t["ent_max"])
        if t.get("gap_pos"): cond = cond and (gap > 0)
        return cond
    if expert == "E2":
        cond = (ratio > t["ratio_min"]) and (abs(z_gap) < t["z_gap_abs_max"])
        if t.get("gap_nonneg"): cond = cond and (gap >= 0)
        return cond
    if expert == "E3":
        cond = (ratio > t["ratio_min"]) and (z_rng > t["z_rng_min"])
        if t.get("gap_neg"): cond = cond and (gap < 0)
        return cond
    if expert == "E4":
        return (ent > t["ent_min"]) and (rl > t["run_len_min"]) and (z_rng < t["z_rng_max"])
    return False


def s2_classify_pattern(trades_deque):
    """Classify recent trade pattern as hot/cold/choppy"""
    if len(trades_deque) < 8:
        return "choppy"
    last = list(trades_deque)
    pnl = np.array([t["pnl_money"] for t in last])
    wins = (pnl > 0).astype(int)
    winrate = wins.mean()
    streak = 0;
    max_win_str = 0;
    max_los_str = 0
    for w in wins:
        if w == 1:
            streak = streak + 1 if streak >= 0 else 1
            max_win_str = max(max_win_str, streak)
        else:
            streak = streak - 1 if streak <= 0 else -1
            max_los_str = max(max_los_str, -streak)
    total_pnl = pnl.sum()
    if (winrate > 0.55 and max_win_str >= s2_HOT_STREAK and total_pnl > 0):
        return "hot"
    if (winrate < 0.45 and max_los_str >= s2_COLD_STREAK and total_pnl < 0):
        return "cold"
    return "choppy"


class s2_TSBanditWeighted:
    """Thompson Sampling Bandit with weighted updates"""

    def __init__(self, experts, ratios, prior=(2.0, 2.0)):
        self.prior_a, self.prior_b = prior
        self.experts = list(experts)
        self.base_ratios = list(ratios)
        self.stats = {reg: {} for reg in ["calm", "trend", "turbulent"]}
        self.allowed = {reg: set((e, r) for e in self.experts for r in self.base_ratios)
                        for reg in ["calm", "trend", "turbulent"]}
        self.epsilon = s2_EPSILON_BASE
        self._init_stats()

    def _init_stats(self):
        for reg in self.stats:
            for e in self.experts:
                for r in self.base_ratios:
                    self.stats[reg][(e, r)] = [self.prior_a, self.prior_b]

    def set_allowed(self, masks, ratios):
        for reg in self.allowed:
            allowed = set((e, rt) for e in masks[reg] for rt in ratios)
            self.allowed[reg] = allowed
            for key in allowed:
                if key not in self.stats[reg]:
                    self.stats[reg][key] = [self.prior_a, self.prior_b]

    def set_epsilon(self, eps):
        self.epsilon = float(max(0.0, min(0.5, eps)))

    def step_decay(self):
        for reg in self.stats:
            for k, (a, b) in self.stats[reg].items():
                a = self.prior_a + (a - self.prior_a) * s2_DECAY_TO_PRIOR
                b = self.prior_b + (b - self.prior_b) * s2_DECAY_TO_PRIOR
                self.stats[reg][k] = [a, b]

    def sample_arm(self, regime):
        if len(self.allowed[regime]) == 0:
            return None
        if random.random() < self.epsilon:
            return random.choice(list(self.allowed[regime]))
        best = None;
        best_val = -1
        for (e, r), (a, b) in self.stats[regime].items():
            if (e, r) not in self.allowed[regime]: continue
            p = np.random.beta(a, b)
            if p > best_val: best_val = p; best = (e, r)
        if best is None:
            return None
        a, b = self.stats[regime][best]
        pulls = (a + b) - (self.prior_a + self.prior_b)
        # RESTORED original behavior: using BANDIT_MIN_PULLS_FOR_CONF
        if pulls < s2_BANDIT_MIN_PULLS_FOR_CONF:
            return best
        post_mean = a / (a + b)
        post_std = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        lower = post_mean - s2_BANDIT_CONF_K * post_std
        if lower < s2_BANDIT_CONF_LOWER:
            return None
        return best

    def update(self, regime, arm, success, weight=1.0):
        if arm is None: return
        a, b = self.stats[regime].get(arm, [self.prior_a, self.prior_b])
        a += weight if success else 0.0
        b += 0.0 if success else weight
        self.stats[regime][arm] = [a, b]


def s2_recency_weight(n_from_end, half_life):
    """Calculate recency weight using half-life decay"""
    return 0.5 ** (n_from_end / max(1, half_life))


def s2_make_time_splits(df: pd.DataFrame):
    """Split data into train/validation/test periods for S2"""
    N = len(df)
    test_len = int(N * s2_TEST_RATIO)
    test_start = max(N - test_len, 0)
    trainval_end = test_start
    val_len = int(trainval_end * s2_VAL_RATIO_WITHIN_TRAINVAL)
    train_end = max(trainval_end - val_len, 0)
    t0 = df["time"].iloc[0]
    t_train_end = df["time"].iloc[train_end - 1] if train_end > 0 else None
    t_val_end = df["time"].iloc[trainval_end - 1] if trainval_end > 0 else None
    t_test_end = df["time"].iloc[-1]
    print(
        f"      Periods: TRAIN [{t0} → {t_train_end}], VAL [{df['time'].iloc[train_end] if train_end < len(df) else None} → {t_val_end}], TEST [{df['time'].iloc[test_start]} → {t_test_end}]")
    return train_end, trainval_end, test_start


def s2_safe_slice_with_horizon(df: pd.DataFrame, start: int, end: int, H: int) -> pd.DataFrame:
    """Safe slice with horizon buffer"""
    if end - start <= H + 2: return df.iloc[0:0].copy()
    return df.iloc[start: end - H - 1].copy()


@dataclass
class s2_Trade:
    """Trade data structure for S2"""
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    entry_price: float
    exit_price: float
    sl_pips: float
    tp_pips: float
    ratio: float
    expert: str
    regime: str
    success: int
    pnl_pips: float
    pnl_money: float
    reason: str


def s2_rolling_pf_ok(trades_deque, roll_n, pf_min):
    """Check if rolling profit factor meets minimum requirement"""
    if len(trades_deque) < roll_n:
        return True
    pnl = np.array([t["pnl_money"] for t in list(trades_deque)[-roll_n:]])
    gp = pnl[pnl > 0].sum();
    gl = -pnl[pnl < 0].sum()
    pf = gp / gl if gl > 0 else np.inf
    return pf >= pf_min


def s2_run_segment_with_switching(df: pd.DataFrame, symbol: str, H: int,
                                  base_hazard=1 / 200.0,
                                  init_prior=(2.0, 2.0),
                                  aggressive_symbol=False):
    """Run S2 strategy with regime switching on a data segment"""
    ps = s2_pip_size(symbol)
    spread = s2_DEFAULT_SPREAD_PIPS.get(symbol, 2.0)
    pip_val = s2_money_per_pip_per_lot(symbol)
    comm = s2_COMMISSION_PER_LOT_USD * s2_LOT
    half_life = s2_RECENCY_WEIGHT_HALF_LIFE_MAP.get(symbol, 80)

    bocpd = s2_BOCPDLight(hazard=base_hazard, decay=0.98)
    evt = s2_EVTPOTLight(N=500, floor_pips=s2_MIN_SL_PIPS)
    msm = s2_MarkovSwitcher(persistence=s2_MSM_PERSISTENCE, lr=s2_MSM_LEARNING)
    bandit = s2_TSBanditWeighted(experts=s2_EXPERT_SET, ratios=s2_RATIO_BUCKETS, prior=init_prior)

    last_trades = deque(maxlen=s2_PATTERN_WINDOW)
    reasons = defaultdict(int)
    open_pos = None
    trades = []
    day_pnl = 0.0;
    cur_day = None

    for i in range(len(df) - H - 2):
        row = df.iloc[i]
        r = row["r1"]

        bar_day = row["time"].date()
        if cur_day is None:
            cur_day = bar_day
        elif bar_day != cur_day:
            day_pnl = 0.0; cur_day = bar_day

        pattern = s2_classify_pattern(last_trades)
        p_change = bocpd.update(r)
        emission = s2_emission_regime_probs(row, p_change)
        p_reg = msm.update(emission)
        regime = ["calm", "trend", "turbulent"][int(np.argmax(p_reg))]

        dyn = s2_regime_params(symbol, regime, pattern, aggressive_symbol)
        bandit.set_epsilon(dyn["epsilon"])

        if regime == "trend":
            masks = {"calm": ["E4"], "trend": ["E1", "E2"], "turbulent": ["E2"]}
            ratios = [1.2, 1.5, 2.0]
        elif regime == "turbulent":
            masks = {"calm": ["E4"], "trend": ["E4"], "turbulent": ["E2", "E3"]}
            ratios = [1.2, 1.5]
        else:
            masks = {"calm": ["E4"], "trend": ["E1", "E2"], "turbulent": ["E2", "E3"]}
            ratios = [1.2, 1.5]
        bandit.set_allowed(masks, ratios)

        evt.update(r)
        dyn_sl = evt.sl_pips(symbol, ps)
        if np.isnan(dyn_sl) or dyn_sl < s2_MIN_SL_PIPS:
            dyn_sl = float(s2_MIN_SL_PIPS)

        bandit.step_decay()
        arm = bandit.sample_arm(regime)
        expert_name, ratio = (arm if arm else (None, None))
        if arm is None:
            reasons["no_confidence_or_no_arm"] += 1

        if open_pos is not None:
            j = i
            high = df.iloc[j]["High"];
            low = df.iloc[j]["Low"]
            tp_level = open_pos["entry_px"] + open_pos["tp_pips"] * ps
            sl_level = open_pos["entry_px"] - open_pos["sl_pips"] * ps
            exit_now = False;
            reason = None;
            exit_px = None
            if high >= tp_level:
                exit_px = tp_level; exit_now = True; reason = "TP"
            elif low <= sl_level:
                exit_px = sl_level; exit_now = True; reason = "SL"
            elif (j - open_pos["entry_idx"]) >= H:
                exit_px = df.iloc[j]["Close"]; exit_now = True; reason = "TIMEOUT"
            if exit_now:
                raw_pips = s2_pips_between(open_pos["entry_exec_px"], exit_px, symbol)
                net_pips = raw_pips - spread
                pnl_money = net_pips * pip_val * s2_LOT - comm
                success = 1 if reason == "TP" else 0
                tr = s2_Trade(
                    symbol=symbol,
                    entry_time=open_pos["entry_time"], exit_time=df.iloc[j]["time"],
                    direction=1, entry_price=open_pos["entry_exec_px"], exit_price=exit_px,
                    sl_pips=open_pos["sl_pips"], tp_pips=open_pos["tp_pips"], ratio=open_pos["ratio"],
                    expert=open_pos["expert"], regime=open_pos["regime"], success=success,
                    pnl_pips=net_pips, pnl_money=pnl_money, reason=reason
                )
                trades.append(tr)
                w = s2_recency_weight(0, half_life)
                bandit.update(open_pos["regime"], (open_pos["expert"], open_pos["ratio"]), success, weight=w)
                last_trades.append({"pnl_money": pnl_money, "success": success})
                day_pnl += pnl_money
                open_pos = None

        if open_pos is None:
            blocked = False
            if day_pnl <= dyn["day_stop"]:
                reasons["day_stop"] += 1;
                blocked = True
            if not blocked and not s2_rolling_pf_ok(last_trades, dyn["roll_n"], dyn["roll_pf_min"]):
                reasons["anti_chop_pf"] += 1;
                blocked = True
            if not blocked and expert_name is None:
                reasons["no_confidence_or_no_arm"] += 1;
                blocked = True
            if blocked:
                continue

            TH = dyn["expert_thr"]
            if expert_name and s2_expert_signal_dynamic(expert_name, row, TH):
                entry_idx = i + 1
                if entry_idx < len(df):
                    entry_time = df.iloc[entry_idx]["time"]
                    entry_px = df.iloc[entry_idx]["Open"]
                    entry_exec = entry_px + (s2_DEFAULT_SPREAD_PIPS.get(symbol, 2.0) * ps) / 2.0
                    sl_pips = float(max(s2_MIN_SL_PIPS, dyn_sl))
                    if s2_DEFAULT_SPREAD_PIPS.get(symbol, 2.0) > dyn["spread_limit"] * sl_pips:
                        reasons["too_costly_spread_vs_SL"] += 1
                        continue
                    tp_pips = float(max(sl_pips * ratio, sl_pips + 5.0))
                    open_pos = dict(
                        entry_idx=entry_idx, entry_time=entry_time,
                        entry_px=entry_px, entry_exec_px=entry_exec,
                        sl_pips=sl_pips, tp_pips=tp_pips,
                        expert=expert_name, ratio=ratio, regime=regime
                    )
            else:
                reasons["expert_no_signal"] += 1

    if open_pos is not None:
        last_idx = len(df) - 1
        last_close = df.iloc[last_idx]["Close"]
        raw_pips = s2_pips_between(open_pos["entry_exec_px"], last_close, symbol)
        net_pips = raw_pips - s2_DEFAULT_SPREAD_PIPS.get(symbol, 2.0)
        pip_val = s2_money_per_pip_per_lot(symbol)
        comm = s2_COMMISSION_PER_LOT_USD * s2_LOT
        pnl_money = net_pips * pip_val * s2_LOT - comm
        success = int(net_pips > 0)
        tr = s2_Trade(
            symbol=symbol,
            entry_time=open_pos["entry_time"], exit_time=df.iloc[last_idx]["time"],
            direction=1, entry_price=open_pos["entry_exec_px"], exit_price=last_close,
            sl_pips=open_pos["sl_pips"], tp_pips=open_pos["tp_pips"], ratio=open_pos["ratio"],
            expert=open_pos["expert"], regime=open_pos["regime"], success=success,
            pnl_pips=net_pips, pnl_money=pnl_money, reason="FORCE_CLOSE"
        )
        trades.append(tr)

    if reasons:
        print("  [DEBUG] Reasons for missed entries:", dict(reasons))
    return trades


def s2_make_time_splits_and_sets(df, H):
    """Create time splits with horizon consideration"""
    train_end, trainval_end, test_start = s2_make_time_splits(df)
    train_df = s2_safe_slice_with_horizon(df, 0, train_end, H).dropna(subset=s2_FEATS)
    val_df = s2_safe_slice_with_horizon(df, train_end, trainval_end, H).dropna(subset=s2_FEATS)
    test_df = df.iloc[test_start:].dropna(subset=s2_FEATS)
    print(
        f"      [LEAK-CHECK] TRAIN samples={len(train_df)}, VAL samples={len(val_df)}, TEST samples={len(test_df)} (H={H}) — OK")
    return train_df, val_df, test_df


def s2_validate_symbol(val_df: pd.DataFrame, symbol: str, H: int, base_hazard_grid, prior_grid, aggressive_symbol):
    """Validate S2 parameters on validation set"""
    best = None;
    bestm = None

    def calc_metrics(trades: list):
        if len(trades) == 0: return dict(trades=0)
        df = pd.DataFrame([t.__dict__ for t in trades]).sort_values("exit_time").reset_index(drop=True)
        total = len(df);
        wins = (df["pnl_pips"] > 0).sum()
        winrate = wins / total if total else 0.0
        gp = df.loc[df["pnl_pips"] > 0, "pnl_money"].sum()
        gl = -df.loc[df["pnl_pips"] < 0, "pnl_money"].sum()
        pf = gp / gl if gl > 0 else np.inf
        net = df["pnl_money"].sum()
        eq = df["pnl_money"].cumsum().values
        peaks = np.maximum.accumulate(eq) if len(eq) else np.array([0.0])
        dd = (peaks - eq) if len(eq) else np.array([0.0])
        maxdd = dd.max() if len(dd) > 0 else 0.0
        ret = np.diff(np.insert(eq, 0, 0.0))
        sharpe = (ret.mean() - s2_RISK_FREE) / (ret.std() + 1e-9)
        return dict(trades=total, winrate=winrate, profit_factor=pf, net_money=net, max_dd=maxdd, sharpe=sharpe, df=df)

    for hz in base_hazard_grid:
        for prior in prior_grid:
            trades = s2_run_segment_with_switching(val_df, symbol, H, base_hazard=hz, init_prior=prior,
                                                   aggressive_symbol=aggressive_symbol)
            m = calc_metrics(trades)
            score = (m.get("net_money", -1e9), m.get("profit_factor", 0.0))
            if (bestm is None) or (score > (bestm.get("net_money", -1e9), bestm.get("profit_factor", 0.0))):
                best = {"hazard": hz, "prior": prior};
                bestm = m
    return best, bestm


def run_strategy2_return_unified() -> Tuple[pd.DataFrame, List[tuple]]:
    """
    Run S2 strategy and return unified format results

    Returns:
    - df_unified_s2: DataFrame with S2 trades in unified format
    - equity_curve_s2: list of (time, equity_usd) for S2 cumulative equity
    """
    print("\n=== STRATEGY 2: Aggressive EURUSD (H1) ===")
    data_by_symbol = {}
    for sym in s2_SYMBOLS:
        print(f"\n[MT5] Loading {sym}...")
        df = s2_fetch_symbol_history(sym, s2_TIMEFRAME, s2_BARS_HISTORY)
        df = s2_build_features(df, sym, w=48)
        data_by_symbol[sym] = df

    rows = [];
    equity_points = []
    for sym in s2_SYMBOLS:
        print(f"\n===== {sym} =====")
        df = data_by_symbol[sym].copy()
        H = 32
        train_df, val_df, test_df = s2_make_time_splits_and_sets(df, H)
        best_cfg, mval = s2_validate_symbol(
            val_df, sym, H,
            base_hazard_grid=[1 / 400.0, 1 / 300.0, 1 / 200.0, 1 / 150.0],
            prior_grid=[(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
            aggressive_symbol=(s2_AGGR_EURUSD and sym == "EURUSD")
        )
        print(f">>> {sym} — best configuration (VAL): H={H}, hazard={best_cfg['hazard']}, prior={best_cfg['prior']}")
        test_tr = s2_run_segment_with_switching(test_df, sym, H,
                                                base_hazard=best_cfg["hazard"],
                                                init_prior=best_cfg["prior"],
                                                aggressive_symbol=(s2_AGGR_EURUSD and sym == "EURUSD"))
        if test_tr:
            eq = 0.0
            for t in sorted(test_tr, key=lambda x: x.exit_time):
                rows.append(dict(
                    strategy="STRAT2",
                    magic=MAGIC_NUMBERS["STRAT2"],
                    comment="S2",
                    symbol=t.symbol,
                    entry_time=t.entry_time,
                    exit_time=t.exit_time,
                    direction="BUY",
                    entry_price=t.entry_price,
                    exit_price=t.exit_price,
                    pnl_pips=t.pnl_pips,
                    pnl_usd=t.pnl_money,
                    reason=t.reason
                ))
                eq += t.pnl_money
                equity_points.append((t.exit_time, eq))
    df_s2 = pd.DataFrame(rows)
    df_s2 = normalize_times_df(df_s2)  # Time normalization
    equity_points = normalize_eq_points(equity_points)
    m = calc_metrics_usd(df_s2) if not df_s2.empty else {}
    print_metrics("STRAT2", m)
    return df_s2, equity_points


# ============================================================================
# ================================  S3  ======================================
# ===== TS2Vec-style Embeddings + Calibrated Classifier (original) ==========
# ============================================================================

@dataclass
class s3_Config:
    """Configuration for S3 strategy"""
    symbol: str = "EURUSD"
    timeframe: int = mt5.TIMEFRAME_M15
    bars: int = 36000
    horizon: int = 17
    eps_label: float = 0.0
    seq_len: int = 64
    emb_dim: int = 64
    encoder_epochs: int = 8
    encoder_lr: float = 1e-3
    batch_size: int = 512
    n_splits: int = 3
    prob_buy: float = 0.60
    prob_sell: float = 0.40
    no_trade_band: float = 0.05
    ema_alpha: float = 0.20
    consec_required: int = 2
    tp_mult: float = 2.8
    sl_mult: float = 1.0
    tp_min: int = 80
    tp_max: int = 300
    sl_min: int = 25
    sl_max: int = 90
    spread: float = 0.00015
    commission: float = 0.00005
    random_state: int = 42
    session_start_utc: int = 7
    session_end_utc: int = 21
    opp_exit_thr: float = 0.75
    trail_prob_drop: float = 0.12
    cooldown_bars: int = 8
    max_trades_per_day: int = 8
    max_loss_pips_per_day: int = 80
    val_frac: float = 0.15
    out_dir: str = "outputs_ts2vec_v3"


s3_CFG = s3_Config()


def s3_log(msg: str):
    """Logging function for S3"""
    print(f"[INFO] {msg}", flush=True)


def s3_load_mt5_data(symbol: str, timeframe: int, bars: int) -> pd.DataFrame:
    """Load data from MT5 for S3"""
    s3_log(f"Connected to MT5. Loading {bars} bars of {symbol} @ {timeframe}…")
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError("Failed to load quotes from MT5")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.rename(columns={"tick_volume": "vol"}, inplace=True)
    df = df.sort_values('time').reset_index(drop=True)
    s3_log(f"Got {len(df):,} bars. Range: {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
    return df


def s3_build_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build relative features for S3"""
    d = df.copy()
    d['ret'] = np.log(d['close']).diff()
    prev_c = d['close'].shift(1)
    for col in ['open', 'high', 'low', 'close']:
        d[f'{col}_rel'] = (d[col] - prev_c) / prev_c
    d['range_rel'] = (d['high'] - d['low']) / prev_c
    d['hlc_spread_rel'] = (d['high'] + d['low'] - 2 * d['close']) / prev_c
    d['vol_rel'] = d['vol'].pct_change().fillna(0.0)
    d = d.dropna().reset_index(drop=True)
    return d


def s3_true_range_pips(df: pd.DataFrame) -> np.ndarray:
    """Calculate True Range in pips"""
    pip = 0.0001
    close_prev = df['close'].shift(1)
    up = np.maximum(df['high'], close_prev)
    down = np.minimum(df['low'], close_prev)
    tr = (up - down) / pip
    return tr.fillna(tr.median()).to_numpy()


def s3_rolling_median_tr_pips(tr_pips: np.ndarray, win: int = 48) -> np.ndarray:
    """Calculate rolling median of True Range"""
    out = np.zeros_like(tr_pips)
    for i in range(len(tr_pips)):
        s = max(0, i - win + 1)
        out[i] = np.median(tr_pips[s:i + 1])
    return out


def s3_jitter(x: np.ndarray, sigma=0.01):
    """Add jitter noise for data augmentation"""
    return x + np.random.normal(0, sigma, x.shape).astype(np.float32)


def s3_scaling(x: np.ndarray, sigma=0.1, per_feature: bool = False):
    """Apply random scaling for data augmentation"""
    if per_feature:
        factor = np.random.normal(1.0, sigma, (x.shape[0], 1, x.shape[2])).astype(np.float32)
    else:
        factor = np.random.normal(1.0, sigma, (x.shape[0], 1, 1)).astype(np.float32)
    return x * factor


def s3_time_mask(x: np.ndarray, max_frac=0.2):
    """Apply time masking for data augmentation"""
    x = x.copy()
    L = x.shape[1]
    w = np.random.randint(1, max(2, int(L * max_frac) + 1))
    start = np.random.randint(0, L - w + 1)
    x[:, start:start + w, :] = 0.0
    return x


import torch
import torch.nn as nn
import torch.optim as optim


class s3_TSEncoder(nn.Module):
    """TS2Vec-style encoder for time series"""

    def __init__(self, in_dim, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, emb_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):  # (B, L, C)
        x = x.transpose(1, 2)
        z = self.net(x)
        z = z.squeeze(-1)
        return z


def s3_nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2):
    """NT-Xent contrastive loss"""
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T)
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim[~mask].view(2 * B, 2 * B - 1)
    pos = torch.sum(z1 * z2, dim=1)
    pos = torch.cat([pos, pos], dim=0)
    loss = -torch.log(torch.exp(pos / tau) / torch.sum(torch.exp(sim / tau), dim=1)).mean()
    return loss


def s3_make_windows(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Create sliding windows from time series"""
    T, C = X.shape
    N = T - seq_len + 1
    assert N > 0, "seq_len too large for current sample"
    out = np.zeros((N, seq_len, C), dtype=np.float32)
    for i in range(N):
        out[i] = X[i:i + seq_len]
    return out


def s3_train_encoder(X_train_win: np.ndarray, in_dim: int, emb_dim: int, epochs: int, lr: float, batch: int,
                     device='cpu') -> s3_TSEncoder:
    """Train TS2Vec encoder"""
    model = s3_TSEncoder(in_dim, emb_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train_win))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for (x,) in dl:
            x = x.to(device)
            x_np = x.detach().cpu().numpy()
            x1 = torch.from_numpy(s3_time_mask(s3_scaling(s3_jitter(x_np), sigma=0.01, per_feature=False))).to(device)
            x2 = torch.from_numpy(s3_time_mask(s3_scaling(s3_jitter(x_np), sigma=0.01, per_feature=True))).to(device)
            z1 = model(x1);
            z2 = model(x2)
            loss = s3_nt_xent_loss(z1, z2, tau=0.2)
            opt.zero_grad();
            loss.backward();
            opt.step()
            losses.append(loss.item())
        s3_log(f"[ENC] Epoch {ep:02d} | Loss={np.mean(losses):.4f}")
    return model


def s3_infer_embeddings(model: s3_TSEncoder, X_win: np.ndarray, device='cpu') -> np.ndarray:
    """Infer embeddings from trained encoder"""
    model.eval();
    embs = []
    with torch.no_grad():
        for i in range(0, len(X_win), 2048):
            x = torch.from_numpy(X_win[i:i + 2048]).to(device)
            z = model(x);
            embs.append(z.cpu().numpy())
    return np.concatenate(embs, axis=0)


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier

    s3_HAS_XGB = True
except Exception:
    s3_HAS_XGB = False


def s3_make_base_clf(random_state=42):
    """Create base classifier (XGBoost or RandomForest)"""
    if s3_HAS_XGB:
        return XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=0
        )
    else:
        s3_log("XGBoost not found — using RandomForestClassifier")
        return RandomForestClassifier(
            n_estimators=600,
            max_depth=6,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1
        )


@dataclass
class s3_Trade:
    """Trade data structure for S3"""
    entry_time: pd.Timestamp
    direction: str
    entry_price: float
    tp: float
    sl: float
    exit_time: pd.Timestamp
    exit_price: float
    pnl_pips: float


def s3_ema_series(x: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential Moving Average"""
    y = np.zeros_like(x);
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def s3_pick_thresholds_grid(proba: np.ndarray, y: np.ndarray, val_frac: float,
                            pbuy_grid, psell_grid, band_grid, alpha_grid,
                            consec_required: int,
                            min_signals: int = 150) -> Tuple[
    Optional[float], Optional[float], Optional[float], Optional[float], bool, float]:
    """Grid search for optimal probability thresholds"""

    def eval_with_proba(praw):
        n = len(praw);
        val_n = max(int(n * val_frac), 1000);
        s = n - val_n
        p = praw[s:];
        yv = y[s:].astype(int)
        if len(p) < min_signals:
            return (None, None, None, None, -1e12)
        res_best = (None, None, None, None, -1e12)
        for pb in pbuy_grid:
            for ps in psell_grid:
                for band in band_grid:
                    for alpha in alpha_grid:
                        ps_s = s3_ema_series(p, alpha)
                        slope = np.zeros_like(ps_s);
                        slope[1:] = ps_s[1:] - ps_s[:-1]
                        consec_up = 0;
                        consec_dn = 0
                        pred = np.full_like(yv, 2)
                        for i in range(len(p)):
                            if abs(ps_s[i] - 0.5) < band:
                                consec_up = 0;
                                consec_dn = 0
                                continue
                            if ps_s[i] >= pb and slope[i] > 0:
                                consec_up += 1
                            else:
                                consec_up = 0
                            if ps_s[i] <= ps and slope[i] < 0:
                                consec_dn += 1
                            else:
                                consec_dn = 0
                            if consec_up >= consec_required:
                                pred[i] = 1
                            elif consec_dn >= consec_required:
                                pred[i] = 0
                        mask = pred != 2
                        if mask.sum() < min_signals:
                            continue
                        f1m = f1_score(yv[mask], pred[mask], average='macro')
                        signals_ratio = mask.mean()
                        score = f1m - 0.05 * signals_ratio
                        if score > res_best[4]:
                            res_best = (pb, ps, band, alpha, score)
        return res_best

    best_dir = eval_with_proba(proba)
    best_inv = eval_with_proba(1.0 - proba)
    if best_dir[0] is None and best_inv[0] is None:
        return None, None, None, None, False, -1e12
    if best_inv[4] > best_dir[4]:
        return best_inv[0], best_inv[1], best_inv[2], best_inv[3], True, best_inv[4]
    else:
        return best_dir[0], best_dir[1], best_dir[2], best_dir[3], False, best_dir[4]


def s3_backtest(df: pd.DataFrame,
                preds_proba: np.ndarray,
                prob_buy: float,
                prob_sell: float,
                no_trade_band: float,
                ema_alpha: float,
                consec_required: int,
                opp_exit_thr: float,
                tr_median_pips: np.ndarray,
                tp_mult: float, sl_mult: float,
                tp_min: int, tp_max: int, sl_min: int, sl_max: int,
                spread: float,
                commission: float,
                cooldown_bars: int,
                session_start_utc: int,
                session_end_utc: int,
                trail_prob_drop: float,
                max_trades_per_day: int,
                max_loss_pips_per_day: int,
                start_idx: int = 0) -> Tuple[List['s3_Trade'], float, np.ndarray]:
    """Backtest S3 strategy"""

    pip = 0.0001
    proba_s = s3_ema_series(preds_proba, ema_alpha)
    slope = np.zeros_like(proba_s);
    slope[1:] = proba_s[1:] - proba_s[:-1]
    trades: List[s3_Trade] = [];
    equity = [];
    cum_pips = 0.0
    in_pos = False;
    pos_dir = None;
    entry_price = None;
    entry_time = None
    tp_level = None;
    sl_level = None;
    cooldown = 0;
    p_peak = None
    consec_up = 0;
    consec_dn = 0
    current_day = None;
    trades_today = 0;
    loss_today = 0.0
    i = start_idx;
    n = len(df)
    while i < n - 1:
        t_cur = pd.to_datetime(df['time'].iloc[i])
        day = t_cur.date();
        hour = t_cur.hour
        if current_day != day:
            current_day = day;
            trades_today = 0;
            loss_today = 0.0
        p = proba_s[i];
        decision = "HOLD"
        allow_session = (session_start_utc <= hour <= session_end_utc)

        if not allow_session or trades_today >= max_trades_per_day or loss_today <= -max_loss_pips_per_day:
            decision = "HOLD";
            consec_up = 0;
            consec_dn = 0
        else:
            if abs(p - 0.5) < no_trade_band:
                consec_up = 0;
                consec_dn = 0
            else:
                if p >= prob_buy and slope[i] > 0:
                    consec_up += 1
                else:
                    consec_up = 0
                if p <= prob_sell and slope[i] < 0:
                    consec_dn += 1
                else:
                    consec_dn = 0
                if consec_up >= consec_required:
                    decision = "BUY"
                elif consec_dn >= consec_required:
                    decision = "SELL"
                else:
                    decision = "HOLD"
        med_tr = tr_median_pips[i]
        tp_pips = int(np.clip(tp_mult * med_tr, tp_min, tp_max))
        sl_pips = int(np.clip(sl_mult * med_tr, sl_min, sl_max))

        if not in_pos and cooldown == 0:
            if decision == "BUY":
                entry_time = df['time'].iloc[i + 1]
                raw_entry = df['open'].iloc[i + 1]
                entry_price = raw_entry + spread / 2.0
                tp_level = entry_price + tp_pips * pip
                sl_level = entry_price - sl_pips * pip
                pos_dir = "BUY";
                in_pos = True;
                p_peak = p;
                consec_up = 0
            elif decision == "SELL":
                entry_time = df['time'].iloc[i + 1]
                raw_entry = df['open'].iloc[i + 1]
                entry_price = raw_entry - spread / 2.0
                tp_level = entry_price - tp_pips * pip
                sl_level = entry_price + sl_pips * pip
                pos_dir = "SELL";
                in_pos = True;
                p_peak = 1 - p;
                consec_dn = 0
        elif in_pos:
            hi = df['high'].iloc[i];
            lo = df['low'].iloc[i];
            t = df['time'].iloc[i];
            closed = False
            if pos_dir == "BUY":
                if p > p_peak: p_peak = p
            else:
                opp_p = 1 - p
                if opp_p > p_peak: p_peak = opp_p

            strong_opp = (pos_dir == "BUY" and p <= 1 - opp_exit_thr) or (pos_dir == "SELL" and p >= opp_exit_thr)
            trail_exit = False
            if pos_dir == "BUY":
                if df['close'].iloc[i] > entry_price and (p_peak - p) >= s3_CFG.trail_prob_drop:
                    trail_exit = True
            else:
                if df['close'].iloc[i] < entry_price and (p_peak - (1 - p)) >= s3_CFG.trail_prob_drop:
                    trail_exit = True

            if pos_dir == "BUY":
                hit_tp = hi >= tp_level;
                hit_sl = lo <= sl_level
                if hit_tp or hit_sl or strong_opp or trail_exit:
                    if hit_tp:
                        exit_price = tp_level - spread / 2.0
                    elif hit_sl:
                        exit_price = sl_level - spread / 2.0
                    else:
                        exit_price = df['close'].iloc[i] - spread / 2.0
                    pnl = (exit_price - entry_price) / pip - (s3_CFG.commission / pip)
                    trades.append(s3_Trade(entry_time, pos_dir, entry_price, tp_level, sl_level, t, exit_price, pnl))
                    in_pos = False;
                    pos_dir = None;
                    closed = True
            else:
                hit_tp = lo <= tp_level;
                hit_sl = hi >= sl_level
                if hit_tp or hit_sl or strong_opp or trail_exit:
                    if hit_tp:
                        exit_price = tp_level + spread / 2.0
                    elif hit_sl:
                        exit_price = sl_level + spread / 2.0
                    else:
                        exit_price = df['close'].iloc[i] + spread / 2.0
                    pnl = (entry_price - exit_price) / pip - (s3_CFG.commission / pip)
                    trades.append(s3_Trade(entry_time, pos_dir, entry_price, tp_level, sl_level, t, exit_price, pnl))
                    in_pos = False;
                    pos_dir = None;
                    closed = True
            if closed:
                trades_today += 1
                loss_today += trades[-1].pnl_pips
                if trades[-1].pnl_pips < 0:
                    cooldown = s3_CFG.cooldown_bars
                p_peak = None

        if trades:
            cum_pips = np.sum([tr.pnl_pips for tr in trades])
        equity.append(cum_pips)
        if cooldown > 0:
            cooldown -= 1
        i += 1

    if in_pos:
        t = df['time'].iloc[-1]
        last = df['close'].iloc[-1]
        if pos_dir == "BUY":
            exit_price = last - s3_CFG.spread / 2.0
            pnl = (exit_price - entry_price) / 0.0001 - (s3_CFG.commission / 0.0001)
        else:
            exit_price = last + s3_CFG.spread / 2.0
            pnl = (entry_price - exit_price) / 0.0001 - (s3_CFG.commission / 0.0001)
        trades.append(s3_Trade(entry_time, pos_dir, entry_price, tp_level, sl_level, t, exit_price, pnl))
        equity.append(np.sum([tr.pnl_pips for tr in trades]))

    total_pnl = sum(t.pnl_pips for t in trades)
    return trades, total_pnl, np.array(equity)


def run_strategy3_return_unified(cfg: s3_Config = s3_CFG) -> Tuple[pd.DataFrame, List[tuple]]:
    """
    Run S3 strategy and return unified format results

    Returns:
    - df_unified_s3: DataFrame with S3 trades (converted to USD via pips → USD with LOT=0.1)
    - equity_curve_s3: list of (time, equity_usd)
    """
    np.random.seed(cfg.random_state);
    torch.manual_seed(cfg.random_state)
    ensure_dir(cfg.out_dir)
    print("\n=== STRATEGY 3: TS2Vec + Classifier (M15, EURUSD) ===")

    # 1) Data
    df_raw = s3_load_mt5_data(cfg.symbol, cfg.timeframe, cfg.bars)

    # 2) Features
    d = s3_build_relative_features(df_raw)
    feat_cols = [c for c in d.columns if c.endswith('_rel') or c in ['ret', 'range_rel', 'hlc_spread_rel', 'vol_rel']]
    X_all = d[feat_cols].to_numpy().astype(np.float32)
    times_all = d['time'].to_numpy()
    close_all = df_raw.loc[d.index, 'close'].to_numpy()

    # 3) Anti-leak preparation
    T = len(X_all);
    H = cfg.horizon
    assert T > H + cfg.seq_len, "Series too short for selected horizon and seq_len"
    X_all = X_all[:T - H];
    times_all = times_all[:T - H];
    close_ref = close_all[:T - H]
    future = close_all[H:];
    ret_h = (future - close_ref) / close_ref
    y_all = np.where(ret_h > cfg.eps_label, 1, np.where(ret_h < -cfg.eps_label, 0, np.nan))
    mask = ~np.isnan(y_all)
    X_all = X_all[mask];
    y_all = y_all[mask];
    times_all = times_all[mask];
    close_ref = close_ref[mask]

    # 4) Time series split
    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)
    tr_i, te_i = list(tscv.split(X_all))[-1]
    X_train_raw, X_test_raw = X_all[tr_i], X_all[te_i]
    y_train, y_test = y_all[tr_i], y_all[te_i]
    t_train, t_test = times_all[tr_i], times_all[te_i]
    s3_log(f"Split (TimeSeriesSplit={cfg.n_splits}): train={len(tr_i)} test={len(te_i)}")

    # 5) Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw);
    X_test = scaler.transform(X_test_raw)

    # 6) Windows
    idx_shift = cfg.seq_len - 1
    X_train_win = s3_make_windows(X_train, cfg.seq_len);
    X_test_win = s3_make_windows(X_test, cfg.seq_len)
    y_train_w = y_train[idx_shift:].astype(int);
    y_test_w = y_test[idx_shift:].astype(int)
    t_train_w = t_train[idx_shift:];
    t_test_w = t_test[idx_shift:]

    # 7) Encoder training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    in_dim = X_train.shape[1]
    encoder = s3_train_encoder(
        X_train_win, in_dim=in_dim, emb_dim=cfg.emb_dim,
        epochs=cfg.encoder_epochs, lr=cfg.encoder_lr, batch=cfg.batch_size, device=device
    )

    # 8) Embeddings
    Z_train = s3_infer_embeddings(encoder, X_train_win, device=device)
    Z_test = s3_infer_embeddings(encoder, X_test_win, device=device)

    # 9) Calibration and mini-grid (with auto-inversion)
    base_clf = s3_make_base_clf(cfg.random_state)
    n_tr = len(Z_train);
    val_n = max(int(n_tr * cfg.val_frac), 1000);
    val_start = n_tr - val_n
    Z_tr_core, y_tr_core = Z_train[:val_start], y_train_w[:val_start]
    Z_val, y_val = Z_train[val_start:], y_train_w[val_start:]
    base_clf.fit(Z_tr_core, y_tr_core)
    calib = CalibratedClassifierCV(estimator=base_clf, cv='prefit', method='isotonic')
    calib.fit(Z_val, y_val)
    proba_tr_all = calib.predict_proba(Z_train)[:, 1]
    proba_test_raw = calib.predict_proba(Z_test)[:, 1]

    pbuy_grid = [0.52, 0.55, 0.58, 0.60, 0.65, 0.70]
    psell_grid = [0.48, 0.45, 0.42, 0.40, 0.35, 0.30]
    band_grid = [0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
    alpha_grid = [0.10, 0.15, 0.20, 0.25, 0.30]

    pb, ps, band, alpha, invert, score = s3_pick_thresholds_grid(
        proba_tr_all, y_train_w, cfg.val_frac,
        pbuy_grid, psell_grid, band_grid, alpha_grid,
        cfg.consec_required,
        min_signals=150
    )
    if pb is None:
        pb = cfg.prob_buy;
        ps = cfg.prob_sell;
        band = cfg.no_trade_band;
        alpha = cfg.ema_alpha;
        invert = False;
        score = float("nan")
        s3_log("Grid search found no valid combinations — applying default parameters.")
    else:
        s3_log(("Inverting signals" if invert else "Keeping direct signals") + f" (score={score:.3f})")

    proba_tr_best = (1.0 - proba_tr_all) if invert else proba_tr_all
    proba_test_best = (1.0 - proba_test_raw) if invert else proba_test_raw

    # Classification report on test set
    y_pred_test_bin = (proba_test_best >= 0.5).astype(int)
    s3_log("Classification (test, after calibration/inversion):")
    try:
        rep = classification_report(y_test_w, y_pred_test_bin, digits=3)
        print(rep)
        print(
            f"[INFO] F1={f1_score(y_test_w, y_pred_test_bin):.3f} | Acc={accuracy_score(y_test_w, y_pred_test_bin):.3f}")
    except Exception:
        pass
    s3_log(
        f"Parameters from grid validation/fallback: prob_buy={pb:.2f} prob_sell={ps:.2f} band={band:.2f} ema_alpha={alpha:.2f}")

    # 10) Backtest
    start_time = pd.Timestamp(t_test_w[0])
    df_times = df_raw['time'].to_numpy()
    start_pos = int(np.searchsorted(df_times, start_time))
    end_pos = start_pos + len(proba_test_best)
    test_df_slice = df_raw.iloc[start_pos:end_pos].reset_index(drop=True)
    assert len(test_df_slice) == len(proba_test_best)

    tr_pips = s3_true_range_pips(test_df_slice)
    med_tr = s3_rolling_median_tr_pips(tr_pips, win=48)

    trades, total_pips, equity = s3_backtest(
        df=test_df_slice,
        preds_proba=proba_test_best,
        prob_buy=pb,
        prob_sell=ps,
        no_trade_band=band,
        ema_alpha=alpha,
        consec_required=cfg.consec_required,
        opp_exit_thr=cfg.opp_exit_thr,
        tr_median_pips=med_tr,
        tp_mult=cfg.tp_mult, sl_mult=cfg.sl_mult,
        tp_min=cfg.tp_min, tp_max=cfg.tp_max, sl_min=cfg.sl_min, sl_max=cfg.sl_max,
        spread=cfg.spread,
        commission=cfg.commission,
        cooldown_bars=cfg.cooldown_bars,
        session_start_utc=cfg.session_start_utc,
        session_end_utc=cfg.session_end_utc,
        trail_prob_drop=cfg.trail_prob_drop,
        max_trades_per_day=cfg.max_trades_per_day,
        max_loss_pips_per_day=cfg.max_loss_pips_per_day,
        start_idx=0
    )

    # 11) Convert to unified format (USD)
    rows = [];
    eq_points = []
    if trades:
        eq = 0.0
        usd_per_pip = S3_PIP_VALUE_PER_LOT * LOT  # $1/pip at LOT=0.1
        for t in sorted(trades, key=lambda z: z.exit_time):
            pnl_usd = t.pnl_pips * usd_per_pip
            rows.append(dict(
                strategy="STRAT3",
                magic=MAGIC_NUMBERS["STRAT3"],
                comment="S3",
                symbol=cfg.symbol,
                entry_time=t.entry_time,
                exit_time=t.exit_time,
                direction=t.direction,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                pnl_pips=t.pnl_pips,
                pnl_usd=pnl_usd,
                reason=""
            ))
            eq += pnl_usd
            eq_points.append((t.exit_time, eq))
    df_s3 = pd.DataFrame(rows)
    df_s3 = normalize_times_df(df_s3)  # remove tz
    eq_points = normalize_eq_points(eq_points)
    m = calc_metrics_usd(df_s3) if not df_s3.empty else {}
    print_metrics("STRAT3", m)
    return df_s3, eq_points


# ============================================================================
# =============================  UNIFIED MAIN  ===============================
# ============================================================================

def main():
    """Main function: run all strategies and combine results"""
    print("[INFO] Initializing MT5…")
    if not MT5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {MT5.last_error()}")
    acc_info = MT5.account_info()
    if acc_info is None:
        print("[WARN] No authorized account. History available if terminal is running.")

    # --- RUN S1 ---
    df_s1, eq_s1 = run_strategy1_return_unified()

    # --- RUN S2 ---
    df_s2, eq_s2 = run_strategy2_return_unified()

    # --- RUN S3 ---
    df_s3, eq_s3 = run_strategy3_return_unified(s3_CFG)

    # --- PORTFOLIO AGGREGATION ---
    ensure_dir(UNIFIED_OUT_DIR)
    dfs_to_cat = [df for df in [df_s1, df_s2, df_s3] if df is not None and not df.empty]
    if not dfs_to_cat:
        print("\n[PORTFOLIO (ALL STRATEGIES)] No trades.")
        MT5.shutdown()
        return

    all_df = pd.concat(dfs_to_cat, ignore_index=True)
    # Critical: normalize time once more on aggregated data
    all_df = normalize_times_df(all_df)

    # Portfolio metrics
    m_port = calc_metrics_usd(all_df)
    print_metrics("PORTFOLIO (ALL STRATEGIES)", m_port)

    # Peak total margin (concurrent positions)
    peak_margin = compute_peak_margin(all_df.to_dict("records"))
    print(f"\n[PORTFOLIO] Peak total margin (USD): {peak_margin:.2f}")

    # Save CSV
    all_df_sorted = all_df.sort_values("exit_time")
    all_df_sorted.to_csv(UNIFIED_TRADES_CSV, index=False, encoding="utf-8-sig")
    print(f"[SAVE] All trades: {UNIFIED_TRADES_CSV}")

    # Unified equity plot
    curves = {
        "PORTFOLIO": list(zip(all_df_sorted["exit_time"], all_df_sorted["pnl_usd"].cumsum())),
        "STRAT1": eq_s1,
        "STRAT2": eq_s2,
        "STRAT3": eq_s3
    }
    plot_unified_equity(curves, "Equity (Portfolio + Strategies)", UNIFIED_EQUITY_PNG)
    print(f"[SAVE] Equity plot (portfolio + strategies): {UNIFIED_EQUITY_PNG}")

    MT5.shutdown()


if __name__ == "__main__":
    main()