"""
EURUSD → XAUUSD Lead-Lag Strategy (M15)
Version: 6.0 - Professional Edition
Author: Vladimir Korneev
Contact: t.me/realistic_algotrading
Repository: github.com/vger-cell/realistic-algotrading

Description: Lead-lag correlation strategy using EURUSD to generate signals
for XAUUSD trading. Features bounce signals from support/resistance levels
with RSI confirmation. Includes comprehensive backtesting with no look-ahead bias.
"""

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import warnings
import matplotlib.pyplot as plt
import os
import json
warnings.filterwarnings('ignore')

# ------------------------------- CONFIGURATION -------------------------------
class Config:
    # Data settings
    SYMBOLS = ["EURUSD", "XAUUSD"]
    TIMEFRAME = 'M15'
    TIMEFRAME_DICT = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1
    }
    DAYS_BACK = 365  # 1 year for more robust testing
    TRAIN_TEST_SPLIT = 0.7  # 70% train, 30% test

    # Strategy parameters
    LOOKBACK_PERIOD = 20
    MIN_RSI_LONG = 30
    MAX_RSI_SHORT = 75
    DELAY_BARS = 1
    HOLD_BARS = 10

    # Trend filter settings
    USE_TREND_FILTER = False
    ADX_PERIOD = 14
    ADX_THRESHOLD = 25
    MA_PERIOD = 50
    MA_TYPE = 'sma'

    # Time filter settings
    USE_TIME_FILTER = False
    TRADING_HOURS_START = 8
    TRADING_HOURS_END = 17

    # Stop settings
    USE_ATR_BASED_STOPS = False
    FIXED_TAKE_PROFIT_PIPS = 180
    FIXED_STOP_LOSS_PIPS = 120
    ATR_PERIOD = 14
    ATR_MULTIPLIER_TP = 2.0
    ATR_MULTIPLIER_SL = 1.5

    # Pip values
    PIP_VALUE = {
        'EURUSD': 10,
        'XAUUSD': 0.10
    }

    PIP_DECIMAL_MULTIPLIER = {
        'EURUSD': 0.0001,
        'XAUUSD': 0.01
    }

    # Position sizing
    RISK_PER_TRADE = 0.01
    MAX_LOT_SIZE = 10.0
    MIN_LOT_SIZE = 0.01

    # Backtest settings
    INITIAL_BALANCE = 10000
    COMMISSION = 2.0
    SLIPPAGE_PIPS = 0.5

CFG = Config()

# ------------------------------- 1. DATA LOADING -------------------------------
def load_data_with_cache():
    """Load data with caching to avoid repeated MT5 calls"""
    cache_file = f"data_cache_{CFG.SYMBOLS[0]}_{CFG.SYMBOLS[1]}_{CFG.TIMEFRAME}.pkl"

    try:
        import pickle
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded data from cache: {cache_file}")
        return data
    except:
        print("Cache not found, loading from MT5...")
        data = load_data_from_mt5()
        if data is not None:
            try:
                import pickle
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
                print(f"Data saved to cache: {cache_file}")
            except:
                print("Warning: Could not save cache")
        return data

def load_data_from_mt5():
    """Load synchronized data from MT5"""
    print(f"Loading {CFG.TIMEFRAME} data from MT5...")

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=CFG.DAYS_BACK)

    all_data = {}
    for sym in CFG.SYMBOLS:
        try:
            if not mt5.initialize():
                print("MT5 initialization failed.")
                return None

            timeframe = CFG.TIMEFRAME_DICT.get(CFG.TIMEFRAME)
            if timeframe is None:
                print(f"Invalid timeframe: {CFG.TIMEFRAME}")
                return None

            utc_from = int(start_date.timestamp())
            utc_to = int(end_date.timestamp())
            rates = mt5.copy_rates_range(sym, timeframe, utc_from, utc_to)

            if rates is None or len(rates) == 0:
                print(f"No data for {sym}")
                return None

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            df.set_index('time', inplace=True)
            df.columns = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']

            all_data[sym] = df
            print(f"  {sym}: {len(df)} bars loaded")

        except Exception as e:
            print(f"Error loading {sym}: {e}")
            return None
        finally:
            try:
                mt5.shutdown()
            except:
                pass

    # Synchronize data
    common_index = None
    for df in all_data.values():
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)

    for sym in all_data.keys():
        all_data[sym] = all_data[sym].reindex(common_index).copy()

    print(f"Synchronized {len(common_index)} bars")
    return all_data

# ------------------------------- 2. INDICATORS (NO LOOK-AHEAD BIAS) -------------------------------
def calculate_rsi_historical(prices, current_idx, period=14):
    """Calculate RSI at specific historical point without look-ahead bias"""
    if current_idx < period:
        return np.nan

    window_prices = prices.iloc[current_idx-period+1:current_idx+1]
    if len(window_prices) < period:
        return np.nan

    delta = window_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean().iloc[-1]
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean().iloc[-1]

    if loss == 0:
        return 100
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr_historical(df, current_idx, period=14):
    """Calculate ATR at specific historical point"""
    if current_idx < period:
        return np.nan

    start_idx = max(0, current_idx - period + 1)
    window_df = df.iloc[start_idx:current_idx+1]

    if len(window_df) < period:
        return np.nan

    high = window_df['high']
    low = window_df['low']
    close = window_df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean().iloc[-1]

def calculate_rolling_high_low_historical(df, current_idx, lookback_period):
    """Calculate rolling high/low at specific historical point"""
    if current_idx < lookback_period:
        return np.nan, np.nan

    start_idx = max(0, current_idx - lookback_period + 1)
    window_df = df.iloc[start_idx:current_idx+1]

    if len(window_df) < lookback_period:
        return np.nan, np.nan

    rolling_high = window_df['high'].max()
    rolling_low = window_df['low'].min()
    return rolling_high, rolling_low

def calculate_adx_historical(df, current_idx, period=14):
    """Calculate ADX at specific historical point"""
    if current_idx < period * 2:
        return np.nan, np.nan, np.nan

    start_idx = max(0, current_idx - period * 2 + 1)
    window_df = df.iloc[start_idx:current_idx+1]

    if len(window_df) < period * 2:
        return np.nan, np.nan, np.nan

    high = window_df['high']
    low = window_df['low']
    close = window_df['close']

    up = high.diff()
    down = -low.diff()

    plus_dm = up.where((up > down) & (up > 0), 0)
    minus_dm = down.where((down > up) & (down > 0), 0)

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-10)
    adx = dx.rolling(window=period).mean()

    return (adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else np.nan,
            plus_di.iloc[-1] if not pd.isna(plus_di.iloc[-1]) else np.nan,
            minus_di.iloc[-1] if not pd.isna(minus_di.iloc[-1]) else np.nan)

# ------------------------------- 3. SIGNAL GENERATION (NO LOOK-AHEAD) -------------------------------
def generate_signals_step_by_step(leader_data, test_start_idx, params=None):
    """
    Generate signals using only data available at each historical point
    WITHOUT look-ahead bias
    """
    if params is None:
        params = {
            'lookback_period': CFG.LOOKBACK_PERIOD,
            'min_rsi_long': CFG.MIN_RSI_LONG,
            'max_rsi_short': CFG.MAX_RSI_SHORT,
            'use_trend_filter': CFG.USE_TREND_FILTER
        }

    lookback_period = params.get('lookback_period', CFG.LOOKBACK_PERIOD)
    use_trend_filter = params.get('use_trend_filter', CFG.USE_TREND_FILTER)

    n_bars = len(leader_data)
    signals = pd.DataFrame(0, index=leader_data.index,
                          columns=['signal', 'price', 'type', 'valid', 'atr', 'trend_strength'])

    # Pre-calculate RSI for all points
    rsi_values = []
    for i in range(n_bars):
        rsi = calculate_rsi_historical(leader_data['close'], i, period=14)
        rsi_values.append(rsi)

    rsi_series = pd.Series(rsi_values, index=leader_data.index)

    # Start from minimum required bars
    start_idx = max(lookback_period * 2, test_start_idx, 30)

    for i in range(start_idx, n_bars):
        current_time = leader_data.index[i]
        current_price = leader_data.iloc[i]['close']

        rolling_high, rolling_low = calculate_rolling_high_low_historical(
            leader_data, i, lookback_period)

        current_rsi = rsi_series.iloc[i]
        atr_value = calculate_atr_historical(leader_data, i, CFG.ATR_PERIOD)

        if i > 0:
            prev_close = leader_data.iloc[i-1]['close']
            prev_rolling_high, prev_rolling_low = calculate_rolling_high_low_historical(
                leader_data, i-1, lookback_period)
        else:
            prev_close = np.nan
            prev_rolling_high = np.nan
            prev_rolling_low = np.nan

        if (pd.isna(rolling_high) or pd.isna(rolling_low) or
            pd.isna(current_rsi) or pd.isna(current_price)):
            continue

        # Time filter
        if CFG.USE_TIME_FILTER:
            hour = current_time.hour
            if not (CFG.TRADING_HOURS_START <= hour < CFG.TRADING_HOURS_END):
                continue

        # Trend filter
        trend_strength = 0
        trend_direction = 0
        if use_trend_filter:
            adx, plus_di, minus_di = calculate_adx_historical(leader_data, i, CFG.ADX_PERIOD)
            if not pd.isna(adx):
                trend_strength = adx
                trend_direction = 1 if plus_di > minus_di else -1 if plus_di < minus_di else 0
                is_strong_trend = adx > CFG.ADX_THRESHOLD
            else:
                is_strong_trend = False
        else:
            is_strong_trend = True

        signal_type = 0

        # Breakout LONG
        if (current_price > rolling_high) and (current_rsi <= 70):
            if not use_trend_filter or (trend_direction >= 0):
                if not use_trend_filter or is_strong_trend:
                    signals.loc[current_time, 'signal'] = 1
                    signals.loc[current_time, 'price'] = current_price
                    signal_type = 1

        # Breakout SHORT
        elif (current_price < rolling_low) and (current_rsi >= 30):
            if not use_trend_filter or (trend_direction <= 0):
                if not use_trend_filter or is_strong_trend:
                    signals.loc[current_time, 'signal'] = -1
                    signals.loc[current_time, 'price'] = current_price
                    signal_type = 2

        # Bounce SHORT
        elif not pd.isna(prev_close):
            if (prev_close >= prev_rolling_high) and (current_price < rolling_high):
                if not use_trend_filter or (trend_direction <= 0):
                    if not use_trend_filter or is_strong_trend:
                        if current_rsi >= 70:
                            signals.loc[current_time, 'signal'] = -1
                            signals.loc[current_time, 'price'] = current_price
                            signal_type = 3

            # Bounce LONG
            elif (prev_close <= prev_rolling_low) and (current_price > rolling_low):
                if not use_trend_filter or (trend_direction >= 0):
                    if not use_trend_filter or is_strong_trend:
                        if current_rsi <= 30:
                            signals.loc[current_time, 'signal'] = 1
                            signals.loc[current_time, 'price'] = current_price
                            signal_type = 4

        signals.loc[current_time, 'type'] = signal_type
        signals.loc[current_time, 'valid'] = 1 if signal_type > 0 else 0
        signals.loc[current_time, 'atr'] = atr_value if not pd.isna(atr_value) else 0
        signals.loc[current_time, 'trend_strength'] = trend_strength

    signals = signals[signals['valid'] == 1]

    return signals

# ------------------------------- 4. POSITION MANAGEMENT -------------------------------
class Position:
    def __init__(self, symbol, entry_time, entry_price, direction, lot_size,
                 take_profit_pips, stop_loss_pips, atr_value=0):
        self.symbol = symbol
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.direction = direction  # 1 for long, -1 for short
        self.lot_size = lot_size
        self.take_profit_pips = take_profit_pips
        self.stop_loss_pips = stop_loss_pips
        self.atr_value = atr_value
        self.exit_time = None
        self.exit_price = None
        self.pnl = 0
        self.pnl_pips = 0
        self.status = 'open'

        pip_multiplier = CFG.PIP_DECIMAL_MULTIPLIER[symbol]

        if direction == 1:
            self.take_profit_price = entry_price + (take_profit_pips * pip_multiplier)
            self.stop_loss_price = entry_price - (stop_loss_pips * pip_multiplier)
        else:
            self.take_profit_price = entry_price - (take_profit_pips * pip_multiplier)
            self.stop_loss_price = entry_price + (stop_loss_pips * pip_multiplier)

        self.tp_pips = take_profit_pips
        self.sl_pips = stop_loss_pips

    def check_exit(self, current_time, current_price, current_high, current_low):
        if self.status != 'open':
            return False

        if self.direction == 1:
            if current_low <= self.stop_loss_price:
                self.exit_time = current_time
                self.exit_price = self.stop_loss_price
                self.status = 'stopped'
                return True
            if current_high >= self.take_profit_price:
                self.exit_time = current_time
                self.exit_price = self.take_profit_price
                self.status = 'closed'
                return True
        else:
            if current_high >= self.stop_loss_price:
                self.exit_time = current_time
                self.exit_price = self.stop_loss_price
                self.status = 'stopped'
                return True
            if current_low <= self.take_profit_price:
                self.exit_time = current_time
                self.exit_price = self.take_profit_price
                self.status = 'closed'
                return True

        return False

    def close_position(self, current_time, current_price):
        if self.status == 'open':
            self.exit_time = current_time
            self.exit_price = current_price
            self.status = 'closed'
            return True
        return False

    def calculate_pnl(self):
        if self.exit_price is None or self.entry_price is None:
            return 0

        price_diff = self.exit_price - self.entry_price
        if self.direction == -1:
            price_diff = -price_diff

        self.pnl_pips = price_diff / CFG.PIP_DECIMAL_MULTIPLIER[self.symbol]
        pip_value_per_lot = CFG.PIP_VALUE[self.symbol]
        self.pnl = self.pnl_pips * self.lot_size * pip_value_per_lot
        self.pnl -= CFG.COMMISSION

        return self.pnl

# ------------------------------- 5. BACKTEST ENGINE -------------------------------
class BacktestEngine:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.positions = []
        self.closed_positions = []
        self.trade_history = []
        self.daily_pnl = {}

    def open_position(self, symbol, entry_time, entry_price, direction, atr_value=0,
                     risk_percent=0.01, take_profit_pips=None, stop_loss_pips=None):

        if CFG.USE_ATR_BASED_STOPS and atr_value > 0:
            atr_in_pips = atr_value / CFG.PIP_DECIMAL_MULTIPLIER[symbol]
            tp_pips = atr_in_pips * CFG.ATR_MULTIPLIER_TP
            sl_pips = atr_in_pips * CFG.ATR_MULTIPLIER_SL

            tp_pips = max(tp_pips, 100)
            sl_pips = max(sl_pips, 75)

            if tp_pips / sl_pips > 2.0:
                tp_pips = sl_pips * 1.5
            elif tp_pips / sl_pips < 1.2:
                tp_pips = sl_pips * 1.3
        else:
            tp_pips = CFG.FIXED_TAKE_PROFIT_PIPS
            sl_pips = CFG.FIXED_STOP_LOSS_PIPS

        if take_profit_pips is not None:
            tp_pips = take_profit_pips
        if stop_loss_pips is not None:
            sl_pips = stop_loss_pips

        risk_amount = self.balance * risk_percent
        pip_value = CFG.PIP_VALUE[symbol]
        risk_per_pip = risk_amount / sl_pips
        lot_size = risk_per_pip / pip_value

        lot_size = max(CFG.MIN_LOT_SIZE, min(lot_size, CFG.MAX_LOT_SIZE))
        lot_size = round(lot_size, 2)

        actual_risk = sl_pips * lot_size * pip_value
        if actual_risk > risk_amount * 1.1:
            lot_size = (risk_amount / (sl_pips * pip_value)) * 0.9
            lot_size = round(lot_size, 2)

        position = Position(
            symbol=symbol,
            entry_time=entry_time,
            entry_price=entry_price,
            direction=direction,
            lot_size=lot_size,
            take_profit_pips=tp_pips,
            stop_loss_pips=sl_pips,
            atr_value=atr_value
        )

        self.positions.append(position)
        return position

    def process_bar(self, current_time, market_data):
        for position in self.positions[:]:
            if position.symbol in market_data:
                symbol_data = market_data[position.symbol]
                if current_time in symbol_data.index:
                    current_price = symbol_data.loc[current_time, 'close']
                    current_high = symbol_data.loc[current_time, 'high']
                    current_low = symbol_data.loc[current_time, 'low']

                    if position.check_exit(current_time, current_price, current_high, current_low):
                        pnl = position.calculate_pnl()
                        self.balance += pnl
                        self.equity = self.balance

                        trade_record = {
                            'entry_time': position.entry_time,
                            'exit_time': position.exit_time,
                            'symbol': position.symbol,
                            'direction': position.direction,
                            'entry_price': position.entry_price,
                            'exit_price': position.exit_price,
                            'lot_size': position.lot_size,
                            'tp_pips': position.tp_pips,
                            'sl_pips': position.sl_pips,
                            'pnl': pnl,
                            'pnl_pips': position.pnl_pips,
                            'status': position.status,
                            'atr_value': position.atr_value
                        }
                        self.trade_history.append(trade_record)

                        self.closed_positions.append(position)
                        self.positions.remove(position)

        self.update_equity(current_time, market_data)

    def update_equity(self, current_time, market_data):
        floating_pnl = 0
        for position in self.positions:
            if position.symbol in market_data and current_time in market_data[position.symbol].index:
                current_price = market_data[position.symbol].loc[current_time, 'close']
                price_diff = current_price - position.entry_price
                if position.direction == -1:
                    price_diff = -price_diff

                pips = price_diff / CFG.PIP_DECIMAL_MULTIPLIER[position.symbol]
                pip_value = CFG.PIP_VALUE[position.symbol]
                floating_pnl += pips * position.lot_size * pip_value

        self.equity = self.balance + floating_pnl

    def get_performance_metrics(self):
        if not self.trade_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'total_loss': 0,
                'profit_factor': 0,
                'net_profit': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'avg_win_pips': 0,
                'avg_loss_pips': 0,
                'avg_tp_pips': 0,
                'avg_sl_pips': 0,
                'tp_sl_ratio': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'expectancy': 0,
                'trades_per_month': 0,
                'final_balance': self.balance,
                'total_return': 0,
                'avg_signal_delay_minutes': 0
            }

        trades_df = pd.DataFrame(self.trade_history)

        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if winning_trades > 0 else 0
        total_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else 0

        profit_factor = total_profit / total_loss if total_loss > 0 else 0

        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        avg_win_pips = trades_df[trades_df['pnl'] > 0]['pnl_pips'].mean() if winning_trades > 0 else 0
        avg_loss_pips = abs(trades_df[trades_df['pnl'] < 0]['pnl_pips'].mean()) if losing_trades > 0 else 0

        avg_tp_pips = trades_df['tp_pips'].mean() if 'tp_pips' in trades_df.columns and len(trades_df) > 0 else 0
        avg_sl_pips = trades_df['sl_pips'].mean() if 'sl_pips' in trades_df.columns and len(trades_df) > 0 else 0
        tp_sl_ratio = avg_tp_pips / avg_sl_pips if avg_sl_pips > 0 else 0

        # Max drawdown calculation
        equity_curve = []
        balance = self.initial_balance
        for trade in self.trade_history:
            balance += trade['pnl']
            equity_curve.append(balance)

        if equity_curve:
            running_max = np.maximum.accumulate(equity_curve)
            drawdowns = (equity_curve - running_max) / running_max * 100
            max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0
        else:
            max_drawdown = 0

        # Sharpe ratio
        if total_trades > 1:
            returns = trades_df['pnl'] / self.initial_balance
            if returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Trades per month
        if total_trades > 0 and self.trade_history:
            first_trade = self.trade_history[0]['entry_time']
            last_trade = self.trade_history[-1]['exit_time']
            if isinstance(first_trade, pd.Timestamp) and isinstance(last_trade, pd.Timestamp):
                days_diff = (last_trade - first_trade).days
                months = max(days_diff / 30, 1)
                trades_per_month = total_trades / months
            else:
                trades_per_month = total_trades / 3
        else:
            trades_per_month = 0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'profit_factor': profit_factor,
            'net_profit': total_profit - total_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_win_pips': avg_win_pips,
            'avg_loss_pips': avg_loss_pips,
            'avg_tp_pips': avg_tp_pips,
            'avg_sl_pips': avg_sl_pips,
            'tp_sl_ratio': tp_sl_ratio,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'expectancy': expectancy,
            'trades_per_month': trades_per_month,
            'final_balance': self.balance,
            'total_return': (self.balance - self.initial_balance) / self.initial_balance * 100,
            'avg_signal_delay_minutes': 0
        }

# ------------------------------- 6. COMPARISON TESTS -------------------------------
def run_comparison_tests(data):
    print("\n" + "="*60)
    print("RUNNING COMPARISON TESTS")
    print("="*60)

    test_configs = [
        {
            'name': 'Fixed TP/SL',
            'use_atr_stops': False,
            'use_trend_filter': False,
            'use_time_filter': False,
            'lookback_period': 20,
            'min_rsi_long': 30,
            'max_rsi_short': 75
        },
        {
            'name': 'ATR-based TP/SL',
            'use_atr_stops': True,
            'use_trend_filter': False,
            'use_time_filter': False,
            'lookback_period': 20,
            'min_rsi_long': 30,
            'max_rsi_short': 75
        },
        {
            'name': 'With Trend Filter',
            'use_atr_stops': True,
            'use_trend_filter': True,
            'use_time_filter': False,
            'lookback_period': 20,
            'min_rsi_long': 30,
            'max_rsi_short': 75
        },
        {
            'name': 'Full System (All Filters)',
            'use_atr_stops': True,
            'use_trend_filter': True,
            'use_time_filter': True,
            'lookback_period': 20,
            'min_rsi_long': 30,
            'max_rsi_short': 75
        },
        {
            'name': 'Aggressive (More Signals)',
            'use_atr_stops': False,
            'use_trend_filter': False,
            'use_time_filter': False,
            'lookback_period': 15,
            'min_rsi_long': 25,
            'max_rsi_short': 80
        }
    ]

    results = []

    for config in test_configs:
        print(f"\nTesting configuration: {config['name']}")

        # Temporarily update config
        original_atr = CFG.USE_ATR_BASED_STOPS
        original_trend = CFG.USE_TREND_FILTER
        original_time = CFG.USE_TIME_FILTER
        original_lookback = CFG.LOOKBACK_PERIOD
        original_min_rsi = CFG.MIN_RSI_LONG
        original_max_rsi = CFG.MAX_RSI_SHORT

        CFG.USE_ATR_BASED_STOPS = config['use_atr_stops']
        CFG.USE_TREND_FILTER = config.get('use_trend_filter', False)
        CFG.USE_TIME_FILTER = config.get('use_time_filter', False)
        CFG.LOOKBACK_PERIOD = config.get('lookback_period', CFG.LOOKBACK_PERIOD)
        CFG.MIN_RSI_LONG = config.get('min_rsi_long', CFG.MIN_RSI_LONG)
        CFG.MAX_RSI_SHORT = config.get('max_rsi_short', CFG.MAX_RSI_SHORT)

        params = {
            'lookback_period': CFG.LOOKBACK_PERIOD,
            'min_rsi_long': CFG.MIN_RSI_LONG,
            'max_rsi_short': CFG.MAX_RSI_SHORT,
            'use_trend_filter': CFG.USE_TREND_FILTER
        }

        # Run backtest
        engine, metrics, signals, valid_signals = run_backtest(data, params)

        if metrics and metrics['total_trades'] > 0:
            avg_delay = calculate_average_signal_delay(engine, signals)
            metrics['avg_signal_delay_minutes'] = avg_delay

            result = {
                'config_name': config['name'],
                'net_profit': metrics['net_profit'],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'total_trades': metrics['total_trades'],
                'max_drawdown': metrics['max_drawdown'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'total_return': metrics['total_return'],
                'trades_per_month': metrics.get('trades_per_month', 0),
                'avg_signal_delay': metrics['avg_signal_delay_minutes']
            }
            results.append(result)

            print(f"Results: Return={metrics['total_return']:.2f}%, "
                  f"Win Rate={metrics['win_rate']*100:.1f}%, "
                  f"Trades={metrics['total_trades']}, "
                  f"PF={metrics['profit_factor']:.2f}")
        else:
            print(f"Results: No trades generated")

        # Restore original config
        CFG.USE_ATR_BASED_STOPS = original_atr
        CFG.USE_TREND_FILTER = original_trend
        CFG.USE_TIME_FILTER = original_time
        CFG.LOOKBACK_PERIOD = original_lookback
        CFG.MIN_RSI_LONG = original_min_rsi
        CFG.MAX_RSI_SHORT = original_max_rsi

    # Display comparison
    if results:
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)

        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))

        # Save comparison results
        os.makedirs('comparison_results', exist_ok=True)
        results_df.to_csv('comparison_results/config_comparison.csv', index=False)

        # Find best configuration
        if len(results_df) > 0:
            best_config = results_df.loc[results_df['total_return'].idxmax()]
            print(f"\nBest configuration: {best_config['config_name']}")
            print(f"Total Return: {best_config['total_return']:.2f}%")
            print(f"Win Rate: {best_config['win_rate']*100:.1f}%")
            print(f"Profit Factor: {best_config['profit_factor']:.2f}")
            print(f"Max Drawdown: {best_config['max_drawdown']:.2f}%")

            return results_df, best_config

    print("No valid results from comparison tests")
    return None, None

def calculate_average_signal_delay(engine, signals):
    if not engine.trade_history or signals is None or len(signals) == 0:
        return 0

    delays = []
    for trade in engine.trade_history:
        signal_time = trade['entry_time'] - timedelta(minutes=15 * CFG.DELAY_BARS)

        if signal_time in signals.index:
            delay = CFG.DELAY_BARS * 15
            delays.append(delay)
        else:
            time_diffs = []
            for signal_idx in signals.index:
                time_diff = abs((signal_idx - signal_time).total_seconds())
                time_diffs.append((time_diff, signal_idx))

            if time_diffs:
                min_time_diff, closest_signal_time = min(time_diffs, key=lambda x: x[0])
                if min_time_diff <= 1800:
                    delay = (trade['entry_time'] - closest_signal_time).total_seconds() / 60
                    delays.append(delay)

    return np.mean(delays) if delays else CFG.DELAY_BARS * 15

# ------------------------------- 7. PERFORMANCE CHARTS -------------------------------
def create_performance_charts(engine, data, signals=None):
    """Create performance and price charts"""

    if not engine.trade_history:
        print("No trades to create charts")
        return

    trades_df = pd.DataFrame(engine.trade_history)
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))

    # 1. Equity Curve
    ax1 = plt.subplot(3, 2, 1)
    equity_curve = []
    balance_times = []
    balance = CFG.INITIAL_BALANCE

    # Sort trades by exit time
    sorted_trades = trades_df.sort_values('exit_time')

    for idx, trade in sorted_trades.iterrows():
        balance += trade['pnl']
        equity_curve.append(balance)
        balance_times.append(trade['exit_time'])

    ax1.plot(balance_times, equity_curve, 'b-', linewidth=2, label='Equity')
    ax1.axhline(y=CFG.INITIAL_BALANCE, color='r', linestyle='--', alpha=0.7, label='Initial Balance')
    ax1.set_title('Equity Curve', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Equity ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Drawdown
    ax2 = plt.subplot(3, 2, 2)
    if equity_curve:
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max * 100
        ax2.fill_between(balance_times, drawdowns, 0, color='red', alpha=0.3)
        ax2.plot(balance_times, drawdowns, 'r-', linewidth=1)
        ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([min(drawdowns) * 1.1, 0])

    # 3. XAUUSD Price with Trades
    ax3 = plt.subplot(3, 2, 3)
    if 'XAUUSD' in data:
        xau_data = data['XAUUSD']
        # Get data for the period of trades
        first_trade = trades_df['entry_time'].min()
        last_trade = trades_df['exit_time'].max()
        mask = (xau_data.index >= first_trade - timedelta(days=5)) & (xau_data.index <= last_trade + timedelta(days=5))
        plot_data = xau_data[mask]

        ax3.plot(plot_data.index, plot_data['close'], 'k-', linewidth=1, alpha=0.7, label='XAUUSD Price')
        ax3.set_title('XAUUSD Price with Trades', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Price')
        ax3.grid(True, alpha=0.3)

        # Plot trades
        for idx, trade in trades_df.iterrows():
            color = 'green' if trade['pnl'] > 0 else 'red'
            marker = '^' if trade['direction'] == 1 else 'v'
            ax3.scatter(trade['entry_time'], trade['entry_price'],
                       color=color, marker=marker, s=100, alpha=0.8, zorder=5)

    # 4. P&L Distribution
    ax4 = plt.subplot(3, 2, 4)
    if len(trades_df) > 0:
        winning_trades = trades_df[trades_df['pnl'] > 0]['pnl']
        losing_trades = trades_df[trades_df['pnl'] < 0]['pnl']

        bins = 20
        if len(winning_trades) > 0:
            ax4.hist(winning_trades, bins=bins, alpha=0.7, color='green', label='Winning Trades')
        if len(losing_trades) > 0:
            ax4.hist(losing_trades, bins=bins, alpha=0.7, color='red', label='Losing Trades')

        ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('P&L Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('P&L ($)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # 5. Monthly Returns
    ax5 = plt.subplot(3, 2, 5)
    if len(trades_df) > 0:
        trades_df['month'] = trades_df['exit_time'].dt.to_period('M')
        monthly_pnl = trades_df.groupby('month')['pnl'].sum()
        monthly_returns = (monthly_pnl / CFG.INITIAL_BALANCE) * 100

        colors = ['green' if x >= 0 else 'red' for x in monthly_returns]
        bars = ax5.bar(range(len(monthly_returns)), monthly_returns.values, color=colors, alpha=0.7)
        ax5.set_title('Monthly Returns', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Month')
        ax5.set_ylabel('Return (%)')
        ax5.set_xticks(range(len(monthly_returns)))
        ax5.set_xticklabels([str(m) for m in monthly_returns.index], rotation=45, ha='right')
        ax5.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8)

    # 6. Trade Duration vs P&L
    ax6 = plt.subplot(3, 2, 6)
    if len(trades_df) > 0:
        trades_df['duration_hours'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600
        colors = ['green' if pnl > 0 else 'red' for pnl in trades_df['pnl']]
        scatter = ax6.scatter(trades_df['duration_hours'], trades_df['pnl'],
                            c=colors, alpha=0.6, s=50)
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax6.set_title('Trade Duration vs P&L', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Duration (hours)')
        ax6.set_ylabel('P&L ($)')
        ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('backtest_results_optimized/performance_charts.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Create separate price vs time chart
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # EURUSD Price
    if 'EURUSD' in data:
        eur_data = data['EURUSD']
        ax1.plot(eur_data.index, eur_data['close'], 'b-', linewidth=1, alpha=0.7, label='EURUSD')
        ax1.set_ylabel('EURUSD Price', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')

    # XAUUSD Price
    if 'XAUUSD' in data:
        xau_data = data['XAUUSD']
        ax2.plot(xau_data.index, xau_data['close'], 'orange', linewidth=1, alpha=0.7, label='XAUUSD')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('XAUUSD Price', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')

    plt.suptitle('EURUSD and XAUUSD Prices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('backtest_results_optimized/price_charts.png', dpi=150, bbox_inches='tight')
    plt.show()

# ------------------------------- 8. MAIN BACKTEST -------------------------------
def run_backtest(data, params=None):
    """Run walk-forward backtest for EURUSD → XAUUSD strategy"""

    if data is None or len(data) < 2:
        print("Insufficient data")
        return None, None, None, None

    leader_data = data['EURUSD'].copy()
    follower_data = data['XAUUSD'].copy()

    split_idx = int(len(leader_data) * CFG.TRAIN_TEST_SPLIT)
    test_data = leader_data.iloc[split_idx:]

    engine = BacktestEngine(initial_balance=CFG.INITIAL_BALANCE)

    signals = generate_signals_step_by_step(leader_data, split_idx, params)
    test_signals = signals[signals.index >= test_data.index[0]]
    valid_signals = test_signals.copy()

    print(f"Signals in test period: {len(valid_signals)}")
    if not valid_signals.empty:
        signal_counts = valid_signals['type'].value_counts()
        type_names = {
            1: 'Breakout LONG',
            2: 'Breakout SHORT',
            3: 'Bounce SHORT',
            4: 'Bounce LONG'
        }

        signal_summary = {}
        for sig_type, count in signal_counts.items():
            signal_summary[type_names.get(sig_type, f'Type {sig_type}')] = count

        print(f"Signal types: {signal_summary}")

    if len(valid_signals) > 0:
        print("Running trades...")

        valid_signals = valid_signals.sort_index()

        for i, (signal_time, signal_row) in enumerate(valid_signals.iterrows()):
            if i % 10 == 0 and len(valid_signals) > 10:
                print(f"Processing signal {i+1}/{len(valid_signals)}...")

            signal_direction = signal_row['signal']
            atr_value = signal_row['atr']

            delay_end_time = signal_time + timedelta(minutes=15 * CFG.DELAY_BARS)

            available_times = [t for t in follower_data.index if t > delay_end_time]
            if not available_times:
                continue

            entry_time = available_times[0]

            if entry_time not in follower_data.index:
                continue

            entry_price = follower_data.loc[entry_time, 'open']

            slippage_amount = CFG.SLIPPAGE_PIPS * CFG.PIP_DECIMAL_MULTIPLIER['XAUUSD']
            if signal_direction == 1:
                entry_price += slippage_amount
            else:
                entry_price -= slippage_amount

            position = engine.open_position(
                symbol='XAUUSD',
                entry_time=entry_time,
                entry_price=entry_price,
                direction=signal_direction,
                atr_value=atr_value,
                risk_percent=CFG.RISK_PER_TRADE
            )

            follower_times = list(follower_data.index)
            if entry_time in follower_times:
                entry_idx = follower_times.index(entry_time)
                exit_idx = min(entry_idx + CFG.HOLD_BARS, len(follower_times) - 1)

                for j in range(entry_idx + 1, exit_idx + 1):
                    process_time = follower_times[j]

                    market_data = {
                        'XAUUSD': follower_data.loc[[process_time]]
                    }

                    engine.process_bar(process_time, market_data)

                    if position.status != 'open':
                        break

                if position.status == 'open':
                    exit_time = follower_times[exit_idx]
                    exit_price = follower_data.loc[exit_time, 'close']
                    position.close_position(exit_time, exit_price)

                    pnl = position.calculate_pnl()
                    engine.balance += pnl
                    engine.equity = engine.balance

                    trade_record = {
                        'entry_time': position.entry_time,
                        'exit_time': position.exit_time,
                        'symbol': position.symbol,
                        'direction': position.direction,
                        'entry_price': position.entry_price,
                        'exit_price': position.exit_price,
                        'lot_size': position.lot_size,
                        'tp_pips': position.tp_pips,
                        'sl_pips': position.sl_pips,
                        'pnl': pnl,
                        'pnl_pips': position.pnl_pips,
                        'status': 'timeout',
                        'atr_value': position.atr_value
                    }
                    engine.trade_history.append(trade_record)
                    engine.closed_positions.append(position)
                    if position in engine.positions:
                        engine.positions.remove(position)
    else:
        print("No valid signals to trade")

    metrics = engine.get_performance_metrics()

    avg_delay = calculate_average_signal_delay(engine, signals)
    metrics['avg_signal_delay_minutes'] = avg_delay
    metrics['total_signals'] = len(valid_signals)

    return engine, metrics, signals, valid_signals

# ------------------------------- 9. MAIN FUNCTION -------------------------------
def main():
    print("=" * 80)
    print("EURUSD → XAUUSD LEAD-LAG STRATEGY - VERSION 6.0")
    print("=" * 80)

    os.makedirs('backtest_results_optimized', exist_ok=True)

    data = load_data_with_cache()

    if data is None:
        print("Failed to load data")
        return

    print("\n" + "="*60)
    print("CURRENT CONFIGURATION")
    print("="*60)
    print(f"Lookback Period: {CFG.LOOKBACK_PERIOD}")
    print(f"RSI Long Threshold: {CFG.MIN_RSI_LONG}")
    print(f"RSI Short Threshold: {CFG.MAX_RSI_SHORT}")
    print(f"Delay Bars: {CFG.DELAY_BARS} ({CFG.DELAY_BARS * 15} min)")
    print(f"Hold Bars: {CFG.HOLD_BARS} ({CFG.HOLD_BARS * 15} min)")
    print(f"Trend Filter: {'ON' if CFG.USE_TREND_FILTER else 'OFF'}")
    print(f"Time Filter: {'ON' if CFG.USE_TIME_FILTER else 'OFF'}")
    print(f"Stop Type: {'ATR-based' if CFG.USE_ATR_BASED_STOPS else 'Fixed'}")
    print(f"TP/SL Ratio: {CFG.FIXED_TAKE_PROFIT_PIPS/CFG.FIXED_STOP_LOSS_PIPS:.2f}")

    print("\n" + "="*60)
    print("RUNNING COMPARISON TESTS")
    print("="*60)

    comparison_results, best_config = run_comparison_tests(data)

    print("\n" + "="*60)
    print("RUNNING FINAL BACKTEST WITH BEST CONFIGURATION")
    print("="*60)

    if best_config is not None:
        print(f"\nUsing best configuration: {best_config['config_name']}")

        if 'Fixed TP/SL' in best_config['config_name']:
            CFG.USE_ATR_BASED_STOPS = False
        elif 'ATR-based' in best_config['config_name']:
            CFG.USE_ATR_BASED_STOPS = True

        if 'Trend Filter' in best_config['config_name'] or 'Full System' in best_config['config_name']:
            CFG.USE_TREND_FILTER = True
        else:
            CFG.USE_TREND_FILTER = False

        if 'Full System' in best_config['config_name']:
            CFG.USE_TIME_FILTER = True
        else:
            CFG.USE_TIME_FILTER = False

        if 'Aggressive' in best_config['config_name']:
            CFG.LOOKBACK_PERIOD = 15
            CFG.MIN_RSI_LONG = 25
            CFG.MAX_RSI_SHORT = 80
    else:
        print("\nNo best configuration found, using defaults")
        best_config = {'config_name': 'Default'}

    final_params = {
        'lookback_period': CFG.LOOKBACK_PERIOD,
        'min_rsi_long': CFG.MIN_RSI_LONG,
        'max_rsi_short': CFG.MAX_RSI_SHORT,
        'use_trend_filter': CFG.USE_TREND_FILTER
    }

    engine, metrics, signals, valid_signals = run_backtest(data, final_params)

    if engine is None or metrics is None:
        print("Backtest failed")
        return

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS - {best_config['config_name']}")
    print(f"{'='*60}")

    key_metrics = [
        'total_return', 'net_profit', 'win_rate', 'profit_factor',
        'total_trades', 'max_drawdown', 'sharpe_ratio', 'expectancy',
        'trades_per_month', 'avg_signal_delay_minutes'
    ]

    for key in key_metrics:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                if 'rate' in key:
                    print(f"{key:25}: {value*100:.1f}%")
                elif 'profit' in key or 'expectancy' in key:
                    print(f"{key:25}: ${value:.2f}")
                elif 'return' in key or 'drawdown' in key:
                    print(f"{key:25}: {value:.2f}%")
                elif 'ratio' in key:
                    print(f"{key:25}: {value:.3f}")
                else:
                    print(f"{key:25}: {value:.2f}")
            else:
                print(f"{key:25}: {value}")

    if engine.trade_history:
        print(f"\n{'='*60}")
        print("TRADE DETAILS")
        print(f"{'='*60}")

        trades_df = pd.DataFrame(engine.trade_history)
        trades_df = trades_df.sort_values('exit_time')

        print(f"\nTrade Statistics:")
        print(f"Total trades: {len(trades_df)}")
        print(f"Win rate: {(trades_df['pnl'] > 0).mean()*100:.1f}%")
        print(f"Avg profit per trade: ${trades_df['pnl'].mean():.2f}")
        print(f"Std of profits: ${trades_df['pnl'].std():.2f}")

        if 'tp_pips' in trades_df.columns and len(trades_df) > 0:
            avg_tp = trades_df['tp_pips'].mean()
            avg_sl = trades_df['sl_pips'].mean()
            print(f"Avg TP: {avg_tp:.1f} pips")
            print(f"Avg SL: {avg_sl:.1f} pips")
            if avg_sl > 0:
                print(f"TP/SL Ratio: {avg_tp/avg_sl:.2f}")

        if len(trades_df) >= 5:
            print(f"\nLast 5 trades:")
            display_cols = ['entry_time', 'exit_time', 'direction', 'entry_price',
                           'exit_price', 'tp_pips', 'sl_pips', 'pnl', 'status']
            print(trades_df[display_cols].tail().to_string())

        trades_df.to_csv('backtest_results_optimized/trade_history.csv', index=False)
        print(f"\nFull trade history saved")
    else:
        print(f"\nNo trades executed")

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('backtest_results_optimized/performance_metrics.csv', index=False)
    print(f"Performance metrics saved")

    config_dict = {k: v for k, v in CFG.__dict__.items() if not k.startswith('_')}
    config_df = pd.DataFrame([config_dict])
    config_df.to_csv('backtest_results_optimized/configuration.csv', index=False)
    print(f"Configuration saved")

    if comparison_results is not None:
        comparison_results.to_csv('backtest_results_optimized/comparison_results.csv', index=False)
        print(f"Comparison results saved")

    if signals is not None and len(signals) > 0:
        signals.to_csv('backtest_results_optimized/signals.csv')
        print(f"Signals saved")

    print(f"\n{'='*60}")
    print("CREATING PERFORMANCE CHARTS")
    print(f"{'='*60}")

    create_performance_charts(engine, data, signals)
    print("Performance charts created and saved")

    print(f"\n{'='*60}")
    print("STRATEGY OPTIMIZATION SUMMARY")
    print(f"{'='*60}")

    if best_config is not None and 'total_return' in best_config:
        print(f"\nBest configuration: {best_config['config_name']}")
        print(f"Total Return: {best_config['total_return']:.2f}%")
        print(f"Win Rate: {best_config['win_rate']*100:.1f}%")
        print(f"Profit Factor: {best_config['profit_factor']:.2f}")
        print(f"Max Drawdown: {best_config['max_drawdown']:.2f}%")
    else:
        print("\nAnalysis of current results:")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Number of Trades: {metrics['total_trades']}")

if __name__ == "__main__":
    main()
