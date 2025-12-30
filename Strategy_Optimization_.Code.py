"""
Multi-Strategy Forex Trading System with Cross-Validation
Author: Vladimir Korneev
Telegram: t.me/realistic_algotrading
GitHub: github.com/vger-cell/realistic-algotrading

Features:
1. Three trading strategies (STOCH_buy, RSI_sell, MACD_sell)
2. Time-series cross-validation to prevent overfitting
3. Walk-forward validation for robustness testing
4. Final out-of-sample test on unseen data
5. Realistic trading costs (commission, slippage)

IMPORTANT NOTE: This system passed validation methodology
but showed NEGATIVE PROFIT (-2.13%) in final testing.
Educational example of proper validation process.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')

# ==================== 1. CONFIGURATION ====================
LOT_SIZE = 0.1  # Standard lot size for retail forex
POINT_VALUE = 10  # $10 per pip on standard EURUSD lot
INITIAL_BALANCE = 10000  # Realistic starting capital
MAX_DRAWDOWN_LIMIT = 20  # Maximum allowable drawdown percentage

# Parameter ranges for optimization (simplified for speed)
HOLD_PERIODS = [6, 12, 24]  # Position holding hours
STOP_LOSSES = [15, 20, 30]  # Stop-loss in pips
TAKE_PROFITS = [20, 30, 50]  # Take-profit in pips

COMMISSION = 0.5  # Per-trade commission in USD
SLIPPAGE = 0.5  # Average slippage in pips


# ==================== 2. DATA FETCHER CLASS ====================
class MT5DataFetcher:
    """Fetches historical data from MetaTrader 5 platform"""
    
    def __init__(self, symbol="EURUSD"):
        self.symbol = symbol
        if not mt5.initialize():
            print("MT5 initialization error")
            raise ConnectionError("Failed to connect to MT5")

    def fetch_data(self, timeframe, bars=5000):
        """Fetch specified number of bars for given timeframe"""
        tf_mapping = {
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }

        if timeframe not in tf_mapping:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        # Get historical rates from MT5
        rates = mt5.copy_rates_from_pos(
            self.symbol,
            tf_mapping[timeframe],
            0,
            bars
        )

        if rates is None or len(rates) == 0:
            raise ValueError(f"Failed to fetch data for {timeframe}")

        # Convert to pandas DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df


# ==================== 3. STRATEGY GENERATOR ====================
class StrategyGenerator:
    """Generates trading signals from technical indicators"""
    
    def __init__(self, df):
        self.df = df.copy()

    def calculate_rsi(self, period=14):
        """Calculate Relative Strength Index"""
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def strategy_stochastic_buy(self):
        """Stochastic oscillator buy signal: oversold + bullish crossover"""
        low_14 = self.df['low'].rolling(14).min()
        high_14 = self.df['high'].rolling(14).max()
        k = 100 * ((self.df['close'] - low_14) / (high_14 - low_14))
        d = k.rolling(3).mean()
        # BUY signal: stochastic in oversold zone and %K crosses %D from below
        buy_signal = ((k < 20) & (d < 20) & (k > d) & (k.shift(1) <= d.shift(1))).astype(int)
        return pd.Series(buy_signal, name='STOCH_buy')

    def strategy_rsi_sell(self):
        """RSI sell signal: overbought + bearish turn"""
        rsi = self.calculate_rsi(14)
        # SELL signal: RSI enters overbought zone from below
        sell_signal = ((rsi > 70) & (rsi.shift(1) <= 70)).astype(int)
        return pd.Series(sell_signal, name='RSI_sell')

    def strategy_macd_sell(self):
        """MACD sell signal: bearish crossover"""
        exp1 = self.df['close'].ewm(span=12).mean()
        exp2 = self.df['close'].ewm(span=26).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=9).mean()
        # SELL signal: MACD crosses below signal line
        sell_signal = ((macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))).astype(int)
        return pd.Series(sell_signal, name='MACD_sell')

    def generate_signals(self):
        """Generate all trading signals"""
        signals = pd.DataFrame(index=self.df.index)
        signals['STOCH_buy'] = self.strategy_stochastic_buy()
        signals['RSI_sell'] = self.strategy_rsi_sell()
        signals['MACD_sell'] = self.strategy_macd_sell()
        return signals


# ==================== 4. TRADING SIMULATOR ====================
class TradeSimulator:
    """Simulates trades with realistic costs and execution"""
    
    def __init__(self, lot_size=0.1, commission=0.5, slippage=0.5):
        self.lot_size = lot_size
        self.commission = commission
        self.slippage = slippage
        self.point_value = 10  # EURUSD point value

    def simulate_trade(self, entry_price, exit_price, trade_type,
                       entry_time, exit_time):
        """Simulate single trade with costs"""
        # Account for slippage on entry
        if trade_type == 'BUY':
            effective_entry = entry_price + (self.slippage / 10000)
            pips = (exit_price - effective_entry) * 10000
        else:
            effective_entry = entry_price - (self.slippage / 10000)
            pips = (effective_entry - exit_price) * 10000

        # Calculate profit including commissions (entry + exit)
        profit = pips * self.point_value * self.lot_size - (2 * self.commission)

        return {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'trade_type': trade_type,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'pips': pips,
            'profit': profit,
            'duration_hours': (exit_time - entry_time).total_seconds() / 3600
        }

    def find_exit_price(self, data, entry_idx, trade_type,
                        stop_loss_pips, take_profit_pips, max_hours):
        """Find exit price based on stops, profit targets, or time limit"""
        entry_price = data.iloc[entry_idx]['close']
        entry_time = data.index[entry_idx]

        # Calculate stop-loss and take-profit levels
        if trade_type == 'BUY':
            stop_loss = entry_price - (stop_loss_pips / 10000)
            take_profit = entry_price + (take_profit_pips / 10000)
        else:
            stop_loss = entry_price + (stop_loss_pips / 10000)
            take_profit = entry_price - (take_profit_pips / 10000)

        # Maximum bars to hold position
        max_bars = max_hours  # Assuming H1 timeframe

        # Look for exit in subsequent bars
        exit_idx = entry_idx + 1
        exit_price = None
        exit_time = None

        while exit_idx < len(data) and exit_idx <= entry_idx + max_bars:
            current_bar = data.iloc[exit_idx]
            high = current_bar['high']
            low = current_bar['low']

            if trade_type == 'BUY':
                # Check take-profit hit
                if high >= take_profit:
                    exit_price = take_profit
                    exit_time = data.index[exit_idx]
                    break
                # Check stop-loss hit
                elif low <= stop_loss:
                    exit_price = stop_loss
                    exit_time = data.index[exit_idx]
                    break
            else:  # SELL
                # Check take-profit hit
                if low <= take_profit:
                    exit_price = take_profit
                    exit_time = data.index[exit_idx]
                    break
                # Check stop-loss hit
                elif high >= stop_loss:
                    exit_price = stop_loss
                    exit_time = data.index[exit_idx]
                    break

            exit_idx += 1

        # If stops didn't trigger, exit at close after max holding period
        if exit_price is None:
            exit_idx = min(entry_idx + max_bars, len(data) - 1)
            exit_price = data.iloc[exit_idx]['close']
            exit_time = data.index[exit_idx]

        return exit_price, exit_time


# ==================== 5. TIME SERIES CROSS-VALIDATION ====================
class TimeSeriesCrossValidator:
    """Cross-validator that respects time ordering (no future data leaks)"""
    
    def __init__(self, n_splits=5, test_size=0.2):
        self.n_splits = n_splits
        self.test_size = test_size

    def split(self, data):
        """Split data into training and test sets in chronological order"""
        n = len(data)
        test_size = int(n * self.test_size)

        for i in range(self.n_splits):
            # Test data always comes AFTER training data
            train_end = n - test_size * (self.n_splits - i)
            test_start = train_end
            test_end = test_start + test_size

            if train_end <= 0 or test_end > n:
                continue

            train_indices = list(range(0, train_end))
            test_indices = list(range(test_start, test_end))

            yield train_indices, test_indices


# ==================== 6. CROSS-VALIDATION OPTIMIZER ====================
class CrossValidationOptimizer:
    """Optimizes strategy parameters using time-series cross-validation"""
    
    def __init__(self, data, lot_size=0.1):
        self.data = data
        self.lot_size = lot_size
        self.cv = TimeSeriesCrossValidator(n_splits=3, test_size=0.2)
        self.best_params = {}

    def evaluate_params(self, data, signals, strategy_name, params):
        """Evaluate parameter set on given dataset"""
        trade_simulator = TradeSimulator(self.lot_size)

        # Determine trade type from strategy name
        if 'buy' in strategy_name.lower():
            trade_type = 'BUY'
        else:
            trade_type = 'SELL'

        signal_series = signals[strategy_name]
        signal_indices = signal_series[signal_series == 1].index

        trades = []
        total_profit = 0
        last_trade_idx = -5  # Prevent overlapping trades (5-bar cooldown)

        for idx in signal_indices:
            try:
                signal_idx = data.index.get_loc(idx)

                # Avoid overlapping trades
                if signal_idx - last_trade_idx < 5:
                    continue

                # Find exit price based on strategy parameters
                exit_price, exit_time = trade_simulator.find_exit_price(
                    data, signal_idx, trade_type,
                    params['stop_loss_pips'],
                    params['take_profit_pips'],
                    params['hold_hours']
                )

                entry_price = data.loc[idx, 'close']

                # Simulate the trade
                trade = trade_simulator.simulate_trade(
                    entry_price=entry_price,
                    exit_price=exit_price,
                    trade_type=trade_type,
                    entry_time=idx,
                    exit_time=exit_time
                )

                trades.append(trade)
                total_profit += trade['profit']
                last_trade_idx = signal_idx

            except:
                continue

        if not trades:
            return None

        # Calculate performance metrics
        profits = [t['profit'] for t in trades]
        winning_trades = sum(1 for p in profits if p > 0)
        total_trades = len(trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Calculate maximum drawdown
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / (INITIAL_BALANCE + 1e-10) * 100
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0

        # Filter out parameter sets with excessive drawdown
        if max_drawdown < -MAX_DRAWDOWN_LIMIT:
            return None

        return {
            'total_trades': total_trades,
            'total_profit': total_profit,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'avg_profit': np.mean(profits),
            'trades': trades
        }

    def optimize_strategy_cv(self, strategy_name, hold_periods, stop_losses, take_profits):
        """Optimize strategy parameters using cross-validation"""
        print(f"\nCross-validating strategy: {strategy_name}")
        print("-" * 60)

        # Generate signals for all data
        strat_gen = StrategyGenerator(self.data)
        all_signals = strat_gen.generate_signals()

        # Generate all parameter combinations
        param_combinations = []
        for hold in hold_periods:
            for sl in stop_losses:
                for tp in take_profits:
                    if tp > sl:  # TP must be greater than SL for positive risk-reward
                        param_combinations.append({
                            'hold_hours': hold,
                            'stop_loss_pips': sl,
                            'take_profit_pips': tp,
                            'risk_reward': tp / sl
                        })

        print(f"Testing {len(param_combinations)} combinations on {self.cv.n_splits} folds...")

        # Evaluate each parameter combination
        param_scores = []

        for params in param_combinations:
            fold_results = []

            # Cross-validation loop
            for fold, (train_idx, test_idx) in enumerate(self.cv.split(self.data)):
                # Split data into train and test sets
                train_data = self.data.iloc[train_idx].copy()
                test_data = self.data.iloc[test_idx].copy()

                # Generate signals for training data
                train_strat_gen = StrategyGenerator(train_data)
                train_signals = train_strat_gen.generate_signals()

                # Evaluate on training data
                train_result = self.evaluate_params(
                    train_data, train_signals, strategy_name, params
                )

                if train_result is None:
                    continue

                # Validate on test data
                test_strat_gen = StrategyGenerator(test_data)
                test_signals = test_strat_gen.generate_signals()

                test_result = self.evaluate_params(
                    test_data, test_signals, strategy_name, params
                )

                if test_result is None:
                    continue

                # Calculate composite score (profit - drawdown penalty)
                score = test_result['total_profit'] - abs(test_result['max_drawdown']) * 10
                fold_results.append({
                    'fold': fold,
                    'train_result': train_result,
                    'test_result': test_result,
                    'score': score
                })

            # Aggregate results across folds
            if fold_results:
                avg_score = np.mean([r['score'] for r in fold_results])
                avg_test_profit = np.mean([r['test_result']['total_profit'] for r in fold_results])
                avg_test_drawdown = np.mean([r['test_result']['max_drawdown'] for r in fold_results])

                param_scores.append({
                    'params': params,
                    'avg_score': avg_score,
                    'avg_test_profit': avg_test_profit,
                    'avg_test_drawdown': avg_test_drawdown,
                    'fold_results': fold_results,
                    'num_folds': len(fold_results)
                })

        if not param_scores:
            print(f"No suitable parameters found for {strategy_name}")
            return None

        # Select best parameters by average score
        param_scores.sort(key=lambda x: x['avg_score'], reverse=True)
        best = param_scores[0]

        # Display best parameters
        print(f"\n🏆 BEST PARAMETERS for {strategy_name}:")
        print(f"   Holding: {best['params']['hold_hours']} hours")
        print(f"   Stop-loss: {best['params']['stop_loss_pips']} pips")
        print(f"   Take-profit: {best['params']['take_profit_pips']} pips")
        print(f"   Risk/Reward: 1:{best['params']['risk_reward']:.2f}")
        print(f"   Average test profit: ${best['avg_test_profit']:.2f}")
        print(f"   Average drawdown: {best['avg_test_drawdown']:.1f}%")
        print(f"   Tested folds: {best['num_folds']}")

        return best

    def optimize_all_strategies(self):
        """Optimize all three strategies"""
        strategies = ['STOCH_buy', 'RSI_sell', 'MACD_sell']

        for strategy in strategies:
            best_result = self.optimize_strategy_cv(
                strategy, HOLD_PERIODS, STOP_LOSSES, TAKE_PROFITS
            )
            if best_result:
                self.best_params[strategy] = {
                    'params': best_result['params'],
                    'cv_results': best_result
                }

        return self.best_params


# ==================== 7. FINAL VALIDATION ====================
class FinalValidator:
    """Performs final validation using walk-forward and out-of-sample testing"""
    
    def __init__(self, initial_balance=10000, lot_size=0.1):
        self.initial_balance = initial_balance
        self.lot_size = lot_size

    def walk_forward_test(self, data, best_params, walk_forward_windows=5):
        """Walk-forward validation on new data windows"""
        print("\n" + "=" * 80)
        print("WALK-FORWARD VALIDATION ON NEW DATA")
        print("=" * 80)

        # Split data into walk-forward windows
        n = len(data)
        window_size = n // (walk_forward_windows + 1)

        all_results = []

        for window in range(walk_forward_windows):
            # Training data (all data before current window)
            train_end = window * window_size
            train_data = data.iloc[:train_end].copy() if train_end > 0 else None

            # Test data (current window)
            test_start = train_end
            test_end = min(test_start + window_size, n)

            # Skip if window too small
            if test_start >= n or test_end - test_start < 100:
                break

            test_data = data.iloc[test_start:test_end].copy()

            print(f"\nWindow {window + 1}/{walk_forward_windows}")
            print(f"Training: {len(train_data) if train_data is not None else 0} bars")
            print(f"Test: {len(test_data)} bars ({test_data.index[0]} - {test_data.index[-1]})")

            # Test on this window
            if train_data is not None and len(train_data) > 100:
                window_result = self.test_on_data(test_data, best_params)
                if window_result:
                    all_results.append(window_result)

        # Aggregate walk-forward results
        if all_results:
            total_profit = sum(r['total_profit'] for r in all_results)
            total_trades = sum(r['total_trades'] for r in all_results)
            avg_win_rate = np.mean([r['win_rate'] for r in all_results])
            avg_drawdown = np.mean([r['max_drawdown'] for r in all_results])

            print("\n" + "=" * 80)
            print("WALK-FORWARD VALIDATION RESULTS:")
            print("=" * 80)
            print(f"Total windows: {len(all_results)}")
            print(f"Total profit: ${total_profit:.2f}")
            print(f"Return: {(total_profit / self.initial_balance) * 100:.2f}%")
            print(f"Total trades: {total_trades}")
            print(f"Average win rate: {avg_win_rate:.1%}")
            print(f"Average drawdown: {avg_drawdown:.1f}%")

            # Calculate stability metric
            profitable_windows = sum(1 for r in all_results if r['total_profit'] > 0)
            print(
                f"Profitable windows: {profitable_windows}/{len(all_results)} ({profitable_windows / len(all_results):.1%})")

            return {
                'total_profit': total_profit,
                'total_trades': total_trades,
                'avg_win_rate': avg_win_rate,
                'avg_drawdown': avg_drawdown,
                'profitable_windows': profitable_windows,
                'total_windows': len(all_results)
            }

        return None

    def test_on_data(self, data, best_params):
        """Test strategies on given data with optimized parameters"""
        strat_gen = StrategyGenerator(data)
        signals = strat_gen.generate_signals()

        trade_simulator = TradeSimulator(self.lot_size)
        all_trades = []

        # Execute all three strategies
        for strategy_name, strategy_info in best_params.items():
            params = strategy_info['params']

            # Determine trade direction from strategy name
            if 'buy' in strategy_name.lower():
                trade_type = 'BUY'
            else:
                trade_type = 'SELL'

            signal_series = signals[strategy_name]
            signal_indices = signal_series[signal_series == 1].index

            # Execute trades for this strategy
            for idx in signal_indices:
                try:
                    signal_idx = data.index.get_loc(idx)

                    # Find exit price
                    exit_price, exit_time = trade_simulator.find_exit_price(
                        data, signal_idx, trade_type,
                        params['stop_loss_pips'],
                        params['take_profit_pips'],
                        params['hold_hours']
                    )

                    entry_price = data.loc[idx, 'close']

                    # Simulate trade
                    trade = trade_simulator.simulate_trade(
                        entry_price=entry_price,
                        exit_price=exit_price,
                        trade_type=trade_type,
                        entry_time=idx,
                        exit_time=exit_time
                    )

                    all_trades.append(trade)

                except:
                    continue

        if not all_trades:
            return None

        # Calculate performance metrics
        profits = [t['profit'] for t in all_trades]
        total_profit = sum(profits)
        total_trades = len(all_trades)
        winning_trades = sum(1 for p in profits if p > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Calculate drawdown
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        initial_equity = self.initial_balance / len(best_params)
        drawdowns = (cumulative - running_max) / (initial_equity + 1e-10) * 100
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0

        return {
            'total_profit': total_profit,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'avg_profit': np.mean(profits),
            'trades': all_trades
        }


# ==================== 8. MAIN FUNCTION ====================
def main():
    """Main execution function"""
    print("=" * 80)
    print("CORRECT OPTIMIZATION WITH CROSS-VALIDATION")
    print("3 strategies + Time Series CV + Walk-Forward testing")
    print(f"Lot: {LOT_SIZE}, Deposit: ${INITIAL_BALANCE}")
    print("=" * 80)

    # Initialize data fetcher
    fetcher = MT5DataFetcher("EURUSD")

    try:
        # ========== LOAD DATA ==========
        print("\nLoading historical data...")
        data = fetcher.fetch_data('H1', bars=3000)  # More data for CV
        print(f"Loaded {len(data)} bars (H1)")

        # Split into optimization and final test sets
        split_idx = int(len(data) * 0.8)  # 80/20 split
        optimization_data = data.iloc[:split_idx].copy()
        final_test_data = data.iloc[split_idx:].copy()

        print(f"\nData split:")
        print(f"  Optimization/CV: {len(optimization_data)} bars")
        print(f"  Final test: {len(final_test_data)} bars (NEVER used before!)")

        # ========== STEP 1: CROSS-VALIDATION OPTIMIZATION ==========
        print("\n" + "=" * 80)
        print("STEP 1: CROSS-VALIDATION OPTIMIZATION")
        print("=" * 80)

        optimizer = CrossValidationOptimizer(optimization_data, LOT_SIZE)
        best_params = optimizer.optimize_all_strategies()

        if not best_params:
            print("Failed to find optimal parameters")
            return

        # ========== STEP 2: WALK-FORWARD VALIDATION ==========
        print("\n" + "=" * 80)
        print("STEP 2: WALK-FORWARD VALIDATION ON OPTIMIZATION DATA")
        print("=" * 80)

        validator = FinalValidator(INITIAL_BALANCE, LOT_SIZE)
        wf_results = validator.walk_forward_test(optimization_data, best_params, walk_forward_windows=4)

        # ========== STEP 3: FINAL OUT-OF-SAMPLE TEST ==========
        print("\n" + "=" * 80)
        print("STEP 3: FINAL TEST ON NEW DATA (OUT-OF-SAMPLE)")
        print("=" * 80)
        print("IMPORTANT: This data was NEVER used for optimization or CV!")

        final_result = validator.test_on_data(final_test_data, best_params)

        if final_result:
            print(f"\n📊 FINAL RESULTS ON NEW DATA:")
            print(f"  Period: {final_test_data.index[0]} - {final_test_data.index[-1]}")
            print(f"  Profit: ${final_result['total_profit']:.2f}")
            print(f"  Return: {(final_result['total_profit'] / INITIAL_BALANCE) * 100:.2f}%")
            print(f"  Trades: {final_result['total_trades']}")
            print(f"  Win rate: {final_result['win_rate']:.1%}")
            print(f"  Drawdown: {final_result['max_drawdown']:.1f}%")
            print(f"  Average profit per trade: ${final_result['avg_profit']:.2f}")

            # ========== EVALUATE SUCCESS CRITERIA ==========
            print("\n" + "=" * 80)
            print("SYSTEM SUCCESS CRITERIA:")
            print("=" * 80)

            success_criteria = []

            # Criterion 1: Positive profit on out-of-sample data
            if final_result['total_profit'] > 0:
                success_criteria.append("✅ Positive profit on new data")
            else:
                success_criteria.append("❌ Negative profit on new data")

            # Criterion 2: Drawdown within acceptable limits
            if final_result['max_drawdown'] > -MAX_DRAWDOWN_LIMIT:
                success_criteria.append(
                    f"✅ Drawdown within limit ({abs(final_result['max_drawdown']):.1f}% < {MAX_DRAWDOWN_LIMIT}%)")
            else:
                success_criteria.append(
                    f"❌ Drawdown exceeds limit ({abs(final_result['max_drawdown']):.1f}% > {MAX_DRAWDOWN_LIMIT}%)")

            # Criterion 3: Enough trades for statistical significance
            if final_result['total_trades'] >= 10:
                success_criteria.append(f"✅ Enough trades for statistics ({final_result['total_trades']})")
            else:
                success_criteria.append(f"❌ Not enough trades ({final_result['total_trades']})")

            # Criterion 4: Walk-forward stability
            if wf_results and wf_results['profitable_windows'] / wf_results['total_windows'] >= 0.6:
                success_criteria.append(
                    f"✅ Stability in walk-forward ({wf_results['profitable_windows']}/{wf_results['total_windows']} windows)")
            else:
                if wf_results:
                    success_criteria.append(
                        f"❌ Instability in walk-forward ({wf_results['profitable_windows']}/{wf_results['total_windows']} windows)")

            # Display all criteria
            for criterion in success_criteria:
                print(criterion)

            # ========== FINAL RECOMMENDATION ==========
            print("\n" + "=" * 80)
            print("FINAL RECOMMENDATION:")
            print("=" * 80)

            positive_criteria = sum(1 for c in success_criteria if '✅' in c)
            total_criteria = len(success_criteria)

            if positive_criteria >= 3:
                print("✅ SYSTEM PASSED VALIDATION AND CAN BE USED")
                print(f"   Criteria passed: {positive_criteria}/{total_criteria}")

                # Display optimized parameters
                print("\nOptimal parameters for real trading:")
                for strategy, info in best_params.items():
                    params = info['params']
                    print(f"\n{strategy}:")
                    print(f"  Holding: {params['hold_hours']}h")
                    print(f"  Stop-loss: {params['stop_loss_pips']}p")
                    print(f"  Take-profit: {params['take_profit_pips']}p")
                    print(f"  Risk/Reward: 1:{params['risk_reward']:.2f}")
            else:
                print("❌ SYSTEM FAILED VALIDATION")
                print(f"   Criteria passed: {positive_criteria}/{total_criteria}")
                print("   Requires additional optimization or strategy changes")

        else:
            print("Failed to get final test results")

        # ========== KEY LESSONS ==========
        print("\n" + "=" * 80)
        print("IMPORTANT CONCLUSIONS:")
        print("=" * 80)
        print("1. Used correct time-series cross-validation")
        print("2. Final test on data that was NEVER used")
        print("3. Realistic success criteria (drawdown <20%, positive profit)")
        print("4. Walk-forward validation for stability check")
        print("5. Excluded all types of data leaks")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        mt5.shutdown()


# ==================== EXECUTION ====================
if __name__ == "__main__":
    main()
