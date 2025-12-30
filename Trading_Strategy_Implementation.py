import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. CONFIGURATION ====================
LOT_SIZE = 0.1  # Lot size
POINT_VALUE = 10  # $10 per point on standard lot
MIN_PROFIT_PIPS = 10  # Minimum target in pips
MAX_LOSS_PIPS = 20  # Maximum stop-loss in pips
TRADE_DURATION = 24  # Maximum trade duration (hours)
COMMISSION = 0.5  # Commission per trade in $
SLIPPAGE = 0.5  # Slippage in pips

# ==================== 2. DATA FETCHING CLASS ====================
class MT5DataFetcher:
    def __init__(self, symbol="EURUSD"):
        self.symbol = symbol
        if not mt5.initialize():
            print("MT5 initialization error")
            raise ConnectionError("Failed to connect to MT5")
    
    def fetch_data(self, timeframe, bars=5000):
        tf_mapping = {
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        
        if timeframe not in tf_mapping:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        rates = mt5.copy_rates_from_pos(
            self.symbol, 
            tf_mapping[timeframe], 
            0, 
            bars
        )
        
        if rates is None or len(rates) == 0:
            raise ValueError(f"Failed to fetch data for {timeframe}")
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df

# ==================== 3. STRATEGY GENERATOR ====================
class StrategyGenerator:
    def __init__(self, df):
        self.df = df.copy()
    
    def calculate_rsi(self, period=14):
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def strategy_rsi(self):
        rsi = self.calculate_rsi(14)
        # Clear signals only
        buy_signal = ((rsi < 30) & (rsi.shift(1) >= 30)).astype(int)
        sell_signal = ((rsi > 70) & (rsi.shift(1) <= 70)).astype(int)
        return {'RSI_buy': buy_signal, 'RSI_sell': sell_signal}
    
    def strategy_stochastic(self):
        low_14 = self.df['low'].rolling(14).min()
        high_14 = self.df['high'].rolling(14).max()
        k = 100 * ((self.df['close'] - low_14) / (high_14 - low_14))
        d = k.rolling(3).mean()
        # Signals on crossover only
        buy_signal = ((k < 20) & (d < 20) & (k > d) & (k.shift(1) <= d.shift(1))).astype(int)
        sell_signal = ((k > 80) & (d > 80) & (k < d) & (k.shift(1) >= d.shift(1))).astype(int)
        return {'STOCH_buy': buy_signal, 'STOCH_sell': sell_signal}
    
    def strategy_macd(self):
        exp1 = self.df['close'].ewm(span=12).mean()
        exp2 = self.df['close'].ewm(span=26).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=9).mean()
        # Signals on crossover only
        buy_signal = ((macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))).astype(int)
        sell_signal = ((macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))).astype(int)
        return {'MACD_buy': buy_signal, 'MACD_sell': sell_signal}
    
    def strategy_moving_averages(self):
        ma_fast = self.df['close'].rolling(9).mean()
        ma_slow = self.df['close'].rolling(21).mean()
        # Signals on crossover only
        buy_signal = ((ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))).astype(int)
        sell_signal = ((ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))).astype(int)
        return {'MA_buy': buy_signal, 'MA_sell': sell_signal}
    
    def strategy_bollinger(self):
        sma20 = self.df['close'].rolling(20).mean()
        std20 = self.df['close'].rolling(20).std()
        upper_band = sma20 + 2 * std20
        lower_band = sma20 - 2 * std20
        # Signals on breakout only
        buy_signal = ((self.df['close'] < lower_band) & (self.df['close'].shift(1) >= lower_band.shift(1))).astype(int)
        sell_signal = ((self.df['close'] > upper_band) & (self.df['close'].shift(1) <= upper_band.shift(1))).astype(int)
        return {'BB_buy': buy_signal, 'BB_sell': sell_signal}
    
    def generate_all_signals(self):
        signals = {}
        strategies = [
            self.strategy_rsi(),
            self.strategy_stochastic(),
            self.strategy_macd(),
            self.strategy_moving_averages(),
            self.strategy_bollinger()
        ]
        
        for strategy in strategies:
            signals.update(strategy)
        
        return pd.DataFrame(signals)

# ==================== 4. TRADING SIMULATOR ====================
class TradeSimulator:
    def __init__(self, lot_size=0.1, commission=0.5, slippage=0.5):
        self.lot_size = lot_size
        self.commission = commission
        self.slippage = slippage  # in pips
        self.point_value = 10  # $10 per point on standard lot
    
    def simulate_trade(self, entry_price, exit_price, trade_type, 
                      entry_time, exit_time, stop_loss=None, take_profit=None):
        """Simulate a single trade with realistic parameters"""
        
        # Account for slippage
        if trade_type == 'BUY':
            effective_entry = entry_price + (self.slippage / 10000)
        else:
            effective_entry = entry_price - (self.slippage / 10000)
        
        # Calculate profit/loss
        if trade_type == 'BUY':
            pips = (exit_price - effective_entry) * 10000
        else:
            pips = (effective_entry - exit_price) * 10000
        
        # Profit in dollars
        profit = pips * self.point_value * self.lot_size - (2 * self.commission)  # entry + exit
        
        # Check stop-loss and take-profit
        hit_sl = False
        hit_tp = False
        
        if stop_loss:
            if trade_type == 'BUY' and exit_price <= stop_loss:
                hit_sl = True
            elif trade_type == 'SELL' and exit_price >= stop_loss:
                hit_sl = True
        
        if take_profit:
            if trade_type == 'BUY' and exit_price >= take_profit:
                hit_tp = True
            elif trade_type == 'SELL' and exit_price <= take_profit:
                hit_tp = True
        
        return {
            'entry_price': entry_price,
            'effective_entry': effective_entry,
            'exit_price': exit_price,
            'trade_type': trade_type,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'pips': pips,
            'profit': profit,
            'duration_hours': (exit_time - entry_time).total_seconds() / 3600,
            'hit_sl': hit_sl,
            'hit_tp': hit_tp
        }
    
    def find_exit_price(self, data, entry_idx, trade_type, max_bars=TRADE_DURATION):
        """Find exit price based on stop-loss, take-profit, or time"""
        entry_price = data.iloc[entry_idx]['close']
        entry_time = data.index[entry_idx]
        
        # Set stop-loss and take-profit
        if trade_type == 'BUY':
            stop_loss = entry_price - (MAX_LOSS_PIPS / 10000)
            take_profit = entry_price + (MIN_PROFIT_PIPS / 10000)
        else:
            stop_loss = entry_price + (MAX_LOSS_PIPS / 10000)
            take_profit = entry_price - (MIN_PROFIT_PIPS / 10000)
        
        # Search for exit in following bars
        exit_idx = entry_idx + 1
        exit_price = None
        exit_time = None
        
        while exit_idx < len(data) and exit_idx <= entry_idx + max_bars:
            current_bar = data.iloc[exit_idx]
            high = current_bar['high']
            low = current_bar['low']
            close = current_bar['close']
            
            if trade_type == 'BUY':
                # Check take-profit
                if high >= take_profit:
                    exit_price = take_profit
                    exit_time = data.index[exit_idx]
                    break
                # Check stop-loss
                elif low <= stop_loss:
                    exit_price = stop_loss
                    exit_time = data.index[exit_idx]
                    break
            else:  # SELL
                # Check take-profit
                if low <= take_profit:
                    exit_price = take_profit
                    exit_time = data.index[exit_idx]
                    break
                # Check stop-loss
                elif high >= stop_loss:
                    exit_price = stop_loss
                    exit_time = data.index[exit_idx]
                    break
            
            exit_idx += 1
        
        # If stops weren't triggered, exit at the last bar's close
        if exit_price is None:
            exit_idx = min(entry_idx + max_bars, len(data) - 1)
            exit_price = data.iloc[exit_idx]['close']
            exit_time = data.index[exit_idx]
        
        return exit_price, exit_time, stop_loss, take_profit

# ==================== 5. STRATEGY EVALUATOR ====================
class StrategyEvaluator:
    def __init__(self, price_data, lot_size=0.1):
        self.price_data = price_data
        self.lot_size = lot_size
        self.trade_simulator = TradeSimulator(lot_size)
    
    def evaluate_strategy(self, signals, signal_name, min_bars_between_trades=10):
        """Evaluate strategy with realistic parameters"""
        if signal_name not in signals.columns:
            return None
        
        signal_series = signals[signal_name]
        signal_indices = signal_series[signal_series == 1].index
        
        if len(signal_indices) == 0:
            return None
        
        trades = []
        total_profit = 0
        last_trade_idx = -min_bars_between_trades
        
        for idx in signal_indices:
            try:
                signal_idx = self.price_data.index.get_loc(idx)
                
                # Check minimum distance between trades
                if signal_idx - last_trade_idx < min_bars_between_trades:
                    continue
                
                # Determine trade type
                if 'buy' in signal_name.lower():
                    trade_type = 'BUY'
                else:
                    trade_type = 'SELL'
                
                # Find exit price
                exit_price, exit_time, stop_loss, take_profit = \
                    self.trade_simulator.find_exit_price(
                        self.price_data, signal_idx, trade_type
                    )
                
                entry_price = self.price_data.loc[idx, 'close']
                
                # Simulate trade
                trade = self.trade_simulator.simulate_trade(
                    entry_price=entry_price,
                    exit_price=exit_price,
                    trade_type=trade_type,
                    entry_time=idx,
                    exit_time=exit_time,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                trades.append(trade)
                total_profit += trade['profit']
                last_trade_idx = signal_idx
                
            except Exception as e:
                continue
        
        if not trades:
            return None
        
        profits = [t['profit'] for t in trades]
        winning_trades = sum(1 for p in profits if p > 0)
        total_trades = len(trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_profit = np.mean(profits)
        avg_win = np.mean([p for p in profits if p > 0]) if any(p > 0 for p in profits) else 0
        avg_loss = np.mean([p for p in profits if p < 0]) if any(p < 0 for p in profits) else 0
        
        # Calculate maximum drawdown
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / 10000  # Drawdown as percentage of deposit
        max_drawdown_pct = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        # Profit factor
        gross_profit = sum(p for p in profits if p > 0)
        gross_loss = abs(sum(p for p in profits if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Average trade duration
        avg_duration = np.mean([t['duration_hours'] for t in trades])
        
        return {
            'signal_name': signal_name,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'total_profit': total_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown_pct': max_drawdown_pct,
            'profit_factor': profit_factor,
            'avg_duration_hours': avg_duration,
            'trades': trades
        }
    
    def evaluate_all_strategies(self, signals):
        results = []
        for signal_name in signals.columns:
            result = self.evaluate_strategy(signals, signal_name, min_bars_between_trades=5)
            if result and result['total_trades'] > 0:
                results.append(result)
        return results

# ==================== 6. COMPARISON SYSTEM ====================
class TradingSystem:
    def __init__(self, initial_balance=10000, lot_size=0.1):
        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.trade_simulator = TradeSimulator(lot_size)
    
    def run_direct_trading(self, data, signals, strategy_results, top_n=3):
        """Direct trading using best strategies"""
        print("\n" + "=" * 80)
        print("DIRECT TRADING - REALISTIC BACKTEST")
        print("=" * 80)
        
        # Select best strategies
        best_strategies = sorted(strategy_results, key=lambda x: x['total_profit'], reverse=True)[:top_n]
        
        if not best_strategies:
            print("No profitable strategies for direct trading")
            return None
        
        all_trades = []
        balance = self.initial_balance
        equity_curve = [balance]
        open_trades = []
        
        # Simulate trading along timeline
        for i in range(len(data)):
            current_time = data.index[i]
            current_price = data.iloc[i]['close']
            
            # Close trades if stops triggered or time expired
            trades_to_close = []
            for trade in open_trades:
                if trade['exit_time'] <= current_time:
                    trades_to_close.append(trade)
                    balance += trade['profit']
                    all_trades.append(trade)
                    equity_curve.append(balance)
            
            # Remove closed trades
            for trade in trades_to_close:
                open_trades.remove(trade)
            
            # Check signals for new trades
            for strategy in best_strategies:
                signal_name = strategy['signal_name']
                if signal_name in signals.columns and i < len(signals):
                    if signals.iloc[i][signal_name] == 1:
                        # Check if there's already an open trade
                        if len(open_trades) < 3:  # Max 3 simultaneous trades
                            # Determine trade type
                            if 'buy' in signal_name.lower():
                                trade_type = 'BUY'
                            else:
                                trade_type = 'SELL'
                            
                            # Find exit price
                            exit_price, exit_time, stop_loss, take_profit = \
                                self.trade_simulator.find_exit_price(data, i, trade_type)
                            
                            # Simulate trade
                            trade = self.trade_simulator.simulate_trade(
                                entry_price=current_price,
                                exit_price=exit_price,
                                trade_type=trade_type,
                                entry_time=current_time,
                                exit_time=exit_time,
                                stop_loss=stop_loss,
                                take_profit=take_profit
                            )
                            
                            open_trades.append(trade)
        
        # Close all open trades at the end
        for trade in open_trades:
            balance += trade['profit']
            all_trades.append(trade)
            equity_curve.append(balance)
        
        # Analyze results
        if all_trades:
            profits = [t['profit'] for t in all_trades]
            total_profit = sum(profits)
            total_trades = len(all_trades)
            winning_trades = sum(1 for p in profits if p > 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            avg_profit = np.mean(profits)
            max_profit = max(profits) if profits else 0
            min_profit = min(profits) if profits else 0
            
            # Calculate drawdown
            equity_array = np.array(equity_curve)
            running_max = np.maximum.accumulate(equity_array)
            drawdowns = (equity_array - running_max) / running_max * 100
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
            
            return {
                'strategy_type': 'DIRECT',
                'strategies_used': [s['signal_name'] for s in best_strategies],
                'initial_balance': self.initial_balance,
                'final_balance': balance,
                'total_profit': total_profit,
                'total_return_pct': (total_profit / self.initial_balance) * 100,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'max_profit': max_profit,
                'min_profit': min_profit,
                'max_drawdown_pct': max_drawdown,
                'avg_trade_duration': np.mean([t['duration_hours'] for t in all_trades]) if all_trades else 0,
                'profit_factor': abs(sum(p for p in profits if p > 0) / 
                                   (abs(sum(p for p in profits if p < 0)) + 1e-10))
            }
        
        return None
    
    def run_contrarian_trading(self, data, signals, strategy_results, top_n=3):
        """Contrarian trading strategy"""
        print("\n" + "=" * 80)
        print("CONTRARIAN TRADING - REALISTIC BACKTEST")
        print("=" * 80)
        
        # Select worst strategies
        worst_strategies = sorted(strategy_results, key=lambda x: x['total_profit'])[:top_n]
        
        if not worst_strategies:
            print("No losing strategies to invert")
            return None
        
        # Create inverted signals
        contrarian_signals = pd.DataFrame(index=signals.index)
        
        for strategy in worst_strategies:
            signal_name = strategy['signal_name']
            original_signal = signals[signal_name]
            
            # Invert signal
            if 'buy' in signal_name.lower():
                new_name = signal_name.replace('buy', 'SELL_INV')
                contrarian_signals[new_name] = original_signal
            else:
                new_name = signal_name.replace('sell', 'BUY_INV')
                contrarian_signals[new_name] = original_signal
        
        # Evaluate inverted strategies
        evaluator = StrategyEvaluator(data, self.lot_size)
        contrarian_results = []
        
        for col in contrarian_signals.columns:
            result = evaluator.evaluate_strategy(contrarian_signals, col, min_bars_between_trades=5)
            if result and result['total_trades'] > 0:
                contrarian_results.append(result)
        
        if not contrarian_results:
            print("No results for inverted strategies")
            return None
        
        # Trade using inverted strategies
        return self.run_direct_trading(
            data, contrarian_signals, contrarian_results, top_n=len(contrarian_results)
        )

# ==================== 7. MAIN FUNCTION ====================
def main():
    print("=" * 80)
    print("REALISTIC TRADING SYSTEM FOR EURUSD")
    print(f"Lot: {LOT_SIZE}, TP: {MIN_PROFIT_PIPS}pips, SL: {MAX_LOSS_PIPS}pips")
    print("=" * 80)
    
    # Initialization
    fetcher = MT5DataFetcher("EURUSD")
    
    try:
        # Load data
        print("\nLoading historical data...")
        data = fetcher.fetch_data('H1', bars=2000)  # Reduced for speed
        print(f"Loaded {len(data)} bars (H1)")
        print(f"Period: {data.index[0]} - {data.index[-1]}")
        
        # Split data
        split_idx = int(len(data) * 0.7)
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()
        
        print(f"\nData split:")
        print(f"  Training: {len(train_data)} bars")
        print(f"  Test: {len(test_data)} bars")
        
        # Generate signals
        print("\nGenerating signals for 5 strategies...")
        strat_gen = StrategyGenerator(train_data)
        train_signals = strat_gen.generate_all_signals()
        
        # Evaluate strategies
        print("\nEvaluating strategy performance...")
        evaluator = StrategyEvaluator(train_data, LOT_SIZE)
        strategy_results = evaluator.evaluate_all_strategies(train_signals)
        
        if not strategy_results:
            print("No strategy evaluation results")
            return
        
        # Sort by profitability
        strategy_results.sort(key=lambda x: x['total_profit'], reverse=True)
        
        # ==================== REPORT 1: PROFITABLE STRATEGIES ====================
        print("\n" + "=" * 80)
        print("REPORT 1: PROFITABLE STRATEGIES")
        print("=" * 80)
        
        profitable_strategies = [s for s in strategy_results if s['total_profit'] > 0]
        if profitable_strategies:
            print(f"\nFound {len(profitable_strategies)} profitable strategies:")
            print("-" * 80)
            for i, strategy in enumerate(profitable_strategies, 1):
                print(f"\n{i}. {strategy['signal_name']}:")
                print(f"   Trades: {strategy['total_trades']}")
                print(f"   Win Rate: {strategy['win_rate']:.1%}")
                print(f"   Total Profit: ${strategy['total_profit']:.2f}")
                print(f"   Average Trade: ${strategy['avg_profit']:.2f}")
                print(f"   Average Win: ${strategy['avg_win']:.2f}")
                print(f"   Average Loss: ${strategy['avg_loss']:.2f}")
                print(f"   Max Drawdown: {strategy['max_drawdown_pct']:.1%}")
                print(f"   Profit Factor: {strategy['profit_factor']:.2f}")
                print(f"   Duration: {strategy['avg_duration_hours']:.1f} hrs")
        else:
            print("\nNo profitable strategies found")
        
        # ==================== REPORT 2: LOSING STRATEGIES ====================
        print("\n" + "=" * 80)
        print("REPORT 2: LOSING STRATEGIES (for inversion)")
        print("=" * 80)
        
        losing_strategies = [s for s in strategy_results if s['total_profit'] < 0]
        if losing_strategies:
            print(f"\nFound {len(losing_strategies)} losing strategies:")
            print("-" * 80)
            for i, strategy in enumerate(losing_strategies, 1):
                print(f"\n{i}. {strategy['signal_name']}:")
                print(f"   Trades: {strategy['total_trades']}")
                print(f"   Win Rate: {strategy['win_rate']:.1%}")
                print(f"   Total Loss: ${strategy['total_profit']:.2f}")
                print(f"   Average Loss: ${strategy['avg_profit']:.2f}")
                print(f"   Inversion Potential:")
                potential_profit = -strategy['total_profit']
                potential_win_rate = 1 - strategy['win_rate']
                print(f"   - Profit: ${potential_profit:.2f}")
                print(f"   - Win Rate: {potential_win_rate:.1%}")
                print(f"   - Profit Factor: {1/strategy['profit_factor']:.2f}")
        else:
            print("\nNo losing strategies found")
        
        # ==================== BACKTESTS ====================
        print("\n" + "=" * 80)
        print("REALISTIC BACKTESTS")
        print("=" * 80)
        
        trading_system = TradingSystem(initial_balance=10000, lot_size=LOT_SIZE)
        
        # 1. Direct trading
        direct_result = trading_system.run_direct_trading(
            train_data, train_signals, profitable_strategies[:3] if profitable_strategies else []
        )
        
        # 2. Contrarian trading
        contrarian_result = trading_system.run_contrarian_trading(
            train_data, train_signals, losing_strategies[:3] if losing_strategies else []
        )
        
        # ==================== COMPARISON SUMMARY ====================
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        
        if direct_result:
            print("\n📍 DIRECT TRADING (best strategies):")
            print(f"   Strategies: {', '.join(direct_result['strategies_used'])}")
            print(f"   Initial Deposit: ${direct_result['initial_balance']:.2f}")
            print(f"   Final Balance: ${direct_result['final_balance']:.2f}")
            print(f"   Total Profit: ${direct_result['total_profit']:.2f}")
            print(f"   Return: {direct_result['total_return_pct']:.2f}%")
            print(f"   Trades: {direct_result['total_trades']}")
            print(f"   Winning: {direct_result['winning_trades']} ({direct_result['win_rate']:.1%})")
            print(f"   Average Profit: ${direct_result['avg_profit']:.2f}")
            print(f"   Max Profit: ${direct_result['max_profit']:.2f}")
            print(f"   Min Profit: ${direct_result['min_profit']:.2f}")
            print(f"   Max Drawdown: {direct_result['max_drawdown_pct']:.1f}%")
            print(f"   Profit Factor: {direct_result['profit_factor']:.2f}")
            print(f"   Average Duration: {direct_result['avg_trade_duration']:.1f} hrs")
        
        if contrarian_result:
            print("\n🔄 CONTRARIAN TRADING (inverted worst strategies):")
            print(f"   Strategies: {', '.join(contrarian_result['strategies_used'])}")
            print(f"   Initial Deposit: ${contrarian_result['initial_balance']:.2f}")
            print(f"   Final Balance: ${contrarian_result['final_balance']:.2f}")
            print(f"   Total Profit: ${contrarian_result['total_profit']:.2f}")
            print(f"   Return: {contrarian_result['total_return_pct']:.2f}%")
            print(f"   Trades: {contrarian_result['total_trades']}")
            print(f"   Winning: {contrarian_result['winning_trades']} ({contrarian_result['win_rate']:.1%})")
            print(f"   Average Profit: ${contrarian_result['avg_profit']:.2f}")
            print(f"   Max Profit: ${contrarian_result['max_profit']:.2f}")
            print(f"   Min Profit: ${contrarian_result['min_profit']:.2f}")
            print(f"   Max Drawdown: {contrarian_result['max_drawdown_pct']:.1f}%")
            print(f"   Profit Factor: {contrarian_result['profit_factor']:.2f}")
            print(f"   Average Duration: {contrarian_result['avg_trade_duration']:.1f} hrs")
        
        # ==================== RECOMMENDATIONS ====================
        print("\n" + "=" * 80)
        print("REAL TRADING RECOMMENDATIONS")
        print("=" * 80)
        
        if direct_result and contrarian_result:
            if direct_result['total_profit'] > contrarian_result['total_profit']:
                print("\n✅ RECOMMENDATION: Use direct trading")
                profit_diff = direct_result['total_profit'] - contrarian_result['total_profit']
                print(f"   Profit advantage: ${profit_diff:.2f}")
            else:
                print("\n✅ RECOMMENDATION: Use contrarian trading")
                profit_diff = contrarian_result['total_profit'] - direct_result['total_profit']
                print(f"   Profit advantage: ${profit_diff:.2f}")
            
            # Compare stability
            if direct_result['win_rate'] > contrarian_result['win_rate']:
                print(f"   Direct trading is more stable (win rate: {direct_result['win_rate']:.1%} vs {contrarian_result['win_rate']:.1%})")
            else:
                print(f"   Contrarian trading is more stable (win rate: {contrarian_result['win_rate']:.1%} vs {direct_result['win_rate']:.1%})")
            
            # Compare drawdowns
            if abs(direct_result['max_drawdown_pct']) < abs(contrarian_result['max_drawdown_pct']):
                print(f"   Direct trading is safer (drawdown: {direct_result['max_drawdown_pct']:.1f}% vs {contrarian_result['max_drawdown_pct']:.1f}%)")
            else:
                print(f"   Contrarian trading is safer (drawdown: {contrarian_result['max_drawdown_pct']:.1f}% vs {direct_result['max_drawdown_pct']:.1f}%)")
        
        # Top strategies
        print("\n" + "=" * 80)
        print("TOP STRATEGIES")
        print("=" * 80)
        
        print("\n🎯 For direct trading:")
        if profitable_strategies:
            for i, strategy in enumerate(profitable_strategies[:3], 1):
                print(f"   {i}. {strategy['signal_name']} (profit: ${strategy['total_profit']:.2f}, win rate: {strategy['win_rate']:.1%})")
        else:
            print("   No profitable strategies")
        
        print("\n🔄 For contrarian trading:")
        if losing_strategies:
            for i, strategy in enumerate(losing_strategies[:3], 1):
                potential = -strategy['total_profit']
                print(f"   {i}. Invert {strategy['signal_name']} (potential: ${potential:.2f}, win rate: {1-strategy['win_rate']:.1%})")
        else:
            print("   No losing strategies to invert")
        
        print("\n" + "=" * 80)
        print("IMPORTANT PARAMETERS FOR REAL TRADING:")
        print("=" * 80)
        print(f"   Lot Size: {LOT_SIZE}")
        print(f"   Take-Profit: {MIN_PROFIT_PIPS} pips")
        print(f"   Stop-Loss: {MAX_LOSS_PIPS} pips")
        print(f"   Commission: ${COMMISSION:.2f} per trade")
        print(f"   Slippage: {SLIPPAGE} pips")
        print(f"   Max Trade Duration: {TRADE_DURATION} hours")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        mt5.shutdown()

# ==================== EXECUTION ====================
if __name__ == "__main__":
    main()
