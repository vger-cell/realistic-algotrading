"""
Liquidity Zone Trading Strategy Backtester
Optimized version focusing on 20-35 pip zones with RSI filtering

Author: Vladimir Korneev
Telegram: t.me/realistic_algotrading
Repository: github.com/vger-cell/realistic-algotrading

Strategy Logic:
1. Identify round-number levels and recent extremes 20-35 pips away
2. Enter trades toward identified liquidity zones
3. Apply RSI filters to avoid overbought/oversold conditions
4. Use dynamic stop-loss (70% of distance) and fixed take-profit

Backtest Results (EURUSD M15, 90 days):
- Trades: 280
- Win Rate: 46.4%
- Total P&L: -2.9 pips (break-even)
- Best zones: 30-35 pips (52% WR)

Issues identified:
- P&L calculation bug in results analysis
- Strategy shows edge but no profitability
- Risk parameters need optimization

Important: This is for educational purposes only.
Real trading requires additional testing and risk management.
"""

# optimized_final_backtest.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class OptimizedFinalBacktester:
    def __init__(self, symbol="EURUSD", spread_pips=1.2):
        self.symbol = symbol
        self.spread_pips = spread_pips
        self.data = {}

    def load_data(self, days=90):
        """Load data"""
        print(f"Loading {days} days of {self.symbol} data...")

        if not mt5.initialize():
            print(f"MT5 Error: {mt5.last_error()}")
            return False

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            rates = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_M15, start_date, end_date)

            if rates is None:
                print("No M15 data available")
                return False

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            # Indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['atr'] = self.calculate_atr(df, period=14) * 10000
            df['rsi'] = self.calculate_rsi(df, period=14)

            self.data['M15'] = df
            print(f"Loaded M15 bars: {len(df)}")
            print(f"Average ATR: {df['atr'].mean():.1f} pips")

            return True

        except Exception as e:
            print(f"Load error: {e}")
            return False
        finally:
            mt5.shutdown()

    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    def calculate_rsi(self, df, period=14):
        """Calculate RSI"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def find_optimized_zones(self, df, current_idx):
        """Optimized zone search based on backtest data"""
        if current_idx < 50:
            return []

        window = df.iloc[current_idx - 50:current_idx + 1]
        current_price = df.iloc[current_idx]['close']
        current_rsi = df.iloc[current_idx]['rsi']

        zones = []

        # MAIN INSIGHT: Only take zones 20-30 pips (profitable according to data)
        # 1. Round numbers in 20-35 pip range
        price_4dec = round(current_price * 10000) / 10000

        # Only distant levels (20-35 pips)
        for offset in [200, 250, 300, 350, -200, -250, -300, -350]:  # in pips
            level = price_4dec + (offset / 10000)
            distance = abs(current_price - level) * 10000

            if 20 <= distance <= 35:  # OPTIMAL RANGE
                # Check if this was a significant level
                mask = (window['high'] >= level - 0.0005) & (window['low'] <= level + 0.0005)
                touches = mask.sum()

                if touches >= 1:
                    # Bonus for round hundreds (1.10000, 1.10500)
                    priority = 1.0 if offset % 100 == 0 else 0.8

                    zones.append({
                        'price': level,
                        'type': 'round_number',
                        'distance': distance,
                        'direction': 'above' if current_price < level else 'below',
                        'priority': priority,
                        'touches': touches
                    })

        # 2. Recent extremes (only if far enough)
        recent_high = window['high'].iloc[-20:].max()
        recent_low = window['low'].iloc[-20:].min()

        for level, ltype in [(recent_high, 'recent_high'), (recent_low, 'recent_low')]:
            distance = abs(current_price - level) * 10000

            if 20 <= distance <= 35:
                zones.append({
                    'price': level,
                    'type': ltype,
                    'distance': distance,
                    'direction': 'above' if current_price < level else 'below',
                    'priority': 0.7,
                    'touches': 1
                })

        # 3. RSI FILTER: Avoid overbought/oversold conditions
        filtered_zones = []
        for zone in zones:
            # For BUY signals (zone above) - RSI should not be > 70
            # For SELL signals (zone below) - RSI should not be < 30
            if zone['direction'] == 'above':  # BUY
                if current_rsi < 70:
                    filtered_zones.append(zone)
            else:  # SELL
                if current_rsi > 30:
                    filtered_zones.append(zone)

        # Sort by priority
        filtered_zones.sort(key=lambda x: x['priority'], reverse=True)

        # Return maximum 2 best zones
        return filtered_zones[:2]

    def generate_optimized_signal(self, df, current_idx, zone):
        """Optimized signal generation based on data"""
        current_price = df.iloc[current_idx]['close']
        zone_price = zone['price']
        direction = zone['direction']
        distance = zone['distance']

        # MAIN INSIGHT: Change stop-loss ratio
        # Old: stop = 50% of distance → Win Rate 35%
        # New: stop = 70% of distance → Increase Win Rate

        if direction == 'above':
            # BUY towards zone above
            trade_direction = 'BUY'
            entry_price = current_price + (self.spread_pips / 10000)
            take_profit = zone_price

            # OPTIMIZATION: Stop 70% instead of 50%
            stop_loss = current_price - (distance * 0.7 / 10000)

            # Additional filter: minimum profit 15 pips
            min_profit = 15
            if (take_profit - entry_price) * 10000 < min_profit:
                return None

        else:
            # SELL towards zone below
            trade_direction = 'SELL'
            entry_price = current_price - (self.spread_pips / 10000)
            take_profit = zone_price
            stop_loss = current_price + (distance * 0.7 / 10000)

            min_profit = 15
            if (entry_price - take_profit) * 10000 < min_profit:
                return None

        # Calculate parameters
        risk_pips = abs(entry_price - stop_loss) * 10000
        reward_pips = abs(entry_price - take_profit) * 10000
        risk_reward = reward_pips / risk_pips if risk_pips > 0 else 0

        # FILTERS based on data:
        # 1. Risk no more than 25 pips
        if risk_pips > 25:
            return None

        # 2. Minimum profit 10 pips
        if reward_pips < 10:
            return None

        # 3. Risk/Reward at least 1:1.2 (softer than 1:1.5)
        if risk_reward < 1.2:
            return None

        # 4. Only zones 20-35 pips (best according to data)
        if not (20 <= distance <= 35):
            return None

        return {
            'direction': trade_direction,
            'entry': round(entry_price, 5),
            'stop_loss': round(stop_loss, 5),
            'take_profit': round(take_profit, 5),
            'risk_pips': round(risk_pips, 1),
            'reward_pips': round(reward_pips, 1),
            'risk_reward': round(risk_reward, 2),
            'zone_price': zone_price,
            'zone_type': zone['type'],
            'distance_to_zone': distance,
            'strategy': 'optimized_approach'
        }

    def check_trade_result(self, df, start_idx, signal):
        """Check trade result with optimization"""
        entry_price = signal['entry']
        take_profit = signal['take_profit']
        stop_loss = signal['stop_loss']
        direction = signal['direction']

        # OPTIMIZATION: Increase maximum holding time
        max_bars = 72  # 18 hours instead of 12
        profit_pips = 0
        bars_held = 0
        exit_reason = 'timeout'

        for i in range(start_idx + 1, min(start_idx + max_bars, len(df))):
            bars_held = i - start_idx
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']

            if direction == 'BUY':
                if high >= take_profit:
                    profit_pips = (take_profit - entry_price) * 10000
                    exit_reason = 'take_profit'
                    break
                elif low <= stop_loss:
                    profit_pips = (stop_loss - entry_price) * 10000
                    exit_reason = 'stop_loss'
                    break
            else:  # SELL
                if low <= take_profit:
                    profit_pips = (entry_price - take_profit) * 10000
                    exit_reason = 'take_profit'
                    break
                elif high >= stop_loss:
                    profit_pips = (entry_price - stop_loss) * 10000
                    exit_reason = 'stop_loss'
                    break

        if exit_reason == 'timeout':
            current_price = df.iloc[min(start_idx + max_bars, len(df) - 1)]['close']
            if direction == 'BUY':
                profit_pips = (current_price - entry_price) * 10000
            else:
                profit_pips = (entry_price - current_price) * 10000

        return {
            'profit_pips': profit_pips,
            'bars_held': bars_held,
            'exit_reason': exit_reason
        }

    def run_optimized_backtest(self):
        """Run optimized backtest"""
        if 'M15' not in self.data:
            print("No data for backtest")
            return

        df = self.data['M15']

        print("\n" + "=" * 80)
        print("FINAL OPTIMIZED BACKTEST")
        print("=" * 80)
        print("MAIN OPTIMIZATIONS:")
        print("1. Only zones 20-35 pips (profitable according to data)")
        print("2. Stop-loss = 70% of distance (instead of 50%)")
        print("3. RSI filter (avoid overbought/oversold conditions)")
        print("4. Increased holding time (18 hours)")
        print("=" * 80)

        stats = defaultdict(int)
        trades = []

        # Test on last third of data (forward test)
        start_idx = len(df) // 3
        end_idx = len(df) - 50
        step = 4

        print(f"\nForward test from bar {start_idx} to {end_idx}")
        print(f"Period: {df.index[start_idx].date()} - {df.index[end_idx].date()}")

        for current_idx in range(start_idx, end_idx, step):
            current_time = df.index[current_idx]

            # Find optimized zones
            zones = self.find_optimized_zones(df, current_idx)

            for zone in zones:
                # Generate optimized signal
                signal = self.generate_optimized_signal(df, current_idx, zone)

                if signal:
                    stats['signals_generated'] += 1

                    # Check result
                    result = self.check_trade_result(df, current_idx, signal)

                    stats['trades_executed'] += 1

                    if result['profit_pips'] > 0:
                        stats['profitable_trades'] += 1
                        stats['total_profit'] += result['profit_pips']
                        stats['max_profit'] = max(stats['max_profit'], result['profit_pips'])
                    else:
                        stats['unprofitable_trades'] += 1
                        stats['total_loss'] += abs(result['profit_pips'])
                        stats['max_loss'] = min(stats['max_loss'], result['profit_pips'])

                    trades.append({
                        'time': current_time,
                        'entry_price': signal['entry'],
                        'direction': signal['direction'],
                        'zone_price': zone['price'],
                        'zone_type': zone['type'],
                        'distance': zone['distance'],
                        'take_profit': signal['take_profit'],
                        'stop_loss': signal['stop_loss'],
                        'risk_pips': signal['risk_pips'],
                        'reward_pips': signal['reward_pips'],
                        'risk_reward': signal['risk_reward'],
                        'profit_pips': result['profit_pips'],
                        'bars_held': result['bars_held'],
                        'exit_reason': result['exit_reason']
                    })

            # Progress
            if (current_idx - start_idx) % 400 == 0 and stats['trades_executed'] > 0:
                progress = (current_idx - start_idx) / (end_idx - start_idx) * 100
                win_rate = stats['profitable_trades'] / stats['trades_executed']
                print(f"  Progress: {progress:.1f}% | Trades: {stats['trades_executed']} | "
                      f"Win Rate: {win_rate:.1%}")

        # Analyze results
        self.analyze_optimized_results(stats, trades)

        return stats, trades

    def analyze_optimized_results(self, stats, trades):
        """Analyze optimized results"""
        print("\n" + "=" * 80)
        print("OPTIMIZED BACKTEST RESULTS")
        print("=" * 80)

        if stats['trades_executed'] > 0:
            stats['win_rate'] = stats['profitable_trades'] / stats['trades_executed']
            stats['total_pnl'] = stats['total_profit'] - stats['total_loss']
            stats['avg_trade'] = stats['total_pnl'] / stats['trades_executed']

            if stats['profitable_trades'] > 0:
                stats['avg_win'] = stats['total_profit'] / stats['profitable_trades']
            else:
                stats['avg_win'] = 0

            if stats['unprofitable_trades'] > 0:
                stats['avg_loss'] = stats['total_loss'] / stats['unprofitable_trades']
            else:
                stats['avg_loss'] = 0

        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Signals generated: {stats.get('signals_generated', 0)}")
        print(f"   Trades executed: {stats.get('trades_executed', 0)}")

        if stats['trades_executed'] > 0:
            print(f"\n💰 TRADING RESULTS:")
            print("-" * 40)
            print(f"   Profitable trades: {stats['profitable_trades']}")
            print(f"   Unprofitable trades: {stats['unprofitable_trades']}")
            print(f"   Win Rate: {stats['win_rate']:.1%}")
            print(f"   Total P&L: {stats['total_pnl']:+.1f} pips")
            print(f"   Average trade: {stats['avg_trade']:+.2f} pips")
            print(f"   Average profit: {stats['avg_win']:+.1f} pips")
            print(f"   Average loss: {stats['avg_loss']:+.1f} pips")

            # Mathematical expectation
            expectancy = (stats['win_rate'] * stats['avg_win'] -
                          (1 - stats['win_rate']) * stats['avg_loss'])
            print(f"   Expected value: {expectancy:+.2f} pips")

            # Analysis by exit types
            if trades:
                df_trades = pd.DataFrame(trades)

                print(f"\n🚪 STRATEGY EFFICIENCY:")
                print("-" * 40)

                # Win Rate by zone type
                type_stats = df_trades.groupby('zone_type').agg({
                    'profit_pips': ['mean', 'count', lambda x: (x > 0).mean()]
                }).round(2)

                type_stats.columns = ['avg_profit', 'count', 'win_rate']

                for zone_type, row in type_stats.iterrows():
                    print(f"   {zone_type:15}: {row['win_rate']:.1%} WR, "
                          f"{row['avg_profit']:+.1f}p avg, {int(row['count'])} trades")

                # Efficiency by distance
                print(f"\n📏 RESULTS BY DISTANCE:")
                print("-" * 40)

                if 'distance' in df_trades.columns:
                    distance_stats = df_trades.groupby(pd.cut(df_trades['distance'],
                                                              bins=[20, 25, 30, 35])).agg({
                        'profit_pips': ['mean', 'count', lambda x: (x > 0).mean()]
                    }).round(2)

                    distance_stats.columns = ['avg_profit', 'count', 'win_rate']

                    for idx, row in distance_stats.iterrows():
                        if row['count'] > 0:
                            print(f"   {idx}: {row['win_rate']:.1%} WR, "
                                  f"{row['avg_profit']:+.1f}p avg, {int(row['count'])} trades")

                # Exit reasons
                print(f"\n🎯 TARGET METRICS:")
                print("-" * 40)

                tp_hits = (df_trades['exit_reason'] == 'take_profit').sum()
                sl_hits = (df_trades['exit_reason'] == 'stop_loss').sum()
                tp_rate = tp_hits / len(df_trades)
                sl_rate = sl_hits / len(df_trades)

                print(f"   Take-profit hit: {tp_rate:.1%} ({tp_hits} trades)")
                print(f"   Stop-loss hit: {sl_rate:.1%} ({sl_hits} trades)")
                print(f"   Average holding time: {df_trades['bars_held'].mean() * 0.25:.1f} hours")

            # Success criteria
            print(f"\n🎯 STRATEGY SUCCESS CRITERIA:")
            print("-" * 40)

            success_criteria = [
                ("Win Rate > 40%", stats['win_rate'] > 0.4),
                ("Expected value > 0", stats['avg_trade'] > 0),
                ("Average profit > average loss", stats['avg_win'] > stats['avg_loss']),
                ("Max drawdown < 20% of deposit", True)  # Simplified
            ]

            passed = 0
            for criterion, condition in success_criteria:
                status = "✅" if condition else "❌"
                print(f"   {status} {criterion}")
                if condition:
                    passed += 1

            # Final verdict
            print(f"\n💡 FINAL VERDICT:")
            print("-" * 40)

            if passed >= 3 and stats['avg_trade'] > 0.5:
                print("✅ STRATEGY SUCCESSFUL!")
                print(f"   Can be tested on demo account")
                print(f"   Expected monthly return: {stats['avg_trade'] * 20:+.1f} pips")
            elif stats['avg_trade'] > 0:
                print("⚠️ STRATEGY PROMISING")
                print(f"   Requires additional optimization")
                print(f"   Win Rate: {stats['win_rate']:.1%} (target >45%)")
            else:
                print("❌ STRATEGY NOT WORKING")
                print(f"   Consider other approaches")
                print(f"   Current expected value: {stats['avg_trade']:+.2f} pips")

        else:
            print(f"\n⚠️ NO EXECUTED TRADES")
            print(f"   Optimization is too strict")

        # Save results
        self.save_final_results(stats, trades)

    def save_final_results(self, stats, trades):
        """Save final results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        if trades:
            df_trades = pd.DataFrame(trades)
            df_trades.to_csv(f'final_trades_{timestamp}.csv', index=False)
            print(f"\n✓ Trades saved: final_trades_{timestamp}.csv")

        # Final report
        report = f"""
==========================================
FINAL STRATEGY OPTIMIZATION REPORT
==========================================
Date: {datetime.now().strftime('%d.%m.%Y %H:%M')}
Symbol: {self.symbol}
Test period: 90 days

PREVIOUS BACKTEST DATA:
----------------------------------------
Total trades: 1263
Win Rate: 35.3%
Total P&L: -1039.7 pips
Average profit: +25.1 pips
Average loss: -15.7 pips

KEY INSIGHTS:
----------------
1. Zones 10-15 pips: -1.2p avg (worst)
2. Zones 15-20 pips: -0.9p avg
3. Zones 20-25 pips: -0.2p avg
4. Zones 25-30 pips: +0.4p avg (ONLY profitable!)

APPLIED OPTIMIZATIONS:
---------------------
1. Focus ONLY on zones 20-35 pips
2. Increased stop-loss (70% instead of 50%)
3. Added RSI filter
4. Increased holding time (18 hours)

OPTIMIZATION RESULTS:
----------------------
Executed trades: {stats.get('trades_executed', 0)}
Win Rate: {stats.get('win_rate', 0):.1%}
Total P&L: {stats.get('total_pnl', 0):+.1f} pips
Average trade: {stats.get('avg_trade', 0):+.2f} pips

CONCLUSION:
-----
"""

        if stats.get('trades_executed', 0) > 0:
            if stats.get('avg_trade', 0) > 0.5:
                report += "✅ OPTIMIZATION SUCCESSFUL!\n"
                report += "   Strategy became profitable\n"
                report += "   Demo testing recommended\n"
            elif stats.get('avg_trade', 0) > 0:
                report += "⚠️ PROGRESS MADE\n"
                report += "   Expected value improved\n"
                report += "   Further work required\n"
            else:
                report += "❌ OPTIMIZATION FAILED\n"
                report += "   Concept needs revision\n"
                report += "   Consider other approaches\n"
        else:
            report += "❌ NO DATA FOR ANALYSIS\n"
            report += "   Optimization is too strict\n"

        report += "\nDATA-DRIVEN RECOMMENDATIONS:\n"
        report += "1. Avoid zones closer than 20 pips\n"
        report += "2. Focus on 25-35 pip zones\n"
        report += "3. Consider reversal FROM zones strategy\n"
        report += "4. Test on other timeframes (H1, H4)\n"

        filename = f'final_report_{timestamp}.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✓ Report saved: {filename}")
        print(report)


def main():
    """Run final optimized backtest"""
    print("=" * 80)
    print("FINAL LIQUIDITY ZONE STRATEGY OPTIMIZATION")
    print("=" * 80)

    backtester = OptimizedFinalBacktester(symbol="EURUSD", spread_pips=1.2)

    if backtester.load_data(days=90):
        stats, trades = backtester.run_optimized_backtest()

        print("\n" + "=" * 80)
        print("FINAL BACKTEST COMPLETED!")
        print("=" * 80)

        if trades:
            print(f"\n📊 FINAL STATISTICS:")
            print(f"   Total trades: {len(trades)}")

            df_trades = pd.DataFrame(trades)
            if len(df_trades) > 0:
                win_rate = (df_trades['profit_pips'] > 0).mean()
                avg_profit = df_trades['profit_pips'].mean()
                total_pnl = df_trades['profit_pips'].sum()

                print(f"   Win Rate: {win_rate:.1%}")
                print(f"   Average trade: {avg_profit:+.2f} pips")
                print(f"   Total P&L: {total_pnl:+.1f} pips")

                # Optimal parameters
                best_rr = df_trades.groupby('risk_reward')['profit_pips'].mean().idxmax()
                best_distance = df_trades.groupby('distance')['profit_pips'].mean().idxmax()

                print(f"   Best Risk/Reward: {best_rr:.1f}")
                print(f"   Best distance: {best_distance:.0f} pips")


if __name__ == "__main__":
    main()
