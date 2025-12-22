"""
Event-Driven Multi-Asset Lead-Lag Analyzer
Author: Vladimir Korneev
Telegram: t.me/realistic_algotrading
Repo: github.com/vger-cell/realistic-algotrading

Description:
Detects statistically significant lead-lag relationships between FX pairs and gold
using breakout/bounce signals with RSI filtering. Includes full backtest engine,
visualization, and significance testing.

Key Features:
- Multi-timeframe analysis (M5, M15)
- Four signal types (breakout/bounce long/short)
- Fixed TP/SL simulation with pip calculations
- Binomial statistical validation (p < 0.05)
- Heatmap visualization and detailed reporting

Disclaimer:
For educational purposes only. Past performance does not guarantee future results.
Always test strategies with proper risk management.

Version: 1.0.0
Last Updated: 2024-03-20
"""

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import warnings
import traceback
import os
import matplotlib.pyplot as plt
from scipy.stats import binomtest
warnings.filterwarnings('ignore')

# ------------------------------- CONFIG -------------------------------
class Config:
    SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "XAUUSD"]
    TIMEFRAMES = ['M5', 'M15']
    TIMEFRAME_DICT = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1
    }
    DAYS_BACK = 180
    LOOKBACK_PERIOD = 20
    MIN_RSI_LONG = 30
    MAX_RSI_SHORT = 70
    MAX_DELAY_BARS = 10
    MIN_SIGNALS_PER_PAIR = 10
    MIN_HIT_RATE = 0.55  # Повышен для реалистичности
    MIN_PROFIT_PIPS = 10  # Минимальная прибыль в пипсах для успеха
    MAX_LOSS_PIPS = 20    # Максимальный убыток для стоп-лосса
    PIP_MULTIPLIER = {
        'EURUSD': 10000, 'GBPUSD': 10000, 'USDJPY': 100,
        'AUDUSD': 10000, 'USDCAD': 10000, 'USDCHF': 10000,
        'NZDUSD': 10000, 'XAUUSD': 10  # Для золота 1 пип = $0.10
    }
    STATISTICAL_SIGNIFICANCE = 0.05  # 95% доверительный интервал

CFG = Config()

# ------------------------------- 1. DATA LOADING -------------------------------
print("1. Loading data from MetaTrader5...")

def get_data_from_mt5(symbol: str, timeframe_str: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    try:
        if not mt5.initialize():
            print("MT5 init failed.")
            return None
        timeframe = CFG.TIMEFRAME_DICT.get(timeframe_str)
        if timeframe is None:
            print(f"Invalid timeframe: {timeframe_str}")
            return None
        utc_from = int(start_date.timestamp())
        utc_to = int(end_date.timestamp())
        rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        df.columns = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
        print(f"Loaded {len(df)} bars for {symbol} on {timeframe_str}")
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        try:
            mt5.shutdown()
        except:
            pass

# ------------------------------- 2. INDICATORS -------------------------------
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['rolling_high'] = df['high'].rolling(CFG.LOOKBACK_PERIOD).max()
    df['rolling_low'] = df['low'].rolling(CFG.LOOKBACK_PERIOD).min()
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()  # ATR для определения волатильности
    return df

# ------------------------------- 3. SIGNALS -------------------------------
def detect_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    signals = pd.DataFrame(0, index=df.index, columns=['signal', 'price', 'type'])

    for i in range(CFG.LOOKBACK_PERIOD, len(df)):
        row = df.iloc[i]
        prev_close = df.iloc[i-1]['close']
        prev_rolling_high = df['rolling_high'].iloc[i-1]
        prev_rolling_low = df['rolling_low'].iloc[i-1]

        signal_type = 0
        # Breakout LONG
        if (row['close'] > row['rolling_high']) and (row['rsi'] <= CFG.MAX_RSI_SHORT):
            signals.iloc[i, 0] = 1
            signals.iloc[i, 1] = row['close']
            signal_type = 1
        # Breakout SHORT
        elif (row['close'] < row['rolling_low']) and (row['rsi'] >= CFG.MIN_RSI_LONG):
            signals.iloc[i, 0] = -1
            signals.iloc[i, 1] = row['close']
            signal_type = 2
        # Bounce SHORT
        elif (prev_close >= prev_rolling_high) and (row['close'] < row['rolling_high']) and (row['rsi'] >= 60):
            signals.iloc[i, 0] = -1
            signals.iloc[i, 1] = row['close']
            signal_type = 3
        # Bounce LONG
        elif (prev_close <= prev_rolling_low) and (row['close'] > row['rolling_low']) and (row['rsi'] <= 40):
            signals.iloc[i, 0] = 1
            signals.iloc[i, 1] = row['close']
            signal_type = 4

        signals.iloc[i, 2] = signal_type

    return signals

# ------------------------------- 4. LEAD-LAG ANALYSIS -------------------------------
def calculate_pips_change(price_change, symbol):
    """Рассчитывает изменение в пипсах"""
    multiplier = CFG.PIP_MULTIPLIER.get(symbol, 10000)
    return abs(price_change) * multiplier

def analyze_lead_lag(signal_events: dict, price_data: dict, timeframe_str: str) -> list:
    results = []
    assets = list(signal_events.keys())
    bar_duration = 5 if timeframe_str == 'M5' else 15

    for leader in assets:
        leader_signals_df = signal_events[leader]
        leader_signals = leader_signals_df[leader_signals_df['signal'] != 0]

        if len(leader_signals) < CFG.MIN_SIGNALS_PER_PAIR:
            continue

        for follower in assets:
            if follower == leader:
                continue

            follower_close = price_data[follower]['close']
            valid_cases = 0
            matching = 0
            total_delay = 0
            total_profit_pips = 0
            total_loss_pips = 0
            winning_cases = 0
            losing_cases = 0

            # Статистика по типам сигналов
            signal_types_count = {1: 0, 2: 0, 3: 0, 4: 0}
            signal_types_hits = {1: 0, 2: 0, 3: 0, 4: 0}

            for idx, signal_row in leader_signals.iterrows():
                t = idx
                t_next = t + pd.Timedelta(minutes=bar_duration)

                if t_next not in follower_close.index:
                    continue

                entry_price = follower_close.loc[t_next]
                signal_dir = signal_row['signal']
                signal_type = signal_row['type']

                signal_types_count[signal_type] += 1

                trade_success = False
                trade_delay = 0
                trade_pnl_pips = 0

                # Проверяем все возможные выходы (до MAX_DELAY_BARS)
                for delay in range(CFG.MAX_DELAY_BARS):
                    t_future = t_next + pd.Timedelta(minutes=bar_duration * delay)

                    if t_future not in follower_close.index:
                        # Если вышли за пределы данных, считаем последнюю цену
                        if delay > 0:
                            t_future = t_next + pd.Timedelta(minutes=bar_duration * (delay - 1))
                            exit_price = follower_close.loc[t_future]
                            trade_delay = delay
                        break

                    exit_price = follower_close.loc[t_future]
                    price_change = exit_price - entry_price
                    pips_change = calculate_pips_change(price_change, follower)

                    # LONG сигнал
                    if signal_dir == 1:
                        # Проверяем достигли ли целевой прибыли
                        if price_change > 0 and pips_change >= CFG.MIN_PROFIT_PIPS:
                            trade_success = True
                            trade_delay = delay + 1
                            trade_pnl_pips = pips_change
                            break
                        # Проверяем достигли ли стоп-лосса
                        elif price_change < 0 and pips_change >= CFG.MAX_LOSS_PIPS:
                            trade_success = False
                            trade_delay = delay + 1
                            trade_pnl_pips = -pips_change
                            break
                        # Последний бар - фиксируем результат
                        elif delay == CFG.MAX_DELAY_BARS - 1:
                            trade_success = price_change > 0
                            trade_delay = delay + 1
                            trade_pnl_pips = pips_change if price_change > 0 else -pips_change

                    # SHORT сигнал
                    elif signal_dir == -1:
                        if price_change < 0 and pips_change >= CFG.MIN_PROFIT_PIPS:
                            trade_success = True
                            trade_delay = delay + 1
                            trade_pnl_pips = pips_change
                            break
                        elif price_change > 0 and pips_change >= CFG.MAX_LOSS_PIPS:
                            trade_success = False
                            trade_delay = delay + 1
                            trade_pnl_pips = -pips_change
                            break
                        elif delay == CFG.MAX_DELAY_BARS - 1:
                            trade_success = price_change < 0
                            trade_delay = delay + 1
                            trade_pnl_pips = pips_change if price_change < 0 else -pips_change

                valid_cases += 1

                if trade_success:
                    matching += 1
                    signal_types_hits[signal_type] += 1
                    total_delay += trade_delay
                    total_profit_pips += trade_pnl_pips
                    winning_cases += 1
                else:
                    total_loss_pips += abs(trade_pnl_pips)
                    losing_cases += 1

            if valid_cases >= CFG.MIN_SIGNALS_PER_PAIR:
                hit_rate = matching / valid_cases if valid_cases > 0 else 0

                # Статистический тест (биномиальный)
                if valid_cases > 0 and matching > 0:
                    binom_result = binomtest(matching, valid_cases, p=0.5, alternative='greater')
                    p_value = binom_result.pvalue
                    significant = p_value < CFG.STATISTICAL_SIGNIFICANCE
                else:
                    p_value = 1.0
                    significant = False

                if hit_rate >= CFG.MIN_HIT_RATE and significant:
                    avg_delay = total_delay / matching if matching > 0 else 0
                    avg_win_pips = total_profit_pips / winning_cases if winning_cases > 0 else 0
                    avg_loss_pips = total_loss_pips / losing_cases if losing_cases > 0 else 0
                    profit_factor = total_profit_pips / total_loss_pips if total_loss_pips > 0 else 999

                    # Эффективность по типам сигналов
                    signal_efficiency = {}
                    for st in [1, 2, 3, 4]:
                        if signal_types_count[st] > 0:
                            eff = signal_types_hits[st] / signal_types_count[st]
                            signal_efficiency[f'S{st}_HitRate'] = round(eff, 3)

                    results.append({
                        'Timeframe': timeframe_str,
                        'Leader': leader,
                        'Follower': follower,
                        'Leader_Signals': len(leader_signals),
                        'Valid_Cases': valid_cases,
                        'Matching_Directions': matching,
                        'Hit_Rate': round(hit_rate, 3),
                        'P_Value': round(p_value, 4),
                        'Significant': significant,
                        'Avg_Delay_Bars': round(avg_delay, 2),
                        'Avg_Win_Pips': round(avg_win_pips, 1),
                        'Avg_Loss_Pips': round(avg_loss_pips, 1),
                        'Profit_Factor': round(profit_factor, 2),
                        'Win_Rate': round(winning_cases/valid_cases, 3) if valid_cases > 0 else 0,
                        **signal_efficiency
                    })

    return results

# ------------------------------- 5. VISUALIZATION -------------------------------
def visualize_results(df_results: pd.DataFrame):
    """Визуализация результатов"""
    if df_results.empty:
        print("No results to visualize.")
        return

    os.makedirs('event_lead_lag/plots', exist_ok=True)

    # 1. Hit Rate Distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    hit_rates = df_results['Hit_Rate']
    plt.hist(hit_rates, bins=20, alpha=0.7, color='blue')
    plt.axvline(x=hit_rates.mean(), color='red', linestyle='--', label=f'Mean: {hit_rates.mean():.3f}')
    plt.xlabel('Hit Rate')
    plt.ylabel('Frequency')
    plt.title('Distribution of Hit Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Average Delay Distribution
    plt.subplot(1, 2, 2)
    delays = df_results['Avg_Delay_Bars']
    plt.hist(delays, bins=20, alpha=0.7, color='green')
    plt.axvline(x=delays.mean(), color='red', linestyle='--', label=f'Mean: {delays.mean():.2f}')
    plt.xlabel('Average Delay (Bars)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Average Delays')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('event_lead_lag/plots/distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Heatmap of Hit Rates
    for timeframe in df_results['Timeframe'].unique():
        tf_data = df_results[df_results['Timeframe'] == timeframe]
        if len(tf_data) > 0:
            try:
                # Create pivot table
                pivot_table = tf_data.pivot(index='Leader', columns='Follower', values='Hit_Rate')

                plt.figure(figsize=(10, 8))
                plt.imshow(pivot_table, cmap='RdYlGn', vmin=0.5, vmax=0.8)
                plt.colorbar(label='Hit Rate')
                plt.xticks(range(len(pivot_table.columns)), pivot_table.columns, rotation=45)
                plt.yticks(range(len(pivot_table.index)), pivot_table.index)
                plt.xlabel('Follower')
                plt.ylabel('Leader')
                plt.title(f'Lead-Lag Hit Rate Heatmap ({timeframe})')

                # Add text annotations
                for i in range(len(pivot_table.index)):
                    for j in range(len(pivot_table.columns)):
                        if not pd.isna(pivot_table.iloc[i, j]):
                            plt.text(j, i, f'{pivot_table.iloc[i, j]:.3f}',
                                    ha='center', va='center', color='black', fontsize=8)

                plt.tight_layout()
                plt.savefig(f'event_lead_lag/plots/heatmap_{timeframe}.png', dpi=150, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Could not create heatmap for {timeframe}: {e}")

    print("Visualizations saved to 'event_lead_lag/plots/'")

# ------------------------------- 6. MAIN PIPELINE -------------------------------
def run_for_timeframe(timeframe_str: str):
    print(f"\n{'='*60}")
    print(f"ANALYZING: {timeframe_str}")
    print(f"{'='*60}")

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=CFG.DAYS_BACK)

    all_data = {}
    for sym in CFG.SYMBOLS:
        df = get_data_from_mt5(sym, timeframe_str, start_date, end_date)
        if df is not None:
            all_data[sym] = df

    if not all_data:
        print("No data. Skipping.")
        return []

    common_index = None
    for df in all_data.values():
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)

    print(f"Aligned {len(common_index)} bars across {len(all_data)} assets.")

    signal_events = {}
    for sym, df in all_data.items():
        df_aligned = df.reindex(common_index)
        df_ind = calculate_indicators(df_aligned)
        signals = detect_signals(df_ind)
        signal_events[sym] = signals
        n_signals = (signals['signal'] != 0).sum()
        print(f"  {sym}: {n_signals} signals")

        # Статистика по типам сигналов
        if n_signals > 0:
            signal_types = signals[signals['signal'] != 0]['type'].value_counts()
            type_names = {1: 'Breakout LONG', 2: 'Breakout SHORT',
                         3: 'Bounce SHORT', 4: 'Bounce LONG'}
            for stype, count in signal_types.items():
                if stype in type_names:
                    print(f"    {type_names[stype]}: {count}")

    lead_lag_results = analyze_lead_lag(
        signal_events,
        {sym: df.reindex(common_index) for sym, df in all_data.items()},
        timeframe_str
    )

    print(f"Found {len(lead_lag_results)} significant pairs on {timeframe_str}.")
    return lead_lag_results

# ------------------------------- 7. REPORT GENERATION -------------------------------
def generate_report(df_results: pd.DataFrame):
    """Генерация детального отчета"""
    if df_results.empty:
        return

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("LEAD-LAG ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total significant pairs: {len(df_results)}")

    for timeframe in sorted(df_results['Timeframe'].unique()):
        tf_data = df_results[df_results['Timeframe'] == timeframe]
        report_lines.append(f"\n{'='*40}")
        report_lines.append(f"TIMEFRAME: {timeframe}")
        report_lines.append(f"{'='*40}")
        report_lines.append(f"Pairs found: {len(tf_data)}")

        if not tf_data.empty:
            # Top 5 leaders by number of significant followers
            leader_stats = tf_data.groupby('Leader').size().sort_values(ascending=False)
            report_lines.append(f"\nTop Leaders (by number of significant followers):")
            for leader, count in leader_stats.head().items():
                report_lines.append(f"  {leader}: {count} followers")

            # Top 5 pairs by hit rate
            top_pairs = tf_data.nlargest(5, 'Hit_Rate')[['Leader', 'Follower', 'Hit_Rate', 'Avg_Delay_Bars', 'Profit_Factor']]
            report_lines.append(f"\nTop 5 Pairs by Hit Rate:")
            for _, row in top_pairs.iterrows():
                report_lines.append(f"  {row['Leader']} -> {row['Follower']}: "
                                  f"Hit={row['Hit_Rate']:.1%}, "
                                  f"Delay={row['Avg_Delay_Bars']:.1f} bars, "
                                  f"PF={row['Profit_Factor']:.2f}")

    # Save report with UTF-8 encoding
    report_path = 'event_lead_lag/analysis_report.txt'
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"\n📊 Report saved to: {report_path}")
    except Exception as e:
        print(f"⚠️ Could not save report: {e}")
        # Try alternative without special characters
        report_path_alt = 'event_lead_lag/analysis_report_simple.txt'
        simple_lines = []
        for line in report_lines:
            # Replace any problematic characters
            safe_line = line.replace('→', '->').replace('📊', '').replace('⚠️', '')
            simple_lines.append(safe_line)
        with open(report_path_alt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(simple_lines))
        print(f"Simple report saved to: {report_path_alt}")

    # Also print to console
    print('\n'.join(report_lines))

# ------------------------------- 8. RUN -------------------------------
def main():
    print("=" * 80)
    print("MULTI-ASSET EVENT-BASED LEAD-LAG ANALYSIS")
    print("=" * 80)

    all_results = []
    for tf in CFG.TIMEFRAMES:
        res = run_for_timeframe(tf)
        all_results.extend(res)

    if not all_results:
        print("\n⚠️  No significant results found. Check parameters.")
        return

    os.makedirs('event_lead_lag', exist_ok=True)

    # Create DataFrame and save
    df_results = pd.DataFrame(all_results)

    # Sort by Timeframe and Hit Rate
    df_results = df_results.sort_values(['Timeframe', 'Hit_Rate'], ascending=[True, False])

    output_file = 'event_lead_lag/lead_lag_results_detailed.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to {output_file}")

    # Display summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for tf in CFG.TIMEFRAMES:
        tf_data = df_results[df_results['Timeframe'] == tf]
        if not tf_data.empty:
            avg_hit = tf_data['Hit_Rate'].mean()
            avg_delay = tf_data['Avg_Delay_Bars'].mean()
            avg_pf = tf_data['Profit_Factor'].mean()
            significant_pairs = len(tf_data)

            print(f"\n{tf.upper()}:")
            print(f"  Significant pairs: {significant_pairs}")
            print(f"  Average Hit Rate: {avg_hit:.1%}")
            print(f"  Average Delay: {avg_delay:.1f} bars")
            print(f"  Average Profit Factor: {avg_pf:.2f}")
            print(f"  Average Win Rate: {tf_data['Win_Rate'].mean():.1%}")

    # Generate visualizations
    try:
        visualize_results(df_results)
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")

    # Generate detailed report
    try:
        generate_report(df_results)
    except Exception as e:
        print(f"⚠️ Report generation error: {e}")

    # Show top 10 pairs overall
    print("\n" + "=" * 80)
    print("TOP 10 LEAD-LAG PAIRS (Overall)")
    print("=" * 80)
    if not df_results.empty:
        top_10 = df_results.nlargest(10, 'Hit_Rate')[['Timeframe', 'Leader', 'Follower', 'Hit_Rate', 'Avg_Delay_Bars', 'Profit_Factor', 'P_Value']]
        print(top_10.to_string(index=False))
    else:
        print("No results to display.")

if __name__ == "__main__":
    main()