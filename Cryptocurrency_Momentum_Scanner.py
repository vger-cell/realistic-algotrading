
"""
Optimized Cryptocurrency Momentum Scanner
Author: Vladimir Korneev
Telegram: t.me/realistic_algotrading
Repository: github.com/vger-cell/realistic-algotrading

Description:
Real-time scanner detecting trend-following (TREND_LONG) and mean-reversion (24H_SHORT)
signals in top cryptocurrencies. Features BTC correlation filtering, strict risk
management (TP/SL = 1.8%/0.9%), and CSV logging.

Strategy Logic:
1. TREND_LONG: Price change ≥0.3% + momentum ≥0.2% + BTC not falling >0.3%
2. 24H_SHORT: 24h drop ≥3.0% + negative momentum ≤-0.2% + BTC not rising >0.3%
3. All trades: Max 40 minutes, max 2 concurrent positions

Real-Time Output: Shows coin analysis, signal generation, and session statistics.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import csv
import os
from typing import Dict, List, Optional

# ================== CONFIGURATION ==================
CONFIG = {
    'check_interval': 60,
    'max_coins': 15,
    'min_volume_usd': 100_000_000,
    'min_price': 0.01,
    'exclude_stablecoins': True,
    'log_to_csv': True,
    'csv_filename': 'crypto_momentum_optimized.csv',

    # Signals: only effective ones
    'enable_trend_long': True,
    'enable_24h_short': True,

    # TP/SL and time parameters (2:1 ratio)
    'trend_long': {
        'profit_target': 1.8,
        'stop_loss': 0.9,
        'min_price_change': 0.3,
        'min_momentum': 0.2,
        'max_duration': 40
    },
    '24h_short': {
        'profit_target': 1.8,
        'stop_loss': 0.9,
        'min_24h_drop': 3.0,
        'min_momentum_down': -0.2,
        'max_duration': 40
    },

    # BTC correlation filter
    'use_btc_correlation_filter': True,
    'btc_max_rise_for_short': 0.3,  # % per 5 min — don't open SHORT above this
    'btc_max_fall_for_long': 0.3,  # % per 5 min — don't open LONG below this

    # Risk management
    'max_concurrent_signals': 2,
    'max_24h_change_filter': 10.0  # filter out pumps/dumps >10%
}

STABLECOINS = {'USDT', 'USDC', 'FDUSD', 'BUSD', 'DAI', 'TUSD', 'USD1', 'BSC-USD', 'USDD'}

DATA_SOURCES = [
    {
        'name': 'Binance',
        'url': 'https://api.binance.com/api/v3/ticker/24hr',
        'filter': lambda x: x['symbol'].endswith('USDT'),
        'transform': lambda x: {
            'id': x['symbol'].replace('USDT', '').lower(),
            'symbol': x['symbol'].replace('USDT', ''),
            'name': x['symbol'].replace('USDT', ''),
            'price': float(x['lastPrice']),
            'volume': float(x['volume']) * float(x['lastPrice']),
            'change_24h': float(x['priceChangePercent']),
            'high_24h': float(x['highPrice']),
            'low_24h': float(x['lowPrice'])
        }
    }
]


class OptimizedMomentumScanner:
    def __init__(self, config: Dict):
        self.config = config
        self.active_signals = {}
        self.closed_signals = []
        self.price_history = {}
        self.scan_count = 0
        self.total_signals = 0
        self.session_profit = 0.0
        self.session_start = datetime.now()
        if config['log_to_csv']:
            self.init_csv()
        self.print_header()

    def print_header(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n" + "=" * 80)
        print("⚡ OPTIMIZED MOMENTUM SCANNER")
        print("=" * 80)
        print("Only profitable signals | TP/SL = 1.8%/0.9% | BTC filter")
        print("=" * 80)
        print(f"\n📊 CONFIGURATION:")
        print(f"   • Signals: TREND_LONG, 24H_SHORT")
        print(f"   • TP/SL: 1.8% / 0.9% (2:1 ratio)")
        print(f"   • Max time: 40 min")
        print(f"   • Max active: {self.config['max_concurrent_signals']}")
        print(f"   • BTC filter: {'ON' if self.config['use_btc_correlation_filter'] else 'OFF'}")
        print("=" * 80)
        print("\n🔄 Starting scan... (Ctrl+C to stop)\n")

    def init_csv(self):
        if not os.path.exists(self.config['csv_filename']):
            with open(self.config['csv_filename'], 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'name', 'signal_type', 'entry_price',
                    'exit_price', 'pnl_percent', 'duration_min', 'volume',
                    'momentum', 'price_change', 'change_24h', 'status', 'close_reason'
                ])

    def log_to_csv(self, signal: Dict):
        with open(self.config['csv_filename'], 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                signal.get('timestamp', ''),
                signal.get('symbol', ''),
                signal.get('name', ''),
                signal.get('signal_type', ''),
                signal.get('entry_price', 0),
                signal.get('exit_price', 0),
                signal.get('pnl_percent', 0),
                signal.get('duration_min', 0),
                signal.get('volume', 0),
                signal.get('momentum', 0),
                signal.get('price_change', 0),
                signal.get('change_24h', 0),
                signal.get('status', ''),
                signal.get('close_reason', '')
            ])

    def print_colored(self, text: str, color: str = 'white', style: str = ''):
        colors = {
            'red': '\033[91m', 'green': '\033[92m', 'yellow': '\033[93m',
            'blue': '\033[94m', 'magenta': '\033[95m', 'cyan': '\033[96m',
            'white': '\033[97m', 'orange': '\033[38;5;208m',
        }
        styles = {'bold': '\033[1m', 'underline': '\033[4m'}
        color_code = colors.get(color, '')
        style_code = styles.get(style, '')
        reset = '\033[0m'
        return f"{style_code}{color_code}{text}{reset}"

    def get_market_data(self) -> tuple:
        all_coins = []
        btc_data = None
        for source in DATA_SOURCES:
            try:
                resp = requests.get(source['url'], timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
                if resp.status_code == 200:
                    data = resp.json()
                    items = [x for x in data if source['filter'](x)]
                    for item in items:
                        try:
                            coin = source['transform'](item)
                            if coin['symbol'] == 'BTC':
                                btc_data = coin
                            if (
                                    coin['volume'] >= self.config['min_volume_usd'] and
                                    coin['price'] >= self.config['min_price'] and
                                    coin['symbol'] not in STABLECOINS and
                                    abs(coin['change_24h']) <= self.config['max_24h_change_filter']
                            ):
                                all_coins.append(coin)
                        except Exception:
                            continue
            except Exception:
                continue
        unique = {}
        for coin in all_coins:
            s = coin['symbol']
            if s not in unique or coin['volume'] > unique[s]['volume']:
                unique[s] = coin
        sorted_coins = sorted(unique.values(), key=lambda x: x['volume'], reverse=True)
        return sorted_coins[:self.config['max_coins']], btc_data

    def update_price_history(self, coin: Dict):
        symbol = coin['symbol']
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append({
            'timestamp': datetime.now(),
            'price': coin['price'],
            'change_24h': coin['change_24h']
        })
        if len(self.price_history[symbol]) > 30:
            self.price_history[symbol] = self.price_history[symbol][-30:]

    def analyze_momentum(self, symbol: str) -> Dict:
        if symbol not in self.price_history or len(self.price_history[symbol]) < 5:
            return {'momentum': 0, 'price_change': 0}
        try:
            prices = [x['price'] for x in self.price_history[symbol]]
            momentum = ((prices[-1] - prices[-3]) / prices[-3]) * 100 if len(prices) >= 3 else 0
            price_change = ((prices[-1] - prices[-5]) / prices[-5]) * 100 if len(prices) >= 5 else 0
            return {'momentum': momentum, 'price_change': price_change}
        except:
            return {'momentum': 0, 'price_change': 0}

    def calculate_btc_5m_change(self) -> float:
        hist = self.price_history.get('BTC', [])
        if len(hist) < 2:
            return 0.0
        try:
            p1, p2 = hist[-2]['price'], hist[-1]['price']
            return ((p2 - p1) / p1) * 100
        except:
            return 0.0

    def should_allow_signal(self, signal_type: str, btc_5m_change: float) -> bool:
        if not self.config['use_btc_correlation_filter']:
            return True
        if signal_type == '24H_SHORT':
            return btc_5m_change <= self.config['btc_max_rise_for_short']
        elif signal_type == 'TREND_LONG':
            return btc_5m_change >= -self.config['btc_max_fall_for_long']
        return True

    def check_signal(self, coin: Dict, momentum_data: Dict, btc_5m_change: float) -> Optional[Dict]:
        symbol = coin['symbol']
        if symbol in self.active_signals or len(self.active_signals) >= self.config['max_concurrent_signals']:
            return None

        if self.config['enable_trend_long']:
            cfg = self.config['trend_long']
            if momentum_data['price_change'] >= cfg['min_price_change'] and momentum_data['momentum'] >= cfg[
                'min_momentum']:
                if self.should_allow_signal('TREND_LONG', btc_5m_change):
                    return self._create_signal(coin, momentum_data, 'TREND_LONG', cfg)

        if self.config['enable_24h_short']:
            cfg = self.config['24h_short']
            if coin['change_24h'] <= -cfg['min_24h_drop'] and momentum_data['momentum'] <= cfg['min_momentum_down']:
                if self.should_allow_signal('24H_SHORT', btc_5m_change):
                    return self._create_signal(coin, momentum_data, '24H_SHORT', cfg)

        return None

    def _create_signal(self, coin: Dict, momentum_data: Dict, signal_type: str, cfg: Dict) -> Dict:
        price = coin['price']
        if 'LONG' in signal_type:
            target = price * (1 + cfg['profit_target'] / 100)
            stop = price * (1 - cfg['stop_loss'] / 100)
        else:
            target = price * (1 - cfg['profit_target'] / 100)
            stop = price * (1 + cfg['stop_loss'] / 100)
        return {
            'symbol': coin['symbol'],
            'name': coin['name'],
            'signal_type': signal_type,
            'entry_price': price,
            'entry_time': datetime.now(),
            'momentum': momentum_data['momentum'],
            'price_change': momentum_data['price_change'],
            'change_24h': coin['change_24h'],
            'volume': coin['volume'],
            'status': 'ACTIVE',
            'target_price': target,
            'stop_price': stop,
            'profit_target': cfg['profit_target'],
            'stop_loss': cfg['stop_loss'],
            'max_duration': cfg['max_duration']
        }

    def update_active_signals(self, market_prices: Dict[str, float]):
        now = datetime.now()
        closed = []
        for symbol, signal in list(self.active_signals.items()):
            if symbol not in market_prices:
                continue
            curr_price = market_prices[symbol]
            entry = signal['entry_price']
            duration = (now - signal['entry_time']).total_seconds() / 60
            pnl = ((curr_price - entry) / entry) * 100 if 'LONG' in signal['signal_type'] else ((
                                                                                                            entry - curr_price) / entry) * 100

            close_signal = False
            reason = ""

            if 'LONG' in signal['signal_type']:
                if curr_price >= signal['target_price']:
                    close_signal = True
                    reason = f"TARGET {signal['profit_target']:.1f}%"
                elif curr_price <= signal['stop_price']:
                    close_signal = True
                    reason = f"STOP {signal['stop_loss']:.1f}%"
            else:
                if curr_price <= signal['target_price']:
                    close_signal = True
                    reason = f"TARGET {signal['profit_target']:.1f}%"
                elif curr_price >= signal['stop_price']:
                    close_signal = True
                    reason = f"STOP {signal['stop_loss']:.1f}%"

            if duration > signal['max_duration']:
                close_signal = True
                reason = f"TIMEOUT {signal['max_duration']} min"

            if close_signal:
                signal['exit_price'] = curr_price
                signal['pnl_percent'] = pnl
                signal['duration_min'] = round(duration, 1)
                signal['status'] = 'CLOSED'
                signal['close_time'] = now
                signal['close_reason'] = reason
                signal['timestamp'] = now.strftime('%Y-%m-%d %H:%M:%S')
                self.closed_signals.append(signal.copy())
                closed.append(signal)
                self.session_profit += pnl
                if self.config['log_to_csv']:
                    self.log_to_csv(signal)
                del self.active_signals[symbol]
            else:
                signal['current_pnl'] = pnl
                signal['current_duration'] = duration
        return closed

    def print_scan_header(self):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n{self.print_colored('=' * 80, 'blue')}")
        print(f"{self.print_colored(f'⚡ SCAN #{self.scan_count} | {ts}', 'cyan', 'bold')}")
        print(f"{self.print_colored('=' * 80, 'blue')}")

    def print_coin_analysis(self, coin: Dict, momentum_data: Dict, has_signal: bool, signal_type: str = ''):
        symbol = coin['symbol']
        price = coin['price']
        ch24 = coin['change_24h']
        vol_m = coin['volume'] / 1_000_000
        ch_color = 'green' if ch24 > 0 else 'red'
        ch_icon = '▲' if ch24 > 0 else '▼'
        line = f"{symbol:6} ${price:10.3f} {self.print_colored(f'{ch_icon}{abs(ch24):5.1f}%', ch_color)} "
        if abs(momentum_data['momentum']) > 0.1:
            mom_col = 'green' if momentum_data['momentum'] > 0 else 'red'
            mom_icon = '↗' if momentum_data['momentum'] > 0.3 else '↘' if momentum_data['momentum'] < -0.3 else '➡'
            mom_str = f"{mom_icon}{momentum_data['momentum']:5.2f}%"
            line += f"Mom: {self.print_colored(mom_str, mom_col)} "
        line += f"Vol: ${vol_m:6.1f}M"
        if has_signal:
            sig_col = 'green' if 'LONG' in signal_type else 'red'
            line += f" {self.print_colored(f'🚨 {signal_type}', sig_col, 'bold')}"
        print(line)

    def print_new_signal(self, signal: Dict):
        col = 'green' if 'LONG' in signal['signal_type'] else 'red'
        print(f"\n{self.print_colored('═' * 80, col)}")
        inner_text = f'🚨 NEW SIGNAL: {signal["signal_type"]}'
        print(f"{self.print_colored(inner_text, col, 'bold')}")
        print(f"{self.print_colored('═' * 80, col)}")
        print(f"{self.print_colored('Coin:', 'white')} {signal['name']} ({signal['symbol']})")
        print(f"{self.print_colored('Entry price:', 'white')} ${signal['entry_price']:.4f}")
        print(f"{self.print_colored('Target:', 'white')} ${signal['target_price']:.4f} ({signal['profit_target']:.1f}%)")
        print(f"{self.print_colored('Stop:', 'white')} ${signal['stop_price']:.4f} ({signal['stop_loss']:.1f}%)")
        print(f"{self.print_colored('Momentum:', 'white')} {signal['momentum']:.2f}%")
        print(f"{self.print_colored('Change:', 'white')} {signal['price_change']:.2f}%")
        print(f"{self.print_colored('24h change:', 'white')} {signal['change_24h']:.2f}%")
        print(f"{self.print_colored('Volume:', 'white')} ${signal['volume'] / 1_000_000:.1f}M")
        print(f"{self.print_colored('Time:', 'white')} {signal['entry_time'].strftime('%H:%M:%S')}")
        print(f"{self.print_colored('Max duration:', 'white')} {signal['max_duration']} min")

    def print_closed_signal(self, signal: Dict):
        pnl = signal['pnl_percent']
        is_profit = pnl > 0
        col = 'green' if is_profit else 'red'
        res = "PROFIT" if is_profit else "LOSS"
        emo = '💰' if is_profit else '📉'
        print(f"\n{self.print_colored('═' * 80, col)}")
        print(f"{self.print_colored(f'{emo} SIGNAL CLOSED: {res}', col, 'bold')}")
        print(f"{self.print_colored('═' * 80, col)}")
        print(f"{self.print_colored('Coin:', 'white')} {signal['symbol']} ({signal['name']})")
        print(f"{self.print_colored('Type:', 'white')} {signal['signal_type']}")
        print(f"{self.print_colored('Entry:', 'white')} ${signal['entry_price']:.4f}")
        print(f"{self.print_colored('Exit:', 'white')} ${signal['exit_price']:.4f}")
        print(f"{self.print_colored('P/L:', 'white')} {self.print_colored(f'{pnl:+.2f}%', col)}")
        print(f"{self.print_colored('Duration:', 'white')} {signal['duration_min']:.1f} min")
        print(f"{self.print_colored('Reason:', 'white')} {signal['close_reason']}")

    def print_session_stats(self):
        dur = datetime.now() - self.session_start
        hrs = dur.total_seconds() / 3600
        total = len(self.closed_signals)
        prof = len([s for s in self.closed_signals if s['pnl_percent'] > 0])
        win_rate = (prof / total * 100) if total > 0 else 0
        avg_pnl = (self.session_profit / total) if total > 0 else 0
        avg_dur = np.mean([s['duration_min'] for s in self.closed_signals]) if total > 0 else 0

        print(f"\n{self.print_colored('=' * 80, 'cyan')}")
        print(f"{self.print_colored('📊 SESSION STATISTICS', 'cyan', 'bold')}")
        print(f"{self.print_colored('=' * 80, 'cyan')}")
        print(f"{self.print_colored('Duration:', 'white')} {hrs:.1f} h")
        print(f"{self.print_colored('Scans:', 'white')} {self.scan_count}")
        print(f"{self.print_colored('Total signals:', 'white')} {self.total_signals}")
        print(f"{self.print_colored('Closed trades:', 'white')} {total}")
        print(f"{self.print_colored('Profitable:', 'white')} {prof} of {total}")
        print(f"{self.print_colored('Win rate:', 'white')} {win_rate:.1f}%")
        pnl_col = 'green' if avg_pnl > 0 else 'red'
        print(f"{self.print_colored('Avg P/L:', 'white')} {self.print_colored(f'{avg_pnl:+.2f}%', pnl_col)}")
        print(f"{self.print_colored('Avg duration:', 'white')} {avg_dur:.1f} min")

        if total > 0:
            types = {}
            for s in self.closed_signals:
                t = s['signal_type']
                types[t] = types.get(t, 0) + 1
            print(f"\n{self.print_colored('SIGNAL TYPES:', 'yellow')}")
            for t, cnt in types.items():
                p_cnt = len([s for s in self.closed_signals if s['signal_type'] == t and s['pnl_percent'] > 0])
                perc = (p_cnt / cnt * 100) if cnt > 0 else 0
                print(f"  {t:15} - {cnt:3} signals, {p_cnt:3} profitable ({perc:.1f}%)")

    def scan(self):
        self.scan_count += 1
        self.print_scan_header()

        coins, btc_data = self.get_market_data()
        if not coins:
            print(f"{self.print_colored('⚠️ No data for analysis', 'yellow')}")
            return

        print(f"{self.print_colored(f'✅ Received {len(coins)} coins', 'green')}")
        print(f"{self.print_colored('─' * 80, 'blue')}")

        market_prices = {}
        new_signals = []

        if btc_data:
            self.update_price_history(btc_data)
        btc_5m_change = self.calculate_btc_5m_change()

        for coin in coins:
            sym = coin['symbol']
            market_prices[sym] = coin['price']
            self.update_price_history(coin)
            mom = self.analyze_momentum(sym)
            sig = self.check_signal(coin, mom, btc_5m_change)
            has_sig = sig is not None
            sig_type = sig['signal_type'] if sig else ''
            self.print_coin_analysis(coin, mom, has_sig, sig_type)
            if sig:
                self.active_signals[sym] = sig
                new_signals.append(sig)
                self.total_signals += 1

        closed_signals = self.update_active_signals(market_prices)

        for sig in new_signals:
            self.print_new_signal(sig)
        for sig in closed_signals:
            self.print_closed_signal(sig)

        print(f"\n{self.print_colored('📊 SCAN SUMMARY:', 'cyan')}")
        print(f"  New signals: {len(new_signals)}")
        print(f"  Closed trades: {len(closed_signals)}")
        print(f"  Active signals: {len(self.active_signals)}")
        print(f"  Total session signals: {self.total_signals}")

        if self.scan_count % 10 == 0:
            self.print_session_stats()

    def countdown(self, seconds: int):
        print(f"\n{self.print_colored('⏳ Next scan in:', 'yellow')}")
        for i in range(seconds, 0, -1):
            mins, secs = divmod(i, 60)
            time_str = f"{mins:02d}:{secs:02d}"
            progress = int((seconds - i) / seconds * 50)
            bar = "█" * progress + "░" * (50 - progress)
            print(f"\r[{bar}] {time_str} ", end='', flush=True)
            time.sleep(1)
        print()

    def run(self):
        try:
            while True:
                self.scan()
                if self.scan_count > 0:
                    self.countdown(self.config['check_interval'])
        except KeyboardInterrupt:
            print(f"\n\n{self.print_colored('=' * 80, 'yellow')}")
            print(f"{self.print_colored('🛑 SCANNER STOPPED', 'yellow', 'bold')}")
            print(f"{self.print_colored('=' * 80, 'yellow')}")
            self.print_session_stats()
            if self.config['log_to_csv']:
                print(f"\n{self.print_colored('💾 All trades saved to:', 'green')} {self.config['csv_filename']}")
            print(f"{self.print_colored('=' * 80, 'yellow')}\n")


if __name__ == "__main__":
    scanner = OptimizedMomentumScanner(CONFIG)
    scanner.run()
