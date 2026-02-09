"""
Trade logging, journaling, and live performance metrics for live trading.
Logs every trade, portfolio snapshot, and model decision to CSV files.
Computes rolling metrics: win rate, Sharpe, Sortino, max drawdown.
"""

import os
import csv
import math
import logging
import numpy as np
from datetime import datetime, date

logger = logging.getLogger(__name__)


class LiveMetrics:
    """Rolling live performance metrics computed from trade and portfolio history."""

    def __init__(self):
        self.portfolio_values = []
        self.portfolio_timestamps = []
        self.trade_pnls = []           # per-trade P&L (realized)
        self.trade_symbols = []
        self.peak_value = 0.0
        self.max_drawdown = 0.0
        self.daily_start_value = 0.0
        self.daily_start_date = None

        # Per-ticker tracking
        self._ticker_wins = {}
        self._ticker_losses = {}

    def record_portfolio(self, value):
        """Record a portfolio value snapshot."""
        self.portfolio_values.append(value)
        self.portfolio_timestamps.append(datetime.now())
        if value > self.peak_value:
            self.peak_value = value
        dd = (self.peak_value - value) / self.peak_value if self.peak_value > 0 else 0
        if dd > self.max_drawdown:
            self.max_drawdown = dd

        # Daily tracking
        today = date.today()
        if self.daily_start_date != today:
            self.daily_start_date = today
            self.daily_start_value = value

    def record_trade(self, symbol, side, amount, price, entry_price=None):
        """Record a trade for P&L tracking."""
        if side == 'sell' and entry_price and entry_price > 0:
            pnl = (price - entry_price) * amount
            self.trade_pnls.append(pnl)
            self.trade_symbols.append(symbol)
            if pnl >= 0:
                self._ticker_wins[symbol] = self._ticker_wins.get(symbol, 0) + 1
            else:
                self._ticker_losses[symbol] = self._ticker_losses.get(symbol, 0) + 1

    @property
    def total_trades(self):
        return len(self.trade_pnls)

    @property
    def win_rate(self):
        if not self.trade_pnls:
            return 0.0
        wins = sum(1 for p in self.trade_pnls if p >= 0)
        return wins / len(self.trade_pnls)

    @property
    def win_rate_per_ticker(self):
        """Returns dict {symbol: win_rate}."""
        result = {}
        all_tickers = set(list(self._ticker_wins.keys()) + list(self._ticker_losses.keys()))
        for t in all_tickers:
            w = self._ticker_wins.get(t, 0)
            l = self._ticker_losses.get(t, 0)
            total = w + l
            result[t] = w / total if total > 0 else 0.0
        return result

    @property
    def total_pnl(self):
        return sum(self.trade_pnls) if self.trade_pnls else 0.0

    @property
    def daily_pnl(self):
        if not self.portfolio_values or self.daily_start_value <= 0:
            return 0.0
        return self.portfolio_values[-1] - self.daily_start_value

    @property
    def daily_pnl_pct(self):
        if self.daily_start_value <= 0:
            return 0.0
        return self.daily_pnl / self.daily_start_value

    def sharpe_ratio(self, annualize_factor=105120):
        """Compute Sharpe ratio from portfolio returns.
        annualize_factor: data points per year (default: 5m candles = 105120)
        """
        if len(self.portfolio_values) < 3:
            return 0.0
        values = np.array(self.portfolio_values)
        returns = np.diff(values) / values[:-1]
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        sr = np.mean(returns) / np.std(returns)
        return sr * math.sqrt(min(annualize_factor, len(returns)))

    def sortino_ratio(self, annualize_factor=105120):
        """Compute Sortino ratio (downside deviation only)."""
        if len(self.portfolio_values) < 3:
            return 0.0
        values = np.array(self.portfolio_values)
        returns = np.diff(values) / values[:-1]
        downside = returns[returns < 0]
        if len(downside) < 1 or np.std(downside) == 0:
            return 0.0 if np.mean(returns) <= 0 else float('inf')
        sr = np.mean(returns) / np.std(downside)
        return sr * math.sqrt(min(annualize_factor, len(returns)))

    def get_summary_dict(self):
        """Return all metrics as a dict."""
        return {
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'win_rate_per_ticker': self.win_rate_per_ticker,
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl_pct,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'peak_value': self.peak_value,
            'current_value': self.portfolio_values[-1] if self.portfolio_values else 0,
        }


class TradeLogger:
    def __init__(self, log_dir='./live_trading_logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.trades_file = os.path.join(log_dir, f'trades_{timestamp}.csv')
        self.portfolio_file = os.path.join(log_dir, f'portfolio_{timestamp}.csv')
        self.decisions_file = os.path.join(log_dir, f'decisions_{timestamp}.csv')
        self.metrics_file = os.path.join(log_dir, f'metrics_{timestamp}.csv')

        self._init_trades_csv()
        self._init_portfolio_csv()
        self._init_decisions_csv()
        self._init_metrics_csv()

        # Live metrics tracker
        self.metrics = LiveMetrics()

        logger.info(f"Trade logger initialized: {log_dir}")

    def _init_trades_csv(self):
        with open(self.trades_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'symbol', 'side', 'amount', 'price',
                'cost', 'fee', 'slippage', 'order_id', 'paper'
            ])

    def _init_portfolio_csv(self):
        with open(self.portfolio_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'cash_usdt', 'portfolio_value',
                'drawdown_pct', 'positions_json'
            ])

    def _init_decisions_csv(self):
        with open(self.decisions_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'raw_actions', 'filtered_actions',
                'current_prices', 'portfolio_value'
            ])

    def _init_metrics_csv(self):
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'total_trades', 'win_rate', 'total_pnl',
                'daily_pnl', 'daily_pnl_pct', 'max_drawdown',
                'sharpe_ratio', 'sortino_ratio', 'portfolio_value'
            ])

    def log_trade(self, order, entry_price=None):
        """Log a single executed trade."""
        if order is None:
            return
        try:
            with open(self.trades_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    order.get('timestamp', datetime.now().isoformat()),
                    order.get('symbol', ''),
                    order.get('side', ''),
                    order.get('amount', 0),
                    order.get('price', 0),
                    order.get('cost', 0),
                    order.get('fee', 0),
                    order.get('slippage', 0),
                    order.get('id', ''),
                    order.get('paper', False),
                ])

            # Track metrics
            self.metrics.record_trade(
                symbol=order.get('symbol', ''),
                side=order.get('side', ''),
                amount=float(order.get('amount', 0)),
                price=float(order.get('price', 0)),
                entry_price=entry_price,
            )
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")

    def log_trades(self, orders, entry_prices=None):
        """Log multiple executed trades.
        Args:
            orders: list of order dicts
            entry_prices: optional dict {symbol: entry_price} for P&L tracking
        """
        for order in orders:
            ep = None
            if entry_prices and order:
                ep = entry_prices.get(order.get('symbol'))
            self.log_trade(order, entry_price=ep)

    def log_portfolio(self, cash, portfolio_value, peak_value, positions):
        """Log portfolio snapshot and update metrics."""
        try:
            drawdown = 0.0
            if peak_value > 0:
                drawdown = (peak_value - portfolio_value) / peak_value

            positions_str = str(positions)

            with open(self.portfolio_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    f'{cash:.4f}',
                    f'{portfolio_value:.4f}',
                    f'{drawdown:.6f}',
                    positions_str,
                ])

            # Update live metrics
            self.metrics.record_portfolio(portfolio_value)

        except Exception as e:
            logger.error(f"Failed to log portfolio: {e}")

    def log_decision(self, raw_actions, filtered_actions, current_prices, portfolio_value):
        """Log model decision details."""
        try:
            with open(self.decisions_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    str(raw_actions.tolist()) if hasattr(raw_actions, 'tolist') else str(raw_actions),
                    str(filtered_actions),
                    str(current_prices),
                    f'{portfolio_value:.4f}',
                ])
        except Exception as e:
            logger.error(f"Failed to log decision: {e}")

    def log_metrics_snapshot(self):
        """Write current metrics to CSV (call periodically)."""
        try:
            m = self.metrics.get_summary_dict()
            with open(self.metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    m['total_trades'],
                    f"{m['win_rate']:.4f}",
                    f"{m['total_pnl']:.4f}",
                    f"{m['daily_pnl']:.4f}",
                    f"{m['daily_pnl_pct']:.6f}",
                    f"{m['max_drawdown']:.6f}",
                    f"{m['sharpe_ratio']:.4f}",
                    f"{m['sortino_ratio']:.4f}",
                    f"{m['current_value']:.4f}",
                ])
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def print_summary(self):
        """Print comprehensive summary of trading session."""
        try:
            m = self.metrics.get_summary_dict()

            print(f"\n{'='*60}")
            print(f"  TRADING SESSION SUMMARY")
            print(f"{'='*60}")
            print(f"  Total trades:      {m['total_trades']}")
            print(f"  Win rate:          {m['win_rate']:.1%}")
            print(f"  Total P&L:         ${m['total_pnl']:.2f}")
            print(f"  Daily P&L:         ${m['daily_pnl']:.2f} ({m['daily_pnl_pct']:.1%})")
            print(f"  Max drawdown:      {m['max_drawdown']:.1%}")
            print(f"  Sharpe ratio:      {m['sharpe_ratio']:.2f}")
            print(f"  Sortino ratio:     {m['sortino_ratio']:.2f}")
            print(f"  Peak value:        ${m['peak_value']:.2f}")
            print(f"  Final value:       ${m['current_value']:.2f}")

            wr_per_ticker = m.get('win_rate_per_ticker', {})
            if wr_per_ticker:
                print(f"\n  Win rate per ticker:")
                for t, wr in wr_per_ticker.items():
                    print(f"    {t}: {wr:.1%}")

            print(f"\n  Log files:")
            print(f"    Trades:    {self.trades_file}")
            print(f"    Portfolio: {self.portfolio_file}")
            print(f"    Decisions: {self.decisions_file}")
            print(f"    Metrics:   {self.metrics_file}")
            print(f"{'='*60}\n")
        except Exception as e:
            logger.error(f"Failed to print summary: {e}")
