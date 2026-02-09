"""
Unified exchange interface using CCXT.
Supports Binance and Bitget (with passphrase) for live trading.
Includes paper trading mode with realistic simulation (slippage, partial fills, fees).
Supports spot and futures with leverage/margin management.
"""

import ccxt
import time
import random
import logging
import numpy as np
from datetime import datetime
from functools import wraps

from config_api import (
    API_KEY_BINANCE, API_SECRET_BINANCE,
    API_KEY_BITGET, API_SECRET_BITGET, API_PASSPHRASE_BITGET
)

logger = logging.getLogger(__name__)


# ---- Retry decorator for API resilience ----

def retry_on_failure(max_retries=3, delay=2.0, backoff=2.0):
    """Decorator: retry API calls on transient failures with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (ccxt.NetworkError, ccxt.ExchangeNotAvailable,
                        ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
                    last_exception = e
                    logger.warning(
                        f"Retry {attempt}/{max_retries} for {func.__name__}: {e}"
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
                except ccxt.RateLimitExceeded as e:
                    last_exception = e
                    wait = current_delay * 2
                    logger.warning(f"Rate limit hit in {func.__name__}, waiting {wait:.0f}s")
                    time.sleep(wait)
                    current_delay *= backoff
            logger.error(f"{func.__name__} failed after {max_retries} retries: {last_exception}")
            raise last_exception
        return wrapper
    return decorator


class ExchangeManager:
    def __init__(self, exchange_name='binance', paper_trading=True,
                 market_type='spot', leverage=1, slippage_pct=0.0005):
        """
        Args:
            exchange_name: 'binance' or 'bitget'
            paper_trading: if True, simulate orders against real prices
            market_type: 'spot' or 'futures'
            leverage: leverage multiplier for futures (1 = no leverage)
            slippage_pct: simulated slippage for paper trading (0.05% default)
        """
        self.exchange_name = exchange_name.lower()
        self.paper_trading = paper_trading
        self.market_type = market_type
        self.leverage = max(1, int(leverage))
        self.slippage_pct = slippage_pct
        self.exchange = None
        self.panic_mode = False

        # Fee rates (updated from exchange on connect)
        self.maker_fee = 0.001   # 0.1% default
        self.taker_fee = 0.001   # 0.1% default

        # Paper trading state
        self._paper_balance = {}
        self._paper_positions = {}
        self._paper_orders = []
        self._paper_margin_used = 0.0

        self._connect()

    def _connect(self):
        """Initialize and test exchange connection."""
        if self.exchange_name == 'binance':
            if not API_KEY_BINANCE or not API_SECRET_BINANCE:
                raise ValueError("Binance API keys not configured. Set API_KEY_BINANCE and API_SECRET_BINANCE in .env")
            self.exchange = ccxt.binance({
                'apiKey': API_KEY_BINANCE,
                'secret': API_SECRET_BINANCE,
                'enableRateLimit': True,
                'options': {
                    'defaultType': self.market_type,
                },
            })

        elif self.exchange_name == 'bitget':
            if not API_KEY_BITGET or not API_SECRET_BITGET or not API_PASSPHRASE_BITGET:
                raise ValueError(
                    "Bitget API keys not configured. "
                    "Set API_KEY_BITGET, API_SECRET_BITGET, and API_PASSPHRASE_BITGET in .env"
                )
            self.exchange = ccxt.bitget({
                'apiKey': API_KEY_BITGET,
                'secret': API_SECRET_BITGET,
                'password': API_PASSPHRASE_BITGET,  # Bitget requires passphrase
                'enableRateLimit': True,
                'options': {
                    'defaultType': self.market_type,
                },
            })
        else:
            raise ValueError(f"Exchange '{self.exchange_name}' not supported. Use 'binance' or 'bitget'.")

        # Test connection and load markets
        try:
            self.exchange.load_markets()
            logger.info(f"Connected to {self.exchange_name} ({self.market_type})")
            if self.paper_trading:
                logger.info("PAPER TRADING MODE — no real orders will be placed")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self.exchange_name}: {e}")

        # Load real fee schedule
        self._load_fees()

        # Set leverage for futures
        if self.market_type == 'futures' and self.leverage > 1:
            logger.info(f"Futures mode: leverage={self.leverage}x")

    def _load_fees(self):
        """Load maker/taker fees from exchange."""
        try:
            if hasattr(self.exchange, 'fees') and 'trading' in self.exchange.fees:
                trading_fees = self.exchange.fees['trading']
                self.maker_fee = trading_fees.get('maker', 0.001)
                self.taker_fee = trading_fees.get('taker', 0.001)
            logger.info(f"Fees: maker={self.maker_fee*100:.3f}%, taker={self.taker_fee*100:.3f}%")
        except Exception:
            logger.warning("Could not load fee schedule, using defaults (0.1%)")

    def set_leverage(self, symbol, leverage):
        """Set leverage for a futures symbol on the exchange."""
        if self.market_type != 'futures':
            return
        self.leverage = max(1, int(leverage))
        if not self.paper_trading:
            try:
                self.exchange.set_leverage(self.leverage, symbol)
                logger.info(f"Leverage set to {self.leverage}x for {symbol}")
            except Exception as e:
                logger.warning(f"Could not set leverage for {symbol}: {e}")

    def get_margin_info(self):
        """Get margin/account info for futures trading.
        Returns dict with total_margin, used_margin, free_margin, margin_ratio.
        """
        if self.market_type != 'futures':
            return {'total_margin': 0, 'used_margin': 0, 'free_margin': 0, 'margin_ratio': 0}

        if self.paper_trading:
            usdt = self._paper_balance.get('USDT', 0.0)
            return {
                'total_margin': usdt,
                'used_margin': self._paper_margin_used,
                'free_margin': usdt - self._paper_margin_used,
                'margin_ratio': self._paper_margin_used / usdt if usdt > 0 else 0,
            }

        try:
            balance = self.exchange.fetch_balance()
            info = balance.get('info', {})
            total = float(balance.get('total', {}).get('USDT', 0))
            used = float(balance.get('used', {}).get('USDT', 0))
            return {
                'total_margin': total,
                'used_margin': used,
                'free_margin': total - used,
                'margin_ratio': used / total if total > 0 else 0,
            }
        except Exception as e:
            logger.error(f"Failed to fetch margin info: {e}")
            return {'total_margin': 0, 'used_margin': 0, 'free_margin': 0, 'margin_ratio': 0}

    def get_liquidation_price(self, symbol, side, entry_price, amount):
        """Estimate liquidation price for a futures position.
        Args:
            symbol: trading pair
            side: 'long' or 'short'
            entry_price: position entry price
            amount: position size in base currency
        Returns:
            estimated liquidation price (float)
        """
        if self.leverage <= 1:
            return 0.0  # no liquidation risk without leverage
        margin = (entry_price * amount) / self.leverage
        maint_margin_rate = 0.005  # 0.5% maintenance margin (conservative estimate)
        if side == 'long':
            liq_price = entry_price * (1 - (1 / self.leverage) + maint_margin_rate)
        else:
            liq_price = entry_price * (1 + (1 / self.leverage) - maint_margin_rate)
        return max(0, liq_price)

    # ---- Panic Button ----

    def activate_panic_mode(self, ticker_list):
        """EMERGENCY: Cancel all orders and close all positions immediately."""
        self.panic_mode = True
        logger.critical("PANIC MODE ACTIVATED — closing all positions")

        # Cancel all open orders
        self.cancel_all_orders()

        # Close all positions
        positions = self.get_positions(ticker_list)
        for symbol, amount in positions.items():
            if amount > 0:
                try:
                    self.place_market_sell(symbol, amount)
                    logger.critical(f"PANIC SELL: {amount} {symbol}")
                except Exception as e:
                    logger.critical(f"PANIC SELL FAILED for {symbol}: {e}")

        logger.critical("Panic mode complete — all positions closed")

    def deactivate_panic_mode(self):
        """Reset panic mode to allow trading again."""
        self.panic_mode = False
        logger.info("Panic mode deactivated")

    @retry_on_failure(max_retries=3, delay=1.0)
    def get_balance(self):
        """Return dict of all balances: {'USDT': 1000.0, 'BTC': 0.01, ...}"""
        if self.paper_trading:
            return self._paper_balance.copy()

        balance = self.exchange.fetch_balance()
        result = {}
        for currency, info in balance['total'].items():
            if info and info > 0:
                result[currency] = float(info)
        return result

    def get_usdt_balance(self):
        """Return available USDT balance."""
        balance = self.get_balance()
        return balance.get('USDT', 0.0)

    @retry_on_failure(max_retries=3, delay=1.0)
    def get_positions(self, ticker_list):
        """Return current positions for given tickers.
        Returns dict: {'BTC/USDT': 0.01, 'ETH/USDT': 0.5, ...}
        """
        if self.paper_trading:
            return {t: self._paper_positions.get(t, 0.0) for t in ticker_list}

        if self.market_type == 'futures':
            return self._get_futures_positions(ticker_list)

        balance = self.exchange.fetch_balance()
        positions = {}
        for ticker in ticker_list:
            base = ticker.split('/')[0]
            amount = float(balance['total'].get(base, 0))
            positions[ticker] = amount
        return positions

    def _get_futures_positions(self, ticker_list):
        """Fetch futures positions via CCXT."""
        positions = {}
        try:
            all_positions = self.exchange.fetch_positions(ticker_list)
            for pos in all_positions:
                symbol = pos.get('symbol', '')
                contracts = float(pos.get('contracts', 0) or 0)
                side = pos.get('side', '')
                if side == 'short':
                    contracts = -contracts
                positions[symbol] = contracts
        except Exception as e:
            logger.error(f"Failed to fetch futures positions: {e}")
        # Fill missing tickers with 0
        for t in ticker_list:
            if t not in positions:
                positions[t] = 0.0
        return positions

    @retry_on_failure(max_retries=3, delay=1.0)
    def get_ticker_price(self, symbol):
        """Get current price for a symbol."""
        ticker = self.exchange.fetch_ticker(symbol)
        return float(ticker['last'])

    def get_current_prices(self, ticker_list):
        """Get current prices for all tickers. Returns dict."""
        prices = {}
        for symbol in ticker_list:
            prices[symbol] = self.get_ticker_price(symbol)
        return prices

    @retry_on_failure(max_retries=3, delay=1.0)
    def get_latest_candles(self, symbol, timeframe='5m', limit=100):
        """Fetch recent OHLCV candles.
        Returns list of [timestamp, open, high, low, close, volume].
        """
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return ohlcv

    def get_min_order_size(self, symbol):
        """Get minimum order size for a symbol from exchange info."""
        try:
            market = self.exchange.market(symbol)
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0.001)
            return float(min_amount) if min_amount else 0.001
        except Exception as e:
            logger.warning(f"Could not get min order size for {symbol}, using default 0.001: {e}")
            return 0.001

    def get_min_order_sizes(self, ticker_list):
        """Get minimum order sizes for all tickers."""
        return {symbol: self.get_min_order_size(symbol) for symbol in ticker_list}

    def place_market_buy(self, symbol, amount):
        """Place a market buy order with retry logic.
        Args:
            symbol: e.g. 'BTC/USDT'
            amount: quantity in base currency (e.g. 0.001 BTC)
        Returns:
            order dict or paper trade record
        """
        if self.panic_mode:
            logger.warning("Panic mode active — buy orders blocked")
            return None

        min_size = self.get_min_order_size(symbol)
        if amount < min_size:
            logger.warning(f"Buy amount {amount} < min {min_size} for {symbol}. Skipping.")
            return None

        if self.paper_trading:
            return self._paper_market_buy(symbol, amount)

        return self._execute_with_retry(
            lambda: self.exchange.create_market_buy_order(symbol, amount),
            f"BUY {amount} {symbol}"
        )

    def place_market_sell(self, symbol, amount):
        """Place a market sell order with retry logic.
        Args:
            symbol: e.g. 'BTC/USDT'
            amount: quantity in base currency
        Returns:
            order dict or paper trade record
        """
        min_size = self.get_min_order_size(symbol)
        if amount < min_size:
            logger.warning(f"Sell amount {amount} < min {min_size} for {symbol}. Skipping.")
            return None

        if self.paper_trading:
            return self._paper_market_sell(symbol, amount)

        return self._execute_with_retry(
            lambda: self.exchange.create_market_sell_order(symbol, amount),
            f"SELL {amount} {symbol}"
        )

    def place_limit_buy(self, symbol, amount, price):
        """Place a limit buy order."""
        if self.panic_mode:
            logger.warning("Panic mode active — buy orders blocked")
            return None

        min_size = self.get_min_order_size(symbol)
        if amount < min_size:
            logger.warning(f"Buy amount {amount} < min {min_size} for {symbol}. Skipping.")
            return None

        if self.paper_trading:
            return self._paper_market_buy(symbol, amount)

        return self._execute_with_retry(
            lambda: self.exchange.create_limit_buy_order(symbol, amount, price),
            f"LIMIT BUY {amount} {symbol} @ {price}"
        )

    def place_limit_sell(self, symbol, amount, price):
        """Place a limit sell order."""
        min_size = self.get_min_order_size(symbol)
        if amount < min_size:
            logger.warning(f"Sell amount {amount} < min {min_size} for {symbol}. Skipping.")
            return None

        if self.paper_trading:
            return self._paper_market_sell(symbol, amount)

        return self._execute_with_retry(
            lambda: self.exchange.create_limit_sell_order(symbol, amount, price),
            f"LIMIT SELL {amount} {symbol} @ {price}"
        )

    def _execute_with_retry(self, order_func, description, max_retries=3):
        """Execute an order function with retry logic for transient failures."""
        for attempt in range(1, max_retries + 1):
            try:
                order = order_func()
                logger.info(f"{description} @ market | order_id={order['id']}")
                return order
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable,
                    ccxt.RequestTimeout) as e:
                logger.warning(f"Retry {attempt}/{max_retries} for {description}: {e}")
                time.sleep(1.0 * attempt)
            except ccxt.InsufficientFunds as e:
                logger.error(f"Insufficient funds for {description}: {e}")
                return None
            except ccxt.InvalidOrder as e:
                logger.error(f"Invalid order {description}: {e}")
                return None
            except Exception as e:
                logger.error(f"Failed {description}: {e}")
                return None
        logger.error(f"{description} failed after {max_retries} retries")
        return None

    def cancel_all_orders(self, symbol=None):
        """Cancel all open orders for a symbol (or all symbols)."""
        if self.paper_trading:
            self._paper_orders.clear()
            return

        try:
            if symbol:
                self.exchange.cancel_all_orders(symbol)
                logger.info(f"Cancelled all orders for {symbol}")
            else:
                open_orders = self.exchange.fetch_open_orders()
                for order in open_orders:
                    self.exchange.cancel_order(order['id'], order['symbol'])
                logger.info(f"Cancelled {len(open_orders)} open orders")
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")

    # ---- Paper Trading Simulation (realistic: slippage, partial fills, fees) ----

    def init_paper_balance(self, usdt_amount):
        """Initialize paper trading with USDT balance."""
        self._paper_balance = {'USDT': float(usdt_amount)}
        self._paper_positions = {}
        self._paper_orders = []
        self._paper_margin_used = 0.0
        logger.info(f"Paper trading initialized with {usdt_amount} USDT")

    def _simulate_slippage(self, price, side):
        """Apply realistic slippage to a price.
        Buys slip up, sells slip down.
        """
        slip = random.uniform(0, self.slippage_pct)
        if side == 'buy':
            return price * (1 + slip)
        else:
            return price * (1 - slip)

    def _simulate_partial_fill(self, amount):
        """Simulate partial fills — 90-100% of requested amount gets filled."""
        fill_ratio = random.uniform(0.90, 1.0)
        return amount * fill_ratio

    def _paper_market_buy(self, symbol, amount):
        """Simulate a market buy with slippage, partial fills, and real fees."""
        real_price = self.get_ticker_price(symbol)
        fill_price = self._simulate_slippage(real_price, 'buy')
        filled_amount = self._simulate_partial_fill(amount)

        cost = filled_amount * fill_price
        fee = cost * self.taker_fee
        total_cost = cost + fee

        usdt_balance = self._paper_balance.get('USDT', 0.0)
        if total_cost > usdt_balance:
            # Try to fill what we can afford
            affordable = usdt_balance / (fill_price * (1 + self.taker_fee))
            if affordable < self.get_min_order_size(symbol):
                logger.warning(f"Paper: Insufficient USDT ({usdt_balance:.2f}) for {symbol}")
                return None
            filled_amount = affordable
            cost = filled_amount * fill_price
            fee = cost * self.taker_fee
            total_cost = cost + fee

        # Update margin for futures
        if self.market_type == 'futures' and self.leverage > 1:
            margin_required = cost / self.leverage
            self._paper_margin_used += margin_required

        self._paper_balance['USDT'] = usdt_balance - total_cost
        current_pos = self._paper_positions.get(symbol, 0.0)
        self._paper_positions[symbol] = current_pos + filled_amount

        order = {
            'id': f'paper_{len(self._paper_orders)}',
            'symbol': symbol,
            'side': 'buy',
            'amount': filled_amount,
            'requested_amount': amount,
            'price': fill_price,
            'market_price': real_price,
            'slippage': fill_price - real_price,
            'cost': cost,
            'fee': fee,
            'timestamp': datetime.now().isoformat(),
            'paper': True,
        }
        self._paper_orders.append(order)
        logger.info(
            f"PAPER BUY {filled_amount:.6f}/{amount:.6f} {symbol} "
            f"@ {fill_price:.2f} (mkt {real_price:.2f}, slip {(fill_price-real_price):.2f}) "
            f"| cost={total_cost:.2f} fee={fee:.4f}"
        )
        return order

    def _paper_market_sell(self, symbol, amount):
        """Simulate a market sell with slippage, partial fills, and real fees."""
        current_pos = self._paper_positions.get(symbol, 0.0)
        if amount > current_pos:
            amount = current_pos

        if amount <= 0:
            logger.warning(f"Paper: No position to sell for {symbol}")
            return None

        real_price = self.get_ticker_price(symbol)
        fill_price = self._simulate_slippage(real_price, 'sell')
        filled_amount = self._simulate_partial_fill(amount)
        filled_amount = min(filled_amount, current_pos)

        revenue = filled_amount * fill_price
        fee = revenue * self.taker_fee
        net_revenue = revenue - fee

        # Release margin for futures
        if self.market_type == 'futures' and self.leverage > 1:
            margin_released = revenue / self.leverage
            self._paper_margin_used = max(0, self._paper_margin_used - margin_released)

        self._paper_balance['USDT'] = self._paper_balance.get('USDT', 0.0) + net_revenue
        self._paper_positions[symbol] = current_pos - filled_amount

        order = {
            'id': f'paper_{len(self._paper_orders)}',
            'symbol': symbol,
            'side': 'sell',
            'amount': filled_amount,
            'requested_amount': amount,
            'price': fill_price,
            'market_price': real_price,
            'slippage': real_price - fill_price,
            'cost': revenue,
            'fee': fee,
            'timestamp': datetime.now().isoformat(),
            'paper': True,
        }
        self._paper_orders.append(order)
        logger.info(
            f"PAPER SELL {filled_amount:.6f}/{amount:.6f} {symbol} "
            f"@ {fill_price:.2f} (mkt {real_price:.2f}, slip {(real_price-fill_price):.2f}) "
            f"| revenue={net_revenue:.2f} fee={fee:.4f}"
        )
        return order

    def get_paper_portfolio_value(self, ticker_list):
        """Calculate total paper portfolio value in USDT."""
        total = self._paper_balance.get('USDT', 0.0)
        for symbol in ticker_list:
            pos = self._paper_positions.get(symbol, 0.0)
            if pos > 0:
                price = self.get_ticker_price(symbol)
                total += pos * price
        return total

    def close(self):
        """Cleanup exchange connection."""
        if self.exchange and hasattr(self.exchange, 'close'):
            try:
                self.exchange.close()
            except Exception:
                pass
