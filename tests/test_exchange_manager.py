"""
Unit tests for ExchangeManager.
Tests paper trading simulation, order execution, balance management,
min order sizes, slippage, partial fills, and panic mode.

Run: python -m pytest tests/test_exchange_manager.py -v
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestExchangeManagerPaperTrading(unittest.TestCase):
    """Test paper trading functionality without real API keys."""

    def _create_paper_manager(self):
        """Create a paper trading ExchangeManager with mocked exchange."""
        with patch('exchange_manager.ccxt') as mock_ccxt:
            mock_exchange = MagicMock()
            mock_exchange.load_markets.return_value = {}
            mock_exchange.fees = {'trading': {'maker': 0.001, 'taker': 0.001}}
            mock_exchange.markets = {'BTC/USDT': {}, 'ETH/USDT': {}}

            # Mock fetch_ticker to return realistic prices
            def mock_fetch_ticker(symbol):
                prices = {'BTC/USDT': 50000.0, 'ETH/USDT': 3000.0}
                return {'last': prices.get(symbol, 100.0)}
            mock_exchange.fetch_ticker = mock_fetch_ticker

            # Mock market info for min order sizes
            def mock_market(symbol):
                return {'limits': {'amount': {'min': 0.0001}}}
            mock_exchange.market = mock_market

            mock_ccxt.binance.return_value = mock_exchange

            with patch('exchange_manager.API_KEY_BINANCE', 'test_key'), \
                 patch('exchange_manager.API_SECRET_BINANCE', 'test_secret'):
                from exchange_manager import ExchangeManager
                mgr = ExchangeManager(
                    exchange_name='binance',
                    paper_trading=True,
                    market_type='spot',
                    slippage_pct=0.0  # disable slippage for deterministic tests
                )
                mgr.init_paper_balance(10000.0)
                return mgr

    def test_init_paper_balance(self):
        mgr = self._create_paper_manager()
        self.assertEqual(mgr.get_usdt_balance(), 10000.0)

    def test_paper_buy(self):
        mgr = self._create_paper_manager()
        order = mgr.place_market_buy('BTC/USDT', 0.01)
        self.assertIsNotNone(order)
        self.assertEqual(order['side'], 'buy')
        self.assertEqual(order['symbol'], 'BTC/USDT')
        self.assertTrue(order['paper'])
        # Balance should decrease
        self.assertLess(mgr.get_usdt_balance(), 10000.0)

    def test_paper_sell(self):
        mgr = self._create_paper_manager()
        # Buy first
        mgr.place_market_buy('BTC/USDT', 0.01)
        balance_after_buy = mgr.get_usdt_balance()
        # Sell
        order = mgr.place_market_sell('BTC/USDT', 0.01)
        self.assertIsNotNone(order)
        self.assertEqual(order['side'], 'sell')
        # Balance should increase after sell
        self.assertGreater(mgr.get_usdt_balance(), balance_after_buy)

    def test_paper_sell_more_than_owned(self):
        mgr = self._create_paper_manager()
        mgr.place_market_buy('BTC/USDT', 0.01)
        # Try to sell more than we have — should cap at position
        order = mgr.place_market_sell('BTC/USDT', 1.0)
        self.assertIsNotNone(order)
        # Filled amount should be <= what we bought (with partial fill)
        self.assertLessEqual(order['amount'], 0.01)

    def test_paper_sell_no_position(self):
        mgr = self._create_paper_manager()
        order = mgr.place_market_sell('BTC/USDT', 0.01)
        self.assertIsNone(order)

    def test_min_order_size_enforcement(self):
        mgr = self._create_paper_manager()
        # Try to buy less than minimum
        order = mgr.place_market_buy('BTC/USDT', 0.00001)
        self.assertIsNone(order)

    def test_insufficient_balance(self):
        mgr = self._create_paper_manager()
        # Try to buy way more than we can afford
        order = mgr.place_market_buy('BTC/USDT', 1000.0)  # 1000 BTC = $50M
        self.assertIsNone(order)

    def test_get_positions(self):
        mgr = self._create_paper_manager()
        positions = mgr.get_positions(['BTC/USDT', 'ETH/USDT'])
        self.assertEqual(positions['BTC/USDT'], 0.0)
        self.assertEqual(positions['ETH/USDT'], 0.0)

        mgr.place_market_buy('BTC/USDT', 0.01)
        positions = mgr.get_positions(['BTC/USDT', 'ETH/USDT'])
        self.assertGreater(positions['BTC/USDT'], 0.0)
        self.assertEqual(positions['ETH/USDT'], 0.0)

    def test_portfolio_value(self):
        mgr = self._create_paper_manager()
        initial = mgr.get_paper_portfolio_value(['BTC/USDT', 'ETH/USDT'])
        self.assertAlmostEqual(initial, 10000.0, places=0)

        mgr.place_market_buy('BTC/USDT', 0.01)
        after_buy = mgr.get_paper_portfolio_value(['BTC/USDT', 'ETH/USDT'])
        # Should be close to initial (minus fees)
        self.assertAlmostEqual(after_buy, 10000.0, delta=100)

    def test_fees_deducted(self):
        mgr = self._create_paper_manager()
        mgr.place_market_buy('BTC/USDT', 0.01)
        # Cost = 0.01 * 50000 = 500, fee = 500 * 0.001 = 0.5
        # Balance should be ~10000 - 500.5 = 9499.5
        balance = mgr.get_usdt_balance()
        self.assertLess(balance, 9500.0)
        self.assertGreater(balance, 9490.0)

    def test_panic_mode_blocks_buys(self):
        mgr = self._create_paper_manager()
        mgr.panic_mode = True
        order = mgr.place_market_buy('BTC/USDT', 0.01)
        self.assertIsNone(order)

    def test_panic_mode_allows_sells(self):
        mgr = self._create_paper_manager()
        mgr.place_market_buy('BTC/USDT', 0.01)
        mgr.panic_mode = True
        # Sells should still work in panic mode
        order = mgr.place_market_sell('BTC/USDT', 0.01)
        self.assertIsNotNone(order)

    def test_cancel_all_orders_paper(self):
        mgr = self._create_paper_manager()
        mgr._paper_orders.append({'id': 'test'})
        mgr.cancel_all_orders()
        self.assertEqual(len(mgr._paper_orders), 0)


class TestExchangeManagerSlippage(unittest.TestCase):
    """Test slippage simulation."""

    def _create_manager_with_slippage(self, slippage_pct=0.01):
        with patch('exchange_manager.ccxt') as mock_ccxt:
            mock_exchange = MagicMock()
            mock_exchange.load_markets.return_value = {}
            mock_exchange.fees = {'trading': {'maker': 0.001, 'taker': 0.001}}

            def mock_fetch_ticker(symbol):
                return {'last': 50000.0}
            mock_exchange.fetch_ticker = mock_fetch_ticker

            def mock_market(symbol):
                return {'limits': {'amount': {'min': 0.0001}}}
            mock_exchange.market = mock_market

            mock_ccxt.binance.return_value = mock_exchange

            with patch('exchange_manager.API_KEY_BINANCE', 'test_key'), \
                 patch('exchange_manager.API_SECRET_BINANCE', 'test_secret'):
                from exchange_manager import ExchangeManager
                mgr = ExchangeManager(
                    exchange_name='binance',
                    paper_trading=True,
                    slippage_pct=slippage_pct,
                )
                mgr.init_paper_balance(100000.0)
                return mgr

    def test_buy_slippage_increases_price(self):
        mgr = self._create_manager_with_slippage(slippage_pct=0.01)
        order = mgr.place_market_buy('BTC/USDT', 0.1)
        self.assertIsNotNone(order)
        # Fill price should be >= market price (slippage up for buys)
        self.assertGreaterEqual(order['price'], order['market_price'])

    def test_sell_slippage_decreases_price(self):
        mgr = self._create_manager_with_slippage(slippage_pct=0.01)
        mgr.place_market_buy('BTC/USDT', 0.1)
        order = mgr.place_market_sell('BTC/USDT', 0.05)
        self.assertIsNotNone(order)
        # Fill price should be <= market price (slippage down for sells)
        self.assertLessEqual(order['price'], order['market_price'])

    def test_slippage_field_in_order(self):
        mgr = self._create_manager_with_slippage(slippage_pct=0.01)
        order = mgr.place_market_buy('BTC/USDT', 0.1)
        self.assertIn('slippage', order)
        self.assertGreaterEqual(order['slippage'], 0)


class TestExchangeManagerMarginFutures(unittest.TestCase):
    """Test futures/margin functionality."""

    def _create_futures_manager(self):
        with patch('exchange_manager.ccxt') as mock_ccxt:
            mock_exchange = MagicMock()
            mock_exchange.load_markets.return_value = {}
            mock_exchange.fees = {'trading': {'maker': 0.0002, 'taker': 0.0004}}
            mock_exchange.set_leverage = MagicMock()

            def mock_fetch_ticker(symbol):
                return {'last': 50000.0}
            mock_exchange.fetch_ticker = mock_fetch_ticker

            def mock_market(symbol):
                return {'limits': {'amount': {'min': 0.001}}}
            mock_exchange.market = mock_market

            mock_ccxt.binance.return_value = mock_exchange

            with patch('exchange_manager.API_KEY_BINANCE', 'test_key'), \
                 patch('exchange_manager.API_SECRET_BINANCE', 'test_secret'):
                from exchange_manager import ExchangeManager
                mgr = ExchangeManager(
                    exchange_name='binance',
                    paper_trading=True,
                    market_type='futures',
                    leverage=10,
                    slippage_pct=0.0,
                )
                mgr.init_paper_balance(10000.0)
                return mgr

    def test_leverage_stored(self):
        mgr = self._create_futures_manager()
        self.assertEqual(mgr.leverage, 10)

    def test_margin_info(self):
        mgr = self._create_futures_manager()
        info = mgr.get_margin_info()
        self.assertEqual(info['total_margin'], 10000.0)
        self.assertEqual(info['used_margin'], 0.0)
        self.assertEqual(info['free_margin'], 10000.0)

    def test_margin_used_after_buy(self):
        mgr = self._create_futures_manager()
        mgr.place_market_buy('BTC/USDT', 0.01)
        info = mgr.get_margin_info()
        # Margin used should be cost / leverage
        self.assertGreater(info['used_margin'], 0)
        self.assertLess(info['used_margin'], 100)  # 500 / 10 = 50

    def test_liquidation_price_long(self):
        mgr = self._create_futures_manager()
        liq = mgr.get_liquidation_price('BTC/USDT', 'long', 50000, 0.01)
        self.assertGreater(liq, 0)
        self.assertLess(liq, 50000)  # liq price below entry for longs

    def test_liquidation_price_no_leverage(self):
        mgr = self._create_futures_manager()
        mgr.leverage = 1
        liq = mgr.get_liquidation_price('BTC/USDT', 'long', 50000, 0.01)
        self.assertEqual(liq, 0.0)


if __name__ == '__main__':
    unittest.main()
