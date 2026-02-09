"""
WebSocket-based real-time price monitor.
Runs in a background thread to detect price spikes between trading intervals.
Triggers stop-loss and panic actions immediately without waiting for the next cycle.
"""

import threading
import time
import logging

logger = logging.getLogger(__name__)


class PriceMonitor:
    """Background thread that polls prices at high frequency to detect
    stop-loss triggers between DRL decision intervals.

    Uses REST polling (via ExchangeManager) rather than raw WebSockets
    for maximum exchange compatibility. CCXT rate limiting is respected.
    """

    def __init__(self, exchange_manager, risk_manager, ticker_list,
                 check_interval_sec=5, on_stop_loss=None, on_panic=None):
        """
        Args:
            exchange_manager: ExchangeManager instance
            risk_manager: RiskManager instance (from environment_live)
            ticker_list: list of symbols to monitor
            check_interval_sec: seconds between price checks
            on_stop_loss: callback(symbol, price) when stop-loss triggers
            on_panic: callback() when portfolio drawdown kill switch triggers
        """
        self.exchange = exchange_manager
        self.risk_manager = risk_manager
        self.ticker_list = ticker_list
        self.check_interval = max(2, check_interval_sec)
        self.on_stop_loss = on_stop_loss
        self.on_panic = on_panic

        self._running = False
        self._thread = None
        self._latest_prices = {}
        self._lock = threading.Lock()

    def start(self):
        """Start the background price monitoring thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"Price monitor started: checking every {self.check_interval}s "
            f"for {len(self.ticker_list)} tickers"
        )

    def stop(self):
        """Stop the background monitoring thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        logger.info("Price monitor stopped")

    def get_latest_prices(self):
        """Thread-safe access to latest prices."""
        with self._lock:
            return self._latest_prices.copy()

    def _monitor_loop(self):
        """Main monitoring loop — runs in background thread."""
        while self._running:
            try:
                prices = {}
                for symbol in self.ticker_list:
                    try:
                        price = self.exchange.get_ticker_price(symbol)
                        prices[symbol] = price
                    except Exception as e:
                        logger.debug(f"Price monitor: failed to get {symbol}: {e}")
                        continue

                with self._lock:
                    self._latest_prices = prices

                # Check stop-loss for each position
                for symbol, price in prices.items():
                    if self.risk_manager.check_stop_loss(symbol, price):
                        logger.warning(f"MONITOR: Stop-loss triggered for {symbol} @ {price}")
                        if self.on_stop_loss:
                            try:
                                self.on_stop_loss(symbol, price)
                            except Exception as e:
                                logger.error(f"Stop-loss callback error: {e}")

                # Check portfolio drawdown (approximate using latest prices)
                if self.risk_manager.kill_switch_active:
                    logger.critical("MONITOR: Kill switch detected")
                    if self.on_panic:
                        try:
                            self.on_panic()
                        except Exception as e:
                            logger.error(f"Panic callback error: {e}")
                    self._running = False
                    break

            except Exception as e:
                logger.error(f"Price monitor error: {e}")

            time.sleep(self.check_interval)

    @property
    def is_running(self):
        return self._running
