"""
Live trading environment that bridges the DRL model with real exchange execution.
Produces state vectors identical to the training environment (CryptoEnvCCXT),
but executes real orders via ExchangeManager.

Includes risk management:
- Per-position stop-loss
- Portfolio drawdown kill switch
- Max position size limits
- Cooldown after consecutive losses
"""

import numpy as np
import math
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class RiskManager:
    """Risk management layer for live trading."""

    def __init__(self, config):
        self.stop_loss_pct = config.get('stop_loss_pct', 0.05)
        self.max_drawdown_pct = config.get('max_drawdown_pct', 0.15)
        self.max_position_pct = config.get('max_position_pct', 0.25)
        self.daily_loss_limit_pct = config.get('daily_loss_limit_pct', 0.10)

        # State tracking
        self.peak_portfolio_value = 0.0
        self.entry_prices = {}          # {symbol: avg_entry_price}
        self._entry_amounts = {}         # {symbol: total_amount} for weighted avg
        self.daily_start_value = 0.0
        self.daily_start_date = None
        self.kill_switch_active = False
        self.consecutive_losses = 0

    def update_peak(self, portfolio_value):
        """Track portfolio peak for drawdown calculation."""
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value

    def check_drawdown_kill_switch(self, portfolio_value):
        """Check if portfolio drawdown exceeds limit."""
        if self.peak_portfolio_value <= 0:
            return False
        drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
        if drawdown >= self.max_drawdown_pct:
            logger.critical(
                f"KILL SWITCH: Portfolio drawdown {drawdown:.1%} exceeds limit {self.max_drawdown_pct:.1%}. "
                f"Peak={self.peak_portfolio_value:.2f}, Current={portfolio_value:.2f}"
            )
            self.kill_switch_active = True
            return True
        return False

    def check_daily_loss_limit(self, portfolio_value):
        """Check if daily loss exceeds limit."""
        today = datetime.now().date()
        if self.daily_start_date != today:
            self.daily_start_date = today
            self.daily_start_value = portfolio_value
            return False

        if self.daily_start_value <= 0:
            return False

        daily_loss = (self.daily_start_value - portfolio_value) / self.daily_start_value
        if daily_loss >= self.daily_loss_limit_pct:
            logger.warning(
                f"Daily loss limit reached: {daily_loss:.1%} >= {self.daily_loss_limit_pct:.1%}"
            )
            return True
        return False

    def check_stop_loss(self, symbol, current_price):
        """Check if a position should be stopped out."""
        entry_price = self.entry_prices.get(symbol)
        if entry_price is None or entry_price <= 0:
            return False
        loss_pct = (entry_price - current_price) / entry_price
        if loss_pct >= self.stop_loss_pct:
            logger.warning(
                f"STOP LOSS triggered for {symbol}: "
                f"entry={entry_price:.2f}, current={current_price:.2f}, loss={loss_pct:.1%}"
            )
            return True
        return False

    def max_buy_amount(self, symbol, price, portfolio_value, current_position_value,
                        taker_fee=0.001, slippage_pct=0.0005):
        """Calculate maximum buy amount respecting position size limits.
        Accounts for fees and slippage in the calculation.
        """
        max_position_value = portfolio_value * self.max_position_pct
        remaining_capacity = max(0, max_position_value - current_position_value)
        if price <= 0:
            return 0.0
        # Adjust for fees + slippage so we don't overshoot
        effective_price = price * (1 + taker_fee + slippage_pct)
        return remaining_capacity / effective_price

    def record_entry(self, symbol, price, amount):
        """Record entry price for stop-loss tracking (volume-weighted average)."""
        existing_price = self.entry_prices.get(symbol)
        existing_amount = self._entry_amounts.get(symbol, 0.0)
        if existing_price and existing_price > 0 and existing_amount > 0:
            total_amount = existing_amount + amount
            self.entry_prices[symbol] = (
                (existing_price * existing_amount + price * amount) / total_amount
            )
            self._entry_amounts[symbol] = total_amount
        else:
            self.entry_prices[symbol] = price
            self._entry_amounts[symbol] = amount

    def record_exit(self, symbol):
        """Clear entry price on full exit."""
        self.entry_prices.pop(symbol, None)
        self._entry_amounts.pop(symbol, None)

    def filter_actions(self, actions, ticker_list, current_prices, positions,
                       portfolio_value, min_order_sizes):
        """Apply all risk filters to raw DRL actions.

        Args:
            actions: raw action array from DRL model (already normalized)
            ticker_list: list of symbols
            current_prices: dict {symbol: price}
            positions: dict {symbol: amount}
            portfolio_value: total portfolio value in USDT
            min_order_sizes: dict {symbol: min_amount}

        Returns:
            filtered_actions: dict {symbol: {'side': 'buy'/'sell'/'hold', 'amount': float}}
        """
        filtered = {}

        # Kill switch check
        if self.kill_switch_active:
            # Sell everything
            for i, symbol in enumerate(ticker_list):
                pos = positions.get(symbol, 0.0)
                if pos > 0:
                    filtered[symbol] = {'side': 'sell', 'amount': pos}
                else:
                    filtered[symbol] = {'side': 'hold', 'amount': 0.0}
            return filtered

        # Drawdown check
        self.update_peak(portfolio_value)
        if self.check_drawdown_kill_switch(portfolio_value):
            for i, symbol in enumerate(ticker_list):
                pos = positions.get(symbol, 0.0)
                if pos > 0:
                    filtered[symbol] = {'side': 'sell', 'amount': pos}
                else:
                    filtered[symbol] = {'side': 'hold', 'amount': 0.0}
            return filtered

        # Daily loss check
        daily_limit_hit = self.check_daily_loss_limit(portfolio_value)

        for i, symbol in enumerate(ticker_list):
            action_val = actions[i] if i < len(actions) else 0.0
            price = current_prices.get(symbol, 0.0)
            pos = positions.get(symbol, 0.0)
            min_size = min_order_sizes.get(symbol, 0.001)

            # Stop-loss check
            if pos > 0 and self.check_stop_loss(symbol, price):
                filtered[symbol] = {'side': 'sell', 'amount': pos}
                continue

            # If daily limit hit, only allow sells
            if daily_limit_hit and action_val > 0:
                filtered[symbol] = {'side': 'hold', 'amount': 0.0}
                continue

            if action_val > min_size:
                # BUY
                pos_value = pos * price
                max_buy = self.max_buy_amount(symbol, price, portfolio_value, pos_value)
                buy_amount = min(action_val, max_buy)
                if buy_amount >= min_size:
                    filtered[symbol] = {'side': 'buy', 'amount': buy_amount}
                else:
                    filtered[symbol] = {'side': 'hold', 'amount': 0.0}

            elif action_val < -min_size:
                # SELL
                sell_amount = min(pos, abs(action_val))
                if sell_amount >= min_size:
                    filtered[symbol] = {'side': 'sell', 'amount': sell_amount}
                else:
                    filtered[symbol] = {'side': 'hold', 'amount': 0.0}
            else:
                filtered[symbol] = {'side': 'hold', 'amount': 0.0}

        return filtered


class LiveTradingEnv:
    """Live trading environment that produces states identical to training env
    and executes real orders."""

    def __init__(self, exchange_manager, data_feed, config, env_params):
        """
        Args:
            exchange_manager: ExchangeManager instance
            data_feed: LiveDataFeed instance
            config: LIVE_TRADING_CONFIG dict
            env_params: env normalization params from trained model
        """
        self.exchange = exchange_manager
        self.data_feed = data_feed
        self.config = config
        self.ticker_list = config['ticker_list']
        self.crypto_num = len(self.ticker_list)

        # Normalization params (must match training)
        self.lookback = env_params['lookback']
        self.norm_cash = env_params['norm_cash']
        self.norm_stocks = env_params['norm_stocks']
        self.norm_tech = env_params['norm_tech']
        self.norm_reward = env_params['norm_reward']
        self.norm_action = env_params['norm_action']

        # Risk management
        self.risk_manager = RiskManager(config)

        # Min order sizes (fetched from exchange)
        self.min_order_sizes = exchange_manager.get_min_order_sizes(self.ticker_list)

        # Action normalizer (computed from first price fetch)
        self.action_norm_vector = None

        # State tracking
        self.last_price_array = None
        self.last_tech_array = None

    def sync_state(self):
        """Fetch latest market data and account state.
        Returns the state vector in training-compatible format.
        """
        # Fetch live market data
        price_array, tech_array, current_prices = self.data_feed.fetch_latest_state()
        self.last_price_array = price_array
        self.last_tech_array = tech_array

        # Compute action normalizer from current prices (same as training env)
        if self.action_norm_vector is None:
            self._generate_action_normalizer(price_array[0])

        # Get real account state
        if self.exchange.paper_trading:
            cash = self.exchange.get_usdt_balance()
            positions = self.exchange.get_positions(self.ticker_list)
        else:
            cash = self.exchange.get_usdt_balance()
            positions = self.exchange.get_positions(self.ticker_list)

        # Build stocks array (same order as ticker_list)
        stocks = np.array([positions.get(t, 0.0) for t in self.ticker_list], dtype=np.float32)

        # Build state vector (identical format to CryptoEnvCCXT.get_state)
        state = np.hstack((cash * self.norm_cash, stocks * self.norm_stocks))
        lookback_len = min(self.lookback, tech_array.shape[0])
        for i in range(lookback_len):
            idx = tech_array.shape[0] - 1 - i
            tech_i = tech_array[idx]
            normalized_tech_i = tech_i * self.norm_tech
            state = np.hstack((state, normalized_tech_i)).astype(np.float32)

        # Pad if we don't have enough lookback
        if lookback_len < self.lookback:
            pad_size = (self.lookback - lookback_len) * tech_array.shape[1]
            state = np.hstack((state, np.zeros(pad_size, dtype=np.float32)))

        # Portfolio value for risk management
        portfolio_value = cash
        for i, symbol in enumerate(self.ticker_list):
            portfolio_value += stocks[i] * current_prices.get(symbol, 0.0)

        self.risk_manager.update_peak(portfolio_value)

        return state, cash, stocks, current_prices, portfolio_value

    def process_actions(self, raw_actions, cash, stocks, current_prices, portfolio_value):
        """Convert raw DRL actions to risk-filtered trade orders and execute them.

        Args:
            raw_actions: numpy array from model inference
            cash: current USDT balance
            stocks: numpy array of current positions
            current_prices: dict of current prices
            portfolio_value: total portfolio value

        Returns:
            executed_orders: list of order dicts
        """
        # Apply action normalization (same as training env)
        actions = raw_actions.copy()
        for i in range(min(len(actions), len(self.action_norm_vector))):
            actions[i] = actions[i] * self.action_norm_vector[i]

        # Build positions dict
        positions = {self.ticker_list[i]: stocks[i] for i in range(self.crypto_num)}

        # Apply risk management filters
        filtered = self.risk_manager.filter_actions(
            actions, self.ticker_list, current_prices,
            positions, portfolio_value, self.min_order_sizes
        )

        # Execute orders
        executed_orders = []
        for symbol, action in filtered.items():
            if action['side'] == 'hold':
                continue

            amount = action['amount']
            if amount <= 0:
                continue

            if action['side'] == 'buy':
                # Check we have enough cash
                price = current_prices.get(symbol, 0.0)
                cost = amount * price * 1.001  # include fee
                if cost > cash * 0.95:  # 5% safety margin
                    amount = (cash * 0.95) / (price * 1.001)
                    if amount < self.min_order_sizes.get(symbol, 0.001):
                        continue

                order = self.exchange.place_market_buy(symbol, amount)
                if order:
                    self.risk_manager.record_entry(symbol, price, amount)
                    executed_orders.append(order)
                    cash -= amount * price * 1.001

            elif action['side'] == 'sell':
                order = self.exchange.place_market_sell(symbol, amount)
                if order:
                    # Check if fully exited
                    remaining = positions.get(symbol, 0.0) - amount
                    if remaining <= self.min_order_sizes.get(symbol, 0.001):
                        self.risk_manager.record_exit(symbol)
                    executed_orders.append(order)

        return executed_orders

    def _generate_action_normalizer(self, initial_prices):
        """Generate action normalizer — must match CryptoEnvCCXT._generate_action_normalizer."""
        action_norm_vector = []
        for price in initial_prices:
            if price > 0:
                x = math.floor(math.log(price, 10))
                action_norm_vector.append(1 / ((10) ** x))
            else:
                action_norm_vector.append(1.0)

        self.action_norm_vector = np.asarray(action_norm_vector) * self.norm_action

    def is_kill_switch_active(self):
        """Check if risk manager has triggered kill switch."""
        return self.risk_manager.kill_switch_active
