"""
Live Trading Bot — Step 6
Loads a trained DRL agent and executes real trades on Binance or Bitget.

Usage:
    python 6_live_trade.py

Configuration:
    - Edit LIVE_TRADING_CONFIG in config_main.py
    - Set API keys in .env file
    - Set model_result_dir to your trained model directory

Safety:
    - paper_trading=True by default (no real money)
    - Set paper_trading=False only after thorough testing
    - Kill switch triggers at max_drawdown_pct
    - Stop-loss per position at stop_loss_pct
    - WebSocket price monitor for real-time stop-loss between intervals
    - Panic button: Ctrl+C twice to emergency close all positions
"""

import os
import sys
import time
import signal
import pickle
import logging
import numpy as np
import torch

from config_main import LIVE_TRADING_CONFIG, CRYPTO_LIMITS
from exchange_manager import ExchangeManager
from data_feed import LiveDataFeed
from environment_live import LiveTradingEnv
from trade_logger import TradeLogger
from price_monitor import PriceMonitor
from alerts import AlertManager
from drl_agents.elegantrl_models import DRLAgent as DRLAgent_erl, MODELS
from train.config import Arguments
from train.run import init_agent

# ---- Logging Setup ----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('live_trading.log'),
    ]
)
logger = logging.getLogger('LiveTrader')


# ---- Globals for graceful shutdown ----
_shutdown_count = 0
_shutdown_requested = False
_exchange_ref = None
_ticker_list_ref = None


def signal_handler(signum, frame):
    global _shutdown_count, _shutdown_requested
    _shutdown_count += 1
    if _shutdown_count >= 2 and _exchange_ref and _ticker_list_ref:
        logger.critical("DOUBLE Ctrl+C — PANIC MODE: emergency closing all positions!")
        try:
            _exchange_ref.activate_panic_mode(_ticker_list_ref)
        except Exception as e:
            logger.critical(f"Panic mode failed: {e}")
        _shutdown_requested = True
    else:
        logger.info("Shutdown signal received (press Ctrl+C again for emergency close)...")
        _shutdown_requested = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def load_trained_model(result_dir):
    """Load trained model parameters from a result directory.

    Returns:
        env_params, net_dim, model_name, cwd (path to stored agent)
    """
    best_trial_path = os.path.join('./train_results', result_dir, 'best_trial')
    if not os.path.exists(best_trial_path):
        raise FileNotFoundError(f"No best_trial file found at {best_trial_path}")

    with open(best_trial_path, 'rb') as handle:
        best_trial = pickle.load(handle)

    model_name = best_trial.user_attrs['model_name']
    net_dim = best_trial.params['net_dimension']

    env_params = {
        "lookback": best_trial.params['lookback'],
        "norm_cash": best_trial.params['norm_cash'],
        "norm_stocks": best_trial.params['norm_stocks'],
        "norm_tech": best_trial.params['norm_tech'],
        "norm_reward": best_trial.params['norm_reward'],
        "norm_action": best_trial.params['norm_action'],
    }

    cwd = os.path.join('./train_results', result_dir, 'stored_agent')
    if not os.path.exists(cwd):
        raise FileNotFoundError(f"No stored_agent directory found at {cwd}")

    logger.info(f"Loaded model: {model_name}, net_dim={net_dim}")
    logger.info(f"Agent path: {cwd}")
    logger.info(f"Env params: {env_params}")

    return env_params, net_dim, model_name, cwd


def init_drl_agent(model_name, cwd, net_dim, state_dim, action_dim, gpu_id=0):
    """Initialize the DRL agent for inference.

    Returns:
        act: the actor network for inference
        device: torch device
    """
    agent_class = MODELS.get(model_name)
    if agent_class is None:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    # Create a minimal environment-like object for Arguments
    class _MinimalEnv:
        pass

    env_stub = _MinimalEnv()
    env_stub.env_num = 1
    env_stub.max_step = 1000
    env_stub.env_name = 'LiveTrading'
    env_stub.state_dim = state_dim
    env_stub.action_dim = action_dim
    env_stub.if_discrete = False
    env_stub.target_return = 1e8

    args = Arguments(agent=agent_class, env=env_stub)
    args.cwd = cwd
    args.net_dim = net_dim

    agent = init_agent(args, gpu_id=gpu_id)
    act = agent.act
    device = agent.device

    logger.info(f"DRL agent initialized on device: {device}")
    return act, device


def preflight_checks(config, exchange, data_feed, cwd):
    """Run safety checks before starting live trading."""
    logger.info("Running pre-flight checks...")

    # 1. Check model files exist
    actor_path = os.path.join(cwd, 'actor.pth')
    if not os.path.exists(actor_path):
        pth_files = [f for f in os.listdir(cwd) if f.endswith('.pth')]
        if not pth_files:
            raise FileNotFoundError(f"No .pth model files found in {cwd}")
        logger.info(f"Found model files: {pth_files}")
    else:
        logger.info(f"Model file found: {actor_path}")

    # 2. Check exchange connectivity + prices
    try:
        prices = exchange.get_current_prices(config['ticker_list'])
        for symbol, price in prices.items():
            logger.info(f"  {symbol}: ${price:.2f}")
    except Exception as e:
        raise ConnectionError(f"Cannot fetch prices: {e}")

    # 3. Check balance
    if config['paper_trading']:
        logger.info(f"Paper trading mode — {config['initial_capital']} USDT")
    else:
        balance = exchange.get_usdt_balance()
        logger.info(f"Live USDT balance: {balance:.2f}")
        if balance < 10:
            raise ValueError(f"Insufficient balance: {balance} USDT")

    # 4. Check data feed
    try:
        price_array, tech_array, _ = data_feed.fetch_latest_state()
        logger.info(f"Data feed OK: price_array={price_array.shape}, tech_array={tech_array.shape}")
    except Exception as e:
        raise RuntimeError(f"Data feed check failed: {e}")

    # 5. Check ticker symbols exist on exchange
    for symbol in config['ticker_list']:
        if symbol not in exchange.exchange.markets:
            raise ValueError(f"Symbol {symbol} not found on {exchange.exchange_name}")

    # 6. Check margin (futures only)
    if config['market_type'] == 'futures' and config.get('leverage', 1) > 1:
        margin_info = exchange.get_margin_info()
        logger.info(f"Margin: total={margin_info['total_margin']:.2f}, "
                     f"free={margin_info['free_margin']:.2f}")

    logger.info("All pre-flight checks passed!\n")


def print_config(config):
    """Print trading configuration."""
    print("\n" + "=" * 60)
    print("  FINRL CRYPTO — LIVE TRADING BOT")
    print("=" * 60)
    print(f"  Exchange:        {config['exchange']}")
    print(f"  Market:          {config['market_type']}")
    print(f"  Timeframe:       {config['timeframe']}")
    print(f"  Tickers:         {config['ticker_list']}")
    print(f"  Capital:         {config['initial_capital']} USDT")
    print(f"  Leverage:        {config.get('leverage', 1)}x")
    print(f"  Paper Trading:   {config['paper_trading']}")
    print(f"  Max Position:    {config['max_position_pct']*100:.0f}%")
    print(f"  Stop Loss:       {config['stop_loss_pct']*100:.0f}%")
    print(f"  Max Drawdown:    {config['max_drawdown_pct']*100:.0f}%")
    print(f"  Daily Loss Limit:{config.get('daily_loss_limit_pct', 0.10)*100:.0f}%")
    print(f"  Loop Interval:   {config['loop_interval_sec']}s")
    print(f"  WebSocket:       {config.get('enable_websocket', False)}")
    print(f"  Model Dir:       {config['model_result_dir']}")
    print("=" * 60)

    if not config['paper_trading']:
        print("\n  WARNING: REAL MONEY MODE — TRADES WILL USE REAL FUNDS\n")
    else:
        print("\n  Paper trading mode — no real money at risk\n")


def main():
    global _shutdown_requested, _exchange_ref, _ticker_list_ref

    config = LIVE_TRADING_CONFIG

    # Validate config
    if not config.get('model_result_dir'):
        print("ERROR: Set 'model_result_dir' in LIVE_TRADING_CONFIG (config_main.py)")
        print("Example: 'res_2025-12-07__01_28_01_model_CPCV_ppo_5m_50H_2k'")
        sys.exit(1)

    print_config(config)

    # ---- Load trained model ----
    env_params, net_dim, model_name, cwd = load_trained_model(config['model_result_dir'])

    # ---- Initialize exchange ----
    exchange = ExchangeManager(
        exchange_name=config['exchange'],
        paper_trading=config['paper_trading'],
        market_type=config['market_type'],
        leverage=config.get('leverage', 1),
        slippage_pct=config.get('slippage_pct', 0.0005),
    )
    _exchange_ref = exchange
    _ticker_list_ref = config['ticker_list']

    if config['paper_trading']:
        exchange.init_paper_balance(config['initial_capital'])

    # Set leverage for futures
    if config['market_type'] == 'futures' and config.get('leverage', 1) > 1:
        for symbol in config['ticker_list']:
            exchange.set_leverage(symbol, config['leverage'])

    # ---- Initialize data feed ----
    data_feed = LiveDataFeed(
        exchange_manager=exchange,
        ticker_list=config['ticker_list'],
        timeframe=config['timeframe'],
        lookback=config['lookback_candles'],
    )

    # ---- Initialize live environment ----
    live_env = LiveTradingEnv(
        exchange_manager=exchange,
        data_feed=data_feed,
        config=config,
        env_params=env_params,
    )

    # ---- Initialize trade logger ----
    trade_logger = TradeLogger()

    # ---- Initialize alerts ----
    alert_mgr = AlertManager(
        telegram_bot_token=config.get('telegram_bot_token', ''),
        telegram_chat_id=config.get('telegram_chat_id', ''),
        discord_webhook_url=config.get('discord_webhook_url', ''),
    )

    # ---- Pre-flight checks ----
    preflight_checks(config, exchange, data_feed, cwd)

    # ---- Compute state/action dimensions from first data fetch ----
    state, cash, stocks, current_prices, portfolio_value = live_env.sync_state()
    state_dim = len(state)
    action_dim = len(config['ticker_list'])

    logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")

    # ---- Initialize DRL agent ----
    act, device = init_drl_agent(model_name, cwd, net_dim, state_dim, action_dim, gpu_id=0)

    # ---- Initialize WebSocket price monitor ----
    price_monitor = None
    if config.get('enable_websocket', False):
        def on_ws_stop_loss(symbol, price):
            """Callback: immediately sell position when stop-loss triggers between cycles."""
            logger.warning(f"WS STOP-LOSS: selling all {symbol} @ ~{price:.2f}")
            pos = exchange.get_positions([symbol]).get(symbol, 0)
            if pos > 0:
                order = exchange.place_market_sell(symbol, pos)
                if order:
                    trade_logger.log_trade(order,
                                           entry_price=live_env.risk_manager.entry_prices.get(symbol))
                    live_env.risk_manager.record_exit(symbol)
                    alert_mgr.alert_stop_loss(
                        symbol,
                        live_env.risk_manager.entry_prices.get(symbol, 0),
                        price,
                        config['stop_loss_pct']
                    )

        def on_ws_panic():
            """Callback: kill switch triggered by price monitor."""
            logger.critical("WS PANIC: kill switch triggered between cycles")
            exchange.activate_panic_mode(config['ticker_list'])
            alert_mgr.alert_panic()

        price_monitor = PriceMonitor(
            exchange_manager=exchange,
            risk_manager=live_env.risk_manager,
            ticker_list=config['ticker_list'],
            check_interval_sec=config.get('ws_check_interval_sec', 5),
            on_stop_loss=on_ws_stop_loss,
            on_panic=on_ws_panic,
        )
        price_monitor.start()

    # ---- Main trading loop ----
    logger.info("Starting live trading loop...\n")
    cycle_count = 0

    try:
        while not _shutdown_requested:
            cycle_count += 1
            cycle_start = time.time()

            try:
                # 1. Sync state from exchange + market data
                state, cash, stocks, current_prices, portfolio_value = live_env.sync_state()

                logger.info(
                    f"[Cycle {cycle_count}] Portfolio: ${portfolio_value:.2f} | "
                    f"Cash: ${cash:.2f} | Positions: {dict(zip(config['ticker_list'], stocks))}"
                )

                # 2. Check kill switch / panic
                if live_env.is_kill_switch_active() or exchange.panic_mode:
                    logger.critical("Kill switch active. Closing all positions and stopping.")
                    sell_all_actions = -np.ones(action_dim) * 1e6
                    orders = live_env.process_actions(
                        sell_all_actions, cash, stocks, current_prices, portfolio_value
                    )
                    trade_logger.log_trades(orders,
                                            entry_prices=live_env.risk_manager.entry_prices)
                    alert_mgr.alert_kill_switch(
                        portfolio_value,
                        live_env.risk_manager.peak_portfolio_value,
                        config['max_drawdown_pct']
                    )
                    break

                # 3. Check margin safety (futures)
                if config['market_type'] == 'futures':
                    margin_info = exchange.get_margin_info()
                    max_ratio = config.get('max_margin_ratio', 0.80)
                    if margin_info['margin_ratio'] > max_ratio:
                        logger.warning(
                            f"Margin ratio {margin_info['margin_ratio']:.1%} > {max_ratio:.1%}. "
                            f"Reducing positions."
                        )

                # 4. Run DRL model inference
                with torch.no_grad():
                    s_tensor = torch.as_tensor(state, device=device).unsqueeze(0)
                    a_tensor = act(s_tensor)
                    raw_actions = a_tensor.detach().cpu().numpy()[0]

                # 5. Process actions through risk management and execute
                orders = live_env.process_actions(
                    raw_actions, cash, stocks, current_prices, portfolio_value
                )

                # 6. Log everything
                entry_prices = live_env.risk_manager.entry_prices.copy()
                trade_logger.log_trades(orders, entry_prices=entry_prices)
                trade_logger.log_portfolio(
                    cash, portfolio_value,
                    live_env.risk_manager.peak_portfolio_value,
                    dict(zip(config['ticker_list'], stocks))
                )
                trade_logger.log_decision(
                    raw_actions,
                    {s: a for s, a in zip(config['ticker_list'], raw_actions)},
                    current_prices,
                    portfolio_value,
                )
                trade_logger.log_metrics_snapshot()

                # 7. Send alerts for executed trades
                if orders:
                    for o in orders:
                        logger.info(
                            f"  Executed: {o.get('side','?')} "
                            f"{o.get('amount',0):.6f} {o.get('symbol','?')} "
                            f"@ {o.get('price',0):.2f} "
                            f"(slip={o.get('slippage',0):.4f}, fee={o.get('fee',0):.4f})"
                        )
                        alert_mgr.alert_trade(
                            o.get('side', ''), o.get('symbol', ''),
                            o.get('amount', 0), o.get('price', 0)
                        )
                else:
                    logger.info("  No trades this cycle.")

                # 8. Periodic metrics display
                if cycle_count % 12 == 0:  # every ~1 hour at 5m intervals
                    m = trade_logger.metrics.get_summary_dict()
                    logger.info(
                        f"  METRICS: trades={m['total_trades']} win={m['win_rate']:.0%} "
                        f"pnl=${m['total_pnl']:.2f} sharpe={m['sharpe_ratio']:.2f} "
                        f"maxDD={m['max_drawdown']:.1%}"
                    )

            except Exception as e:
                logger.error(f"Error in trading cycle {cycle_count}: {e}", exc_info=True)

            # 9. Sleep until next interval
            elapsed = time.time() - cycle_start
            sleep_time = max(0, config['loop_interval_sec'] - elapsed)
            if sleep_time > 0 and not _shutdown_requested:
                logger.info(f"  Sleeping {sleep_time:.0f}s until next cycle...\n")
                for _ in range(int(sleep_time)):
                    if _shutdown_requested:
                        break
                    time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        logger.info("Shutting down...")

        # Stop price monitor
        if price_monitor:
            price_monitor.stop()

        # Final metrics and summary
        trade_logger.log_metrics_snapshot()
        trade_logger.print_summary()

        # Send daily summary alert
        m = trade_logger.metrics.get_summary_dict()
        alert_mgr.alert_daily_summary(
            m['current_value'], m['daily_pnl'], m['daily_pnl_pct'],
            m['total_trades'], m['win_rate']
        )

        exchange.close()
        logger.info("Live trading bot stopped.")


if __name__ == "__main__":
    main()
