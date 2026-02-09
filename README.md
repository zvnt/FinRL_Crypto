# FinRL_Crypto: DRL-Based Cryptocurrency Trading Bot

Deep Reinforcement Learning agents for cryptocurrency trading with **live trading support** on Binance and Bitget.

Based on the research paper: [Addressing Overfitting on Cryptocurrency Trading with DRL](https://arxiv.org/abs/2209.05559)

## Features

- **Train** DRL agents (PPO, A2C, DDPG, TD3, SAC) on historical crypto data
- **Optimize** hyperparameters via Optuna (CPCV, KCV, Walk-Forward)
- **Backtest** trained agents with overfitting analysis (PBO)
- **Live trade** on Binance or Bitget with real or paper money
- **Risk management**: stop-loss, drawdown kill switch, position limits

## Requirements

- **Python 3.10+**
- **TA-Lib** (see install instructions below)
- Exchange API keys (Binance and/or Bitget)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**TA-Lib** must be installed separately:
```bash
# Windows: download wheel from https://github.com/cgohlke/talib-build/
# Linux:
sudo apt-get install libta-lib-dev && pip install TA-Lib
# macOS:
brew install ta-lib && pip install TA-Lib
```

### 2. Configure API Keys

Copy the example env file and fill in your keys:
```bash
cp .env.example .env
```

Edit `.env`:
```
# Binance
API_KEY_BINANCE=your_key
API_SECRET_BINANCE=your_secret

# Bitget (requires passphrase)
API_KEY_BITGET=your_key
API_SECRET_BITGET=your_secret
API_PASSPHRASE_BITGET=your_passphrase
```

### 3. Configure Trading

Edit `config_main.py`:
- **TICKER_LIST**: Cryptocurrencies to trade (e.g., `['BTCUSDT', 'ETHUSDT']`)
- **TIMEFRAME**: Candle interval (e.g., `'5m'`)
- **LIVE_TRADING_CONFIG**: Live trading parameters (exchange, capital, risk limits)

## Training Pipeline

### Step 0: Download Data
```bash
python 0_dl_trainval_data.py   # Training/validation data
python 0_dl_trade_data.py      # Backtest data
```

### Step 1: Optimize Hyperparameters
```bash
python 1_optimize_cpcv.py      # Combinatorial Purged Cross-Validation
python 1_optimize_kcv.py       # K-Fold Cross-Validation
python 1_optimize_wf.py        # Walk-Forward
```

### Step 2: Validate
```bash
python 2_validate.py
```

### Step 4: Backtest
```bash
python 4_backtest.py
```

### Step 5: PBO Analysis
```bash
python 5_pbo.py
```

## Live Trading

### Step 6: Run Live Trading Bot
```bash
python 6_live_trade.py
```

**Before running live:**
1. Set `model_result_dir` in `LIVE_TRADING_CONFIG` to your trained model directory
2. Start with `paper_trading: True` (default) to test without real money
3. Only set `paper_trading: False` after thorough paper testing

### Live Trading Config (`config_main.py`)
```python
LIVE_TRADING_CONFIG = {
    'exchange': 'binance',        # 'binance' or 'bitget'
    'market_type': 'spot',
    'timeframe': '5m',
    'ticker_list': ['BTC/USDT', 'ETH/USDT'],
    'initial_capital': 1000,
    'paper_trading': True,        # SET TO False FOR REAL TRADING
    'stop_loss_pct': 0.05,        # 5% per position
    'max_drawdown_pct': 0.15,     # 15% portfolio kill switch
    'max_position_pct': 0.25,     # 25% max per asset
    'model_result_dir': 'res_...',
}
```

### Supported Exchanges

| Exchange | API Key | Secret | Passphrase |
|----------|---------|--------|------------|
| Binance  | Yes     | Yes    | No         |
| Bitget   | Yes     | Yes    | **Yes**    |

### Risk Management

- **Stop-loss**: Auto-sells position if it drops X% from entry
- **Drawdown kill switch**: Stops all trading if portfolio drops X% from peak
- **Position limits**: Never allocates more than X% to one asset
- **Daily loss limit**: Stops buying if daily loss exceeds threshold

### Trade Logs

All trades are logged to `./live_trading_logs/`:
- `trades_*.csv` — every executed order
- `portfolio_*.csv` — portfolio snapshots each cycle
- `decisions_*.csv` — raw model actions vs filtered actions

## Project Structure

```
config_api.py              — API keys (env vars)
config_main.py             — Training + live trading config
exchange_manager.py        — Unified exchange interface (Binance/Bitget)
data_feed.py               — Live market data provider
environment_CCXT.py        — Backtesting environment
environment_live.py        — Live trading environment + risk management
trade_logger.py            — Trade logging
processor_Binance.py       — Historical data processor
6_live_trade.py            — Live trading entry point
drl_agents/                — DRL algorithms (PPO, A2C, DDPG, TD3, SAC)
train/                     — ElegantRL training infrastructure
```

## Supported Cryptocurrencies

BTC, ETH, NEAR, LINK, LTC, MATIC, UNI, SOL, AAVE, AVAX — and any pair available on your exchange.

## Important Warnings

1. **Never** commit API keys to version control
2. **Always** test with paper trading first
3. Cryptocurrency trading involves **significant financial risk**
4. Past backtest performance does **not** guarantee future results
5. Start with small capital amounts when going live