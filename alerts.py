"""
Alert system for live trading notifications.
Supports Telegram and Discord webhooks.
Sends alerts for stop-loss triggers, kill switch, new signals, and portfolio updates.
"""

import logging
import requests
from datetime import datetime

logger = logging.getLogger(__name__)


class AlertManager:
    """Send trading alerts via Telegram and/or Discord."""

    def __init__(self, telegram_bot_token='', telegram_chat_id='',
                 discord_webhook_url=''):
        self.telegram_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self.discord_url = discord_webhook_url

        self.telegram_enabled = bool(telegram_bot_token and telegram_chat_id)
        self.discord_enabled = bool(discord_webhook_url)

        if self.telegram_enabled:
            logger.info("Telegram alerts enabled")
        if self.discord_enabled:
            logger.info("Discord alerts enabled")
        if not self.telegram_enabled and not self.discord_enabled:
            logger.info("No alert channels configured (optional)")

    def send(self, message, level='info'):
        """Send alert to all configured channels.
        Args:
            message: alert text
            level: 'info', 'warning', 'critical'
        """
        prefix = {
            'info': 'INFO',
            'warning': 'WARNING',
            'critical': 'CRITICAL',
        }.get(level, 'INFO')

        timestamp = datetime.now().strftime('%H:%M:%S')
        full_msg = f"[{prefix}] {timestamp} | {message}"

        if self.telegram_enabled:
            self._send_telegram(full_msg)
        if self.discord_enabled:
            self._send_discord(full_msg)

    def alert_stop_loss(self, symbol, entry_price, current_price, loss_pct):
        self.send(
            f"STOP LOSS: {symbol}\n"
            f"Entry: ${entry_price:.2f} -> Current: ${current_price:.2f}\n"
            f"Loss: {loss_pct:.1%}",
            level='warning'
        )

    def alert_kill_switch(self, portfolio_value, peak_value, drawdown_pct):
        self.send(
            f"KILL SWITCH ACTIVATED\n"
            f"Portfolio: ${portfolio_value:.2f} (peak: ${peak_value:.2f})\n"
            f"Drawdown: {drawdown_pct:.1%}",
            level='critical'
        )

    def alert_trade(self, side, symbol, amount, price):
        self.send(
            f"TRADE: {side.upper()} {amount:.6f} {symbol} @ ${price:.2f}",
            level='info'
        )

    def alert_daily_summary(self, portfolio_value, daily_pnl, daily_pnl_pct,
                            trade_count, win_rate):
        self.send(
            f"DAILY SUMMARY\n"
            f"Portfolio: ${portfolio_value:.2f}\n"
            f"Daily P&L: ${daily_pnl:.2f} ({daily_pnl_pct:.1%})\n"
            f"Trades: {trade_count} | Win rate: {win_rate:.0%}",
            level='info'
        )

    def alert_panic(self):
        self.send("PANIC MODE ACTIVATED — All positions being closed!", level='critical')

    def _send_telegram(self, message):
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML',
            }
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code != 200:
                logger.warning(f"Telegram alert failed: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.warning(f"Telegram alert error: {e}")

    def _send_discord(self, message):
        try:
            payload = {'content': message}
            resp = requests.post(self.discord_url, json=payload, timeout=10)
            if resp.status_code not in (200, 204):
                logger.warning(f"Discord alert failed: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.warning(f"Discord alert error: {e}")
