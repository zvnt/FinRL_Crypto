'''
Exchange API configuration.
Reads API keys from environment variables for security.
Supports: Binance, Bitget (requires passphrase)
'''

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Binance
API_KEY_BINANCE = os.getenv('API_KEY_BINANCE', '')
API_SECRET_BINANCE = os.getenv('API_SECRET_BINANCE', '')

# Bitget (requires passphrase for API access)
API_KEY_BITGET = os.getenv('API_KEY_BITGET', '')
API_SECRET_BITGET = os.getenv('API_SECRET_BITGET', '')
API_PASSPHRASE_BITGET = os.getenv('API_PASSPHRASE_BITGET', '')