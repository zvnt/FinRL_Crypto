"""
Live data feed that produces state arrays identical in format to the training pipeline.
Fetches latest candles via ExchangeManager, computes TALib indicators,
and returns price_array + tech_array matching processor_Binance.py output.
Uses concurrent fetching for multiple tickers to reduce latency.
"""

import numpy as np
import pandas as pd
import logging
import concurrent.futures
from talib import RSI, MACD, CCI, DX, ROC, ULTOSC, WILLR, OBV, HT_DCPHASE

logger = logging.getLogger(__name__)


class LiveDataFeed:
    def __init__(self, exchange_manager, ticker_list, timeframe='5m', lookback=100):
        """
        Args:
            exchange_manager: ExchangeManager instance
            ticker_list: list of symbols e.g. ['BTC/USDT', 'ETH/USDT']
            timeframe: candle timeframe e.g. '5m'
            lookback: number of candles to fetch (must be > indicator warmup period)
        """
        self.exchange = exchange_manager
        self.ticker_list = ticker_list
        self.timeframe = timeframe
        self.lookback = max(lookback, 60)  # minimum 60 for indicator warmup
        self.correlation_threshold = 0.9

    def _fetch_and_process_symbol(self, symbol):
        """Fetch candles and compute indicators for a single symbol.
        Designed to run in a thread pool for concurrent execution.
        """
        ohlcv = self.exchange.get_latest_candles(
            symbol, self.timeframe, self.lookback
        )
        df = self._ohlcv_to_dataframe(ohlcv, symbol)
        df = self._add_technical_indicators(df)
        return symbol, df

    def fetch_latest_state(self):
        """Fetch latest candles for all tickers concurrently, compute indicators,
        and return (price_array, tech_array) in training-compatible format.

        Returns:
            price_array: np.ndarray shape (T, num_tickers) — close prices
            tech_array: np.ndarray shape (T, num_tickers * num_features) — tech indicators
            current_prices: dict {symbol: price}
        """
        all_coin_dfs = []

        if len(self.ticker_list) > 1:
            # Concurrent fetching for multiple tickers
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(len(self.ticker_list), 4)
            ) as executor:
                futures = {
                    executor.submit(self._fetch_and_process_symbol, symbol): symbol
                    for symbol in self.ticker_list
                }
                results = {}
                for future in concurrent.futures.as_completed(futures):
                    symbol = futures[future]
                    try:
                        sym, df = future.result()
                        results[sym] = df
                    except Exception as e:
                        logger.error(f"Failed to fetch data for {symbol}: {e}")
                        raise

            # Maintain original ticker order
            for symbol in self.ticker_list:
                all_coin_dfs.append(results[symbol])
        else:
            # Single ticker — no need for thread pool
            for symbol in self.ticker_list:
                try:
                    _, df = self._fetch_and_process_symbol(symbol)
                    all_coin_dfs.append(df)
                except Exception as e:
                    logger.error(f"Failed to fetch data for {symbol}: {e}")
                    raise

        # Build arrays in same format as processor_Binance.df_to_array
        price_array, tech_array = self._build_arrays(all_coin_dfs)

        # Current prices
        current_prices = {}
        for i, symbol in enumerate(self.ticker_list):
            current_prices[symbol] = price_array[-1, i]

        return price_array, tech_array, current_prices

    def get_current_prices(self):
        """Get real-time prices for all tickers."""
        return self.exchange.get_current_prices(self.ticker_list)

    def _ohlcv_to_dataframe(self, ohlcv, symbol):
        """Convert CCXT OHLCV data to DataFrame matching processor_Binance format."""
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        df = df.astype(float)
        df['tic'] = symbol
        return df

    def _add_technical_indicators(self, df):
        """Compute TALib indicators — must match processor_Binance.get_TALib_features_for_each_coin exactly."""
        df['rsi'] = RSI(df['close'], timeperiod=14)
        df['macd'], _, _ = MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['cci'] = CCI(df['high'], df['low'], df['close'], timeperiod=14)
        df['dx'] = DX(df['high'], df['low'], df['close'], timeperiod=14)
        df['roc'] = ROC(df['close'], timeperiod=10)
        df['ultosc'] = ULTOSC(df['high'], df['low'], df['close'])
        df['willr'] = WILLR(df['high'], df['low'], df['close'])
        df['obv'] = OBV(df['close'], df['volume'])
        df['ht_dcphase'] = HT_DCPHASE(df['close'])

        # Drop NaN rows from indicator warmup
        df = df.dropna()
        return df

    def _drop_correlated_features(self, df):
        """Drop correlated features — must match processor_Binance.drop_correlated_features."""
        real_drop = ['high', 'low', 'open', 'macd', 'cci', 'roc', 'willr']
        real_drop = [col for col in real_drop if col in df.columns]
        df = df.drop(real_drop, axis=1)
        return df

    def _build_arrays(self, coin_dfs):
        """Build price_array and tech_array from list of per-coin DataFrames.
        Matches the output format of processor_Binance.df_to_array.
        """
        # Apply same feature dropping as training pipeline
        processed_dfs = []
        for df in coin_dfs:
            df_processed = self._drop_correlated_features(df.copy())
            processed_dfs.append(df_processed)

        # Find common time range (intersection of all coins)
        common_index = processed_dfs[0].index
        for df in processed_dfs[1:]:
            common_index = common_index.intersection(df.index)
        common_index = common_index.sort_values()

        if len(common_index) == 0:
            raise ValueError("No common timestamps across tickers after indicator computation")

        # Build tech indicator list (excluding 'tic')
        tech_cols = [c for c in processed_dfs[0].columns if c != 'tic']

        # Build arrays
        price_list = []
        tech_list = []

        for df in processed_dfs:
            df_aligned = df.loc[common_index]
            price_list.append(df_aligned[['close']].values)
            tech_list.append(df_aligned[tech_cols].values)

        price_array = np.hstack(price_list)  # shape (T, num_tickers)
        tech_array = np.hstack(tech_list)    # shape (T, num_tickers * num_features)

        # Replace NaN with 0 (same as processor_Binance.run)
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0

        logger.info(
            f"Live data: {len(common_index)} candles, "
            f"price_array={price_array.shape}, tech_array={tech_array.shape}"
        )

        return price_array, tech_array
