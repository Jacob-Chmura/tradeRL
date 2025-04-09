import logging
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from trade_rl.util.args import FeatureArgs


class Data:
    def __init__(self, feature_args: FeatureArgs, train: bool = True) -> None:
        if train:
            data_path = feature_args.train_data_path
        else:
            data_path = feature_args.test_data_path

        logging.debug(f'Reading raw data from: {data_path}')
        data = pd.read_parquet(data_path)
        logging.debug(f'Read {data.memory_usage(deep=True).sum() / 1e9} GB')

        self.day_to_data = preprocess_data(data, feature_args)
        self.unique_days = list(self.day_to_data.keys())

    def get_random_day_of_data(self) -> Tuple[str, pd.DataFrame]:
        date = random.choice(self.unique_days)
        return date, self.day_to_data[date]


def preprocess_data(
    data: pd.DataFrame, feature_args: FeatureArgs
) -> Dict[str, pd.DataFrame]:
    short_window = feature_args.short_window
    mid_window = feature_args.medium_window
    long_window = feature_args.long_window
    ochl_cols = ['open', 'high', 'low', 'close']

    def fill_missing_data(df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index('market_second')
        df = df.reindex(pd.RangeIndex(23400))
        df[ochl_cols] = df[ochl_cols].ffill()
        df['volume'] = df['volume'].fillna(0)
        df = df[ochl_cols + ['volume']].reset_index(names=['market_second']).dropna()
        return df

    def compute_market_features(df: pd.DataFrame) -> pd.DataFrame:
        df['log_return'] = np.log(df['open'] / df['open'].shift(1)).fillna(0)
        df['volatility'] = df['log_return'].rolling(window=mid_window).std().fillna(0)

        # Moving Averages
        df['sma_return_short'] = df['log_return'].rolling(window=short_window).mean()
        df['sma_return_long'] = df['log_return'].rolling(window=long_window).mean()
        df['ema_return_short'] = df['log_return'].ewm(span=short_window).mean()
        df['ema_return_long'] = df['log_return'].ewm(span=long_window).mean()

        # MACD (Moving Average Convergence Divergence)
        df['macd'] = df['ema_return_short'] - df['ema_return_long']
        df['signal'] = df['macd'].ewm(span=mid_window).mean()

        # RSI (Relative strength index)
        delta = df['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=long_window).mean()
        avg_loss = pd.Series(loss).rolling(window=long_window).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger bands percentage
        sma_20 = df['open'].rolling(long_window).mean()
        std_20 = df['open'].rolling(long_window).std()
        upper = sma_20 + 2 * std_20
        lower = sma_20 - 2 * std_20
        df['bollinger_percentage'] = (df['open'] - lower) / (upper - lower + 1e-9)

        # %K of stochastic oscillator
        lowest = df['low'].rolling(long_window).min()
        highest = df['high'].rolling(long_window).max()
        df['stoch_k'] = (df['open'] - lowest) / (highest - lowest + 1e-9)

        # VWAP (Volume Weighted Average Price)
        df['vwap'] = (df['volume'] * df['open']).cumsum() / df['volume'].cumsum()

        # Volume moving average
        df['volume_sma'] = df['volume'].rolling(window=long_window).mean()
        return df

    day_to_data = {}  # TODO: Map symbol as well
    for date, df in data.groupby('date'):
        df = fill_missing_data(df)
        df = compute_market_features(df)
        day_to_data[date] = df
    return day_to_data
