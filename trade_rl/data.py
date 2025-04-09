import logging
import random
from typing import List

import numpy as np
import pandas as pd

from trade_rl.order import Order
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
        self.unique_days = data['date'].unique()
        self.data = preprocess_data(data, feature_args)

    def get_random_day_of_data(self) -> pd.DataFrame:
        date = random.choice(self.unique_days)
        return self.data[self.data.date == date].reset_index(drop=True)


def preprocess_data(data: pd.DataFrame, feature_args: FeatureArgs) -> pd.DataFrame:
    short_window = feature_args.short_window
    mid_window = feature_args.medium_window
    long_window = feature_args.long_window

    df = fill_missing_data(data)
    df['log_return'] = np.log(df['open'] / df['open'].shift(1))
    df['log_return'] = df['log_return'].fillna(0)

    # Long and short moving averages
    df['sma_return_short'] = df['log_return'].rolling(window=short_window).mean()
    df['sma_return_long'] = df['log_return'].rolling(window=long_window).mean()

    # Long and short exponential moving averages
    df['ema_return_short'] = df['log_return'].ewm(span=short_window).mean()
    df['ema_return_long'] = df['log_return'].ewm(span=long_window).mean()

    # MACD (Moving Average Convergence Divergence)
    df['macd'] = df['ema_return_short'] - df['ema_return_long']
    df['signal'] = df['macd'].ewm(span=mid_window).mean()

    # Volatility
    df['volatility'] = df['log_return'].rolling(window=mid_window).std()
    df['volatility'] = df['volatility'].fillna(0)

    # Relative strength index (RSI)
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


def fill_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    filled_dfs = []
    for (date, sym), df_ in df.groupby(['date', 'symbol']):
        df_ = df_.set_index('market_second')
        df_ = df_.reindex(pd.RangeIndex(23400))

        ochl_cols = ['open', 'high', 'low', 'close']
        df_[ochl_cols] = df_[ochl_cols].ffill()
        df_['volume'] = df_['volume'].fillna(0)
        df_ = df_[ochl_cols + ['volume']]

        df_['date'] = date
        df_['symbol'] = sym
        df_ = df_.reset_index(names=['market_second'])
        filled_dfs.append(df_)
    df = pd.concat(filled_dfs).dropna()
    return df


def get_vleft_norm(remaining_qty: float, order: Order) -> float:
    return remaining_qty / order.qty


def get_tleft_norm(current_step: int, order: Order) -> float:
    return (order.duration - current_step) / order.duration


def get_vwap_norm(portfolio: List, market_vwap: int) -> float:
    agent_vwap = np.mean(np.array(portfolio)[:, 0])
    return agent_vwap / market_vwap if market_vwap != 0 else 0


def get_return(previous_price: float, current_price: float) -> float:
    return (current_price - previous_price) / previous_price


def linear_schedule(start: float, end: float, duration: float, t: int) -> float:
    slope = (end - start) / duration
    return max(slope * t + start, end)


def get_elapsed_time_percentage(market_seconds: int) -> float:
    return market_seconds / 23400.0


def get_volume_norm(volume: int, volume_sma: float) -> float:
    return volume / volume_sma if volume_sma != 0 else 0
