from typing import List

import numpy as np
import pandas as pd
import torch

from trade_rl.env import TradingEnvironment
from trade_rl.order import Order

# TODO: Move to config
SHORT_WINDOW = 5
MEDIUM_WINDOW = 10
LONG_WINDOW = 30


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    df['log_return'] = np.log(df['open'] / df['open'].shift(1))
    df['log_return'] = df['log_return'].fillna(0)

    # Long and short moving averages
    df['sma_return_short'] = df['log_return'].rolling(window=SHORT_WINDOW).mean()
    df['sma_return_long'] = df['log_return'].rolling(window=LONG_WINDOW).mean()

    # Long and short exponential moving averages
    df['ema_return_short'] = df['log_return'].ewm(span=SHORT_WINDOW).mean()
    df['ema_return_long'] = df['log_return'].ewm(span=LONG_WINDOW).mean()

    # MACD (Moving Average Convergence Divergence)
    df['macd'] = df['ema_return_short'] - df['ema_return_long']
    df['signal'] = df['macd'].ewm(span=MEDIUM_WINDOW).mean()

    # Volatility
    df['volatility'] = df['log_return'].rolling(window=MEDIUM_WINDOW).std()
    df['volatility'] = df['volatility'].fillna(0)

    # Relative strength index (RSI)
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=LONG_WINDOW).mean()
    avg_loss = pd.Series(loss).rolling(window=LONG_WINDOW).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger bands percentage
    sma_20 = df['open'].rolling(LONG_WINDOW).mean()
    std_20 = df['open'].rolling(LONG_WINDOW).std()
    upper = sma_20 + 2 * std_20
    lower = sma_20 - 2 * std_20
    df['bollinger_percentage'] = (df['open'] - lower) / (upper - lower + 1e-9)

    # %K of stochastic oscillator
    lowest = df['low'].rolling(LONG_WINDOW).min()
    highest = df['high'].rolling(LONG_WINDOW).max()
    df['stoch_k'] = (df['open'] - lowest) / (highest - lowest + 1e-9)

    # VWAP (Volume Weighted Average Price)
    df['vwap'] = (df['volume'] * df['open']).cumsum() / df['volume'].cumsum()

    # Volume moving average
    df['volume_sma'] = df['volume'].rolling(window=LONG_WINDOW).mean()

    return df


def computeFeatureVector(env: 'TradingEnvironment') -> torch.Tensor:
    # Price features
    current_index = env.start_index + env.episode_step
    current_price = env.order_data['open'][current_index]
    all_current_day_prices = env.order_data['open'][: current_index + 1]
    max_day_price = max(all_current_day_prices)
    min_day_price = min(all_current_day_prices)
    previous_price = env.order_data['open'][current_index - 1]
    episode_first_price = env.order_data['open'][env.start_index]
    day_first_price = env.order_data['open'][0]
    current_market_vwap = env.order_data['vwap'][current_index - 1]

    # Time features
    market_seconds = env.order_data['market_second'][current_index]

    # Volume features
    prev_volume = env.order_data['volume'][current_index - 1]
    volume_sma = env.order_data['volume_sma'][current_index - 1]

    return torch.tensor(
        [
            get_vleft_norm(env.remaining_qty, env.order),
            get_tleft_norm(env.episode_step, env.order),
            get_return(previous_price, current_price),
            get_return(day_first_price, current_price),
            get_return(episode_first_price, current_price),
            get_return(max_day_price, current_price),
            get_return(min_day_price, current_price),
            get_elapsed_time_percentage(market_seconds),
            get_vwap_norm(env.portfolio, current_market_vwap),
            get_volume_norm(prev_volume, volume_sma),
            env.order_data['sma_return_short'][current_index],
            env.order_data['sma_return_long'][current_index],
            env.order_data['ema_return_short'][current_index],
            env.order_data['ema_return_long'][current_index],
            env.order_data['macd'][current_index],
            env.order_data['signal'][current_index],
            env.order_data['volatility'][current_index],
            env.order_data['rsi'][current_index],
            env.order_data['bollinger_percentage'][current_index],
            env.order_data['stoch_k'][current_index],
        ]
    )


# Agent features
def get_vleft_norm(remaining_qty: float, order: Order) -> float:
    return remaining_qty / order.qty


def get_tleft_norm(current_step: int, order: Order) -> float:
    return (order.end_time - current_step) / order.end_time


def get_vwap_norm(portfolio: List, market_vwap: int) -> float:
    agent_vwap = np.mean(np.array(portfolio)[:, 0])
    return agent_vwap / market_vwap if market_vwap != 0 else 0


# Market features
def get_return(previous_price: float, current_price: float) -> float:
    return (current_price - previous_price) / previous_price


def linear_schedule(start: float, end: float, duration: float, t: int) -> float:
    slope = (end - start) / duration
    return max(slope * t + start, end)


def get_elapsed_time_percentage(market_seconds: int) -> float:
    return market_seconds / 23400.0


def get_volume_norm(volume: int, volume_sma: float) -> float:
    return volume / volume_sma if volume_sma != 0 else 0
