from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
import torch

from trade_rl.env import TradingEnvironment
from trade_rl.order import Order


def computeFeatureVector(env: 'TradingEnvironment') -> torch.Tensor:
    current_index = env.start_index + env.episode_step
    current_price = env.order_data['open'][current_index]
    all_current_day_prices = env.order_data['open'][: current_index + 1]
    previous_price = env.order_data['open'][current_index - 1]
    episode_first_price = env.order_data['open'][env.start_index]
    day_first_price = env.order_data['open'][0]

    current_time = env.order_data['ts_event'][current_index]

    current_volume = env.order_data['volume'][current_index]
    all_current_day_volume = np.sum(env.order_data['volume'][: current_index + 1])

    return torch.tensor(
        [
            current_price,
            get_vleft_norm(env.remaining_qty, env.order),
            get_tleft_norm(env.episode_step, env.order),
            get_return(previous_price, current_price),
            get_return(day_first_price, current_price),
            get_return(episode_first_price, current_price),
            get_max_price_day(all_current_day_prices),
            get_min_price_day(all_current_day_prices),
            get_elapsed_time_percentage(current_time),
            get_vwap(env.portfolio),
            get_log_volume(current_volume),
            get_log_volume(all_current_day_volume),
        ]
    )


# Incremental features
def update_ma(prev_average: float, new_value: float, n: int) -> float:
    return (prev_average * (n - 1) + new_value) / n


def update_ema(prev_average: float, new_value: float, alpha: float) -> float:
    return alpha * new_value + (1 - alpha) * prev_average


# Agent features
def get_vleft_norm(remaining_qty: float, order: Order) -> float:
    return remaining_qty / order.qty


def get_tleft_norm(current_step: int, order: Order) -> float:
    return (order.end_time - current_step) / order.end_time


def get_vwap(portfolio: List) -> float:
    return np.mean(np.array(portfolio)[:, 0])


# Market features
def get_return(previous_price: float, current_price: float) -> float:
    return (current_price - previous_price) / previous_price


def linear_schedule(start: float, end: float, duration: float, t: int) -> float:
    slope = (end - start) / duration
    return max(slope * t + start, end)


def get_elapsed_time_percentage(timestamp: datetime) -> float:
    trading_start = timestamp.replace(hour=8, minute=0, second=0)
    trading_end = timestamp.replace(hour=23, minute=59, second=59)

    total_trading_duration = trading_end - trading_start

    elapsed_time = max(timedelta(0), timestamp - trading_start)

    percentage = (elapsed_time / total_trading_duration) * 100
    return percentage


def get_max_price_day(day_prices: pd.Series) -> float:
    return max(day_prices)


def get_min_price_day(day_prices: pd.Series) -> float:
    return min(day_prices)


def get_log_volume(volume: int) -> float:
    return np.log(volume + 1)
