from datetime import datetime, timedelta

import torch

from trade_rl.env import TradingEnvironment
from trade_rl.order import Order


def computeFeatureVector(env: 'TradingEnvironment') -> torch.Tensor:
    # Placeholder for actual feature extraction logic.
    # This should be replaced with the actual logic to compute all features from the environment state.
    return torch.tensor([])


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


def getVolumeWeightedAveragePrice() -> float:
    # Placeholder for actual VWAP calculation logic
    # This should be replaced with the actual logic to compute VWAP from the order book data
    return 0.0


# Market features
def getPriceReturn(previous_price: float, current_price: float) -> float:
    return (current_price - previous_price) / previous_price


def linear_schedule(start: float, end: float, duration: float, t: int) -> float:
    slope = (end - start) / duration
    return max(slope * t + start, end)


def getTradingDayPercentage(timestamp_str: str) -> float:
    cleaned_timestamp = timestamp_str.split('.')[0] + 'Z'

    timestamp = datetime.strptime(cleaned_timestamp, '%Y-%m-%dT%H:%M:%SZ')

    trading_start = timestamp.replace(hour=8, minute=0, second=0)
    trading_end = timestamp.replace(hour=23, minute=59, second=59)

    total_trading_duration = trading_end - trading_start

    elapsed_time = max(timedelta(0), timestamp - trading_start)

    percentage = (elapsed_time / total_trading_duration) * 100
    return percentage
