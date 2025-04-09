from typing import List

import numpy as np
import pandas as pd

from trade_rl.order import Order

# TODO: Move to config
SHORT_WINDOW = 5
MEDIUM_WINDOW = 10
LONG_WINDOW = 30


def fill_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    symbol = df['symbol'].iloc[0]
    # Combine 'data' and 'time' columns into a single datetime column
    df['datetime'] = pd.to_datetime(
        df['date'].astype(str) + ' ' + df['time'].astype(str)
    )
    df.set_index('datetime', inplace=True)

    # Only keep necessary columns
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    df = df[ohlcv_cols]

    # Get unique dates
    all_dates = df.index.normalize().unique()

    filled_dfs = []

    for date in all_dates:
        # Create a full date range of seconds for the trading day
        start = pd.Timestamp(date) + pd.Timedelta(hours=9, minutes=30)
        end = pd.Timestamp(date) + pd.Timedelta(hours=16)
        full_range = pd.date_range(start=start, end=end, freq='s')

        # Filter the DataFrame for the current trading day
        df_day = df.loc[(df.index >= start) & (df.index <= end)]

        # Reindex to full 1-second range
        df_day_filled = df_day.reindex(full_range)

        # Forward fill OHLC values and fill volume with 0
        df_day_filled[['open', 'high', 'low', 'close']] = df_day_filled[
            ['open', 'high', 'low', 'close']
        ].ffill()
        df_day_filled['volume'] = df_day_filled['volume'].fillna(0)
        df_day_filled['market_seconds'] = (
            (df_day_filled.index - start).total_seconds().astype(int)
        )

        filled_dfs.append(df_day_filled)

    # Concatenate all the filled daily data
    df_filled = pd.concat(filled_dfs)

    df_filled = df_filled.reset_index().rename(columns={'index': 'datetime'})
    df_filled['date'] = df_filled['datetime'].dt.date
    df_filled['time'] = df_filled['datetime'].dt.time
    df_filled['symbol'] = symbol
    df_filled.drop(columns=['datetime'], inplace=True)
    df_filled = df_filled[
        [
            'date',
            'time',
            'market_seconds',
            'symbol',
            'open',
            'high',
            'low',
            'close',
            'volume',
        ]
    ]
    df_filled.dropna(inplace=True)
    df_filled.reset_index(drop=True, inplace=True)

    return df_filled


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df = fill_missing_data(df)

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
