import logging
import random
from typing import Tuple

import pandas as pd


class Data:
    def __init__(self, data_path: str) -> None:
        logging.debug(f'Reading raw data from: {data_path}')
        self.data = pd.read_parquet(data_path)
        logging.debug(f'Read {self.data.memory_usage(deep=True).sum() / 1e9} GB')
        self.unique_days = self.data['date'].unique()

    def get_order_data(
        self, start_time: int, max_steps: int
    ) -> Tuple[pd.DataFrame, int, int]:
        date = random.choice(self.unique_days)
        max_end_time = start_time + max_steps
        mask = (self.data.date == date) & (self.data.market_second <= max_end_time)
        data = self.data[mask].copy()

        order_start_index = len(data[data.market_second < start_time])
        max_steps = len(data) - order_start_index - 1  # TODO: 'Fill in' with data
        return data, order_start_index, max_steps
