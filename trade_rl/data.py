import logging
import random
from typing import Tuple

import pandas as pd

from trade_rl.util.args import FeatureArgs
from trade_rl.util.feature_engineering import preprocess_data


class Data:
    def __init__(self, feature_args: FeatureArgs, train: bool = True) -> None:
        if train:
            data_path = feature_args.train_data_path
        else:
            data_path = feature_args.test_data_path

        logging.debug(f'Reading raw data from: {data_path}')
        self.data = pd.read_parquet(data_path)
        logging.debug(f'Read {self.data.memory_usage(deep=True).sum() / 1e9} GB')
        self.unique_days = self.data['date'].unique()
        self.data = preprocess_data(self.data, feature_args)

    def get_order_data(
        self, start_time: int, max_steps: int
    ) -> Tuple[pd.DataFrame, int, int]:
        date = random.choice(self.unique_days)
        max_end_time = start_time + max_steps
        mask = (self.data.date == date) & (self.data.market_seconds <= max_end_time)
        data = self.data[mask].copy()

        order_start_index = len(data[data.market_seconds < start_time])
        max_steps = len(data) - order_start_index - 1  # TODO: 'Fill in' with data
        return data, order_start_index, max_steps
