import glob
import logging
import os
import random
from typing import Tuple

import pandas as pd


class Data:
    def __init__(self, data_path: str) -> None:
        file_pattern = os.path.join(data_path, '*.parquet')
        files = glob.glob(file_pattern)
        if not files:
            raise ValueError('Data not found')

        logging.debug(f'Reading raw data files: {files}')
        self.data = pd.concat([pd.read_parquet(file) for file in files])
        logging.debug(
            f'Read {self.data.memory_usage(deep=True).sum() / 1e9} GB ({len(self.data)} rows)'
        )
        self.data['ts_event'] = pd.to_datetime(self.data['ts_event'])
        self.data['date'] = self.data['ts_event'].dt.date
        self.unique_days = self.data['date'].unique()

    def get_order_data(
        self, start_time: int, end_time: int
    ) -> Tuple[pd.DataFrame, int, int]:
        chosen_date = random.choice(self.unique_days)

        day_data = self.data[self.data['date'] == chosen_date].copy()
        day_data.reset_index(drop=True, inplace=True)
        available_rows = len(day_data)

        if not 0 <= start_time < available_rows:
            raise ValueError(
                f'Order starts at {start_time} but only {available_rows} data samples available for date: {chosen_date}'
            )
        if start_time + end_time > available_rows:
            end_time = available_rows - start_time

        data_chunk = day_data.iloc[: start_time + end_time]

        return data_chunk, start_time - 1, end_time
