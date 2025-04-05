import glob
import os
import random

import pandas as pd


class Data:
    def __init__(self, data_path: str) -> None:
        file_pattern = os.path.join(data_path, '*.parquet')
        files = glob.glob(file_pattern)
        if not files:
            raise ValueError('Data not found')

        self.data = pd.concat([pd.read_parquet(file) for file in files])
        self.data['ts_event'] = pd.to_datetime(self.data['ts_event'])
        self.data['date'] = self.data['ts_event'].dt.date

    def get_order_data(self, start_time: int, end_time: int) -> tuple:
        unique_days = self.data['date'].unique()
        chosen_date = random.choice(unique_days)

        day_data = self.data[self.data['date'] == chosen_date].copy()
        day_data.sort_values('ts_event', inplace=True)
        day_data.reset_index(drop=True, inplace=True)
        available_rows = len(day_data)

        if start_time < 0 or start_time >= available_rows:
            raise ValueError(f'start_time out of range')
        if end_time <= start_time:
            raise ValueError('end_time must be greater than start_time')
        if start_time + end_time > available_rows:
            end_time = available_rows - start_time

        data_chunk = day_data.iloc[: start_time + end_time].reset_index(drop=True)

        return data_chunk, start_time
