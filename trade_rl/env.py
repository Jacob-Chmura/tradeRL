import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import pandas as pd
import torch

from trade_rl.data import (
    Data,
    get_elapsed_time_percentage,
    get_return,
    get_tleft_norm,
    get_vleft_norm,
    get_volume_norm,
    get_vwap_norm,
)
from trade_rl.order import Order, OrderGenerator
from trade_rl.reward_manager import RewardManager
from trade_rl.util.args import Args
from trade_rl.util.perf import PerfTracker


class TradingEnvironment(gym.Env):
    def __init__(self, args: Args, data: Data) -> None:
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)  # Skip or Take
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,))  # TODO
        logging.info(f'Created Environment')
        self.tracker = PerfTracker(args)
        self.reward_manager = RewardManager(self, args.env.reward_args)

        # TODO: Careful about using max steps for train/test
        self.max_global_step = args.env.max_train_steps
        self.global_step = 0
        self.episode = 0
        self.episode_step = 0
        self.episode_return = 0

        self.portfolio: List[Tuple[float, int]] = []

        self.data = data
        self.order_generator = OrderGenerator(args.env.order_gen_args)
        self.order, self.day_data, self.order_data = self._new_order()
        self.remaining_qty = self.order.qty

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Any, ...]:
        super().reset(seed=seed)
        self.tracker(self)
        self.order, self.day_data, self.order_data = self._new_order()
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[Any, ...]:
        if action:
            px = self.order_data['close'].iloc[self.episode_step]
            self.portfolio.append((px, self.episode_step))

        self.remaining_qty -= action
        terminated = self.episode_step >= self.order.duration or self.remaining_qty == 0
        truncated = False
        reward = self.reward_manager(terminated)
        obs, info = self._get_obs(), self._get_info()
        self.global_step += 1
        self.episode_step += 1
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> Any:
        return self.order_data.iloc[self.episode_step]

    def _get_info(self) -> Dict[Any, Any]:
        return {
            'global_step': self.global_step,
            'max_global_step': self.max_global_step,
        }

    def _new_order(self) -> Tuple[Order, pd.DataFrame, pd.DataFrame]:
        self.episode += 1
        self.episode_step = self.episode_return = 0
        self.portfolio = []
        order = self.order_generator()
        logging.debug(f'New order: {order}')

        day_data = self.data.get_random_day_of_data()
        order_mask = day_data.market_second.between(
            order.start_time, order.start_time + order.duration
        )
        order_data = day_data[order_mask].reset_index(drop=True)
        return order, day_data, order_data

    def _compute_feature_vector(self) -> torch.Tensor:
        # Price Features
        current_px = self.order_data['open'][self.episode_step]
        all_day_pxs = self.day_data['open'][
            : self.order.start_time + self.episode_step + 1
        ]
        max_day_px, min_day_px = max(all_day_pxs), min(all_day_pxs)
        market_open_px = self.day_data['open'][0]
        arrival_px = self.order_data['open'][0]
        previous_px = self.order_data['open'][self.episode_step - 1]
        market_vwap = self.order_data['vwap'][self.episode_step - 1]

        # Time features
        market_second = self.order_data['market_second'][self.episode_step]

        # Volume features
        prev_volume = self.order_data['volume'][self.episode_step - 1]
        volume_sma = self.order_data['volume_sma'][self.episode_step - 1]

        return torch.tensor(
            [
                get_vleft_norm(self.remaining_qty, self.order),
                get_tleft_norm(self.episode_step, self.order),
                get_return(previous_px, current_px),
                get_return(market_open_px, current_px),
                get_return(arrival_px, current_px),
                get_return(max_day_px, current_px),
                get_return(min_day_px, current_px),
                get_elapsed_time_percentage(market_second),
                get_vwap_norm(self.portfolio, market_vwap),
                get_volume_norm(prev_volume, volume_sma),
                self.order_data['sma_return_short'][self.episode_step],
                self.order_data['sma_return_long'][self.episode_step],
                self.order_data['ema_return_short'][self.episode_step],
                self.order_data['ema_return_long'][self.episode_step],
                self.order_data['macd'][self.episode_step],
                self.order_data['signal'][self.episode_step],
                self.order_data['volatility'][self.episode_step],
                self.order_data['rsi'][self.episode_step],
                self.order_data['bollinger_percentage'][self.episode_step],
                self.order_data['stoch_k'][self.episode_step],
            ]
        )
