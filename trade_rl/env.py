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
        self.order, self.order_data, self.start_index, self.max_steps = (
            self._new_order()
        )
        self.remaining_qty = self.order.qty

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Any, ...]:
        super().reset(seed=seed)
        self.tracker(self)
        self.order, self.order_data, self.start_index, self.max_steps = (
            self._new_order()
        )
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[Any, ...]:
        if action:
            price = self.order_data['close'].iloc[self.start_index + self.episode_step]
            self.portfolio.append((price, self.episode_step))

        self.remaining_qty -= action
        terminated = self.episode_step >= self.max_steps or self.remaining_qty == 0
        truncated = False
        reward = self.reward_manager(terminated)
        obs, info = self._get_obs(), self._get_info()
        self.global_step += 1
        self.episode_step += 1
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> Any:
        return self.order_data.iloc[self.start_index + self.episode_step]

    def _get_info(self) -> Dict[Any, Any]:
        return {
            'global_step': self.global_step,
            'max_global_step': self.max_global_step,
        }

    def _new_order(self) -> Tuple[Order, pd.DataFrame, int, int]:
        self.episode += 1
        self.episode_step = self.episode_return = 0
        self.portfolio = []
        order = self.order_generator()
        order_data, start_index, max_steps = self.data.get_order_data(
            order.start_time, order.duration
        )
        logging.debug(f'New order: {order}')
        return order, order_data, start_index, max_steps

    def _compute_feature_vector(self) -> torch.Tensor:
        # Price features
        current_index = self.start_index + self.episode_step
        current_price = self.order_data['open'][current_index]
        all_current_day_prices = self.order_data['open'][: current_index + 1]
        max_day_price = max(all_current_day_prices)
        min_day_price = min(all_current_day_prices)
        previous_price = self.order_data['open'][current_index - 1]
        episode_first_price = self.order_data['open'][self.start_index]
        day_first_price = self.order_data['open'][0]
        current_market_vwap = self.order_data['vwap'][current_index - 1]

        # Time features
        market_second = self.order_data['market_second'][current_index]

        # Volume features
        prev_volume = self.order_data['volume'][current_index - 1]
        volume_sma = self.order_data['volume_sma'][current_index - 1]

        return torch.tensor(
            [
                get_vleft_norm(self.remaining_qty, self.order),
                get_tleft_norm(self.episode_step, self.order),
                get_return(previous_price, current_price),
                get_return(day_first_price, current_price),
                get_return(episode_first_price, current_price),
                get_return(max_day_price, current_price),
                get_return(min_day_price, current_price),
                get_elapsed_time_percentage(market_second),
                get_vwap_norm(self.portfolio, current_market_vwap),
                get_volume_norm(prev_volume, volume_sma),
                self.order_data['sma_return_short'][current_index],
                self.order_data['sma_return_long'][current_index],
                self.order_data['ema_return_short'][current_index],
                self.order_data['ema_return_long'][current_index],
                self.order_data['macd'][current_index],
                self.order_data['signal'][current_index],
                self.order_data['volatility'][current_index],
                self.order_data['rsi'][current_index],
                self.order_data['bollinger_percentage'][current_index],
                self.order_data['stoch_k'][current_index],
            ]
        )
