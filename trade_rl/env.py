import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import pandas as pd

from trade_rl.order import Order, OrderGenerator
from trade_rl.util.args import Args
from trade_rl.util.data import Data
from trade_rl.util.perf import PerfTracker


class TradingEnvironment(gym.Env):
    def __init__(self, args: Args, data: Data) -> None:
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)  # Skip or Take
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,))  # TODO
        logging.info(f'Created Environment')
        self.tracker = PerfTracker(args)

        self.max_global_step = args.env.max_global_steps
        self.global_step = 0
        self.episode = 0
        self.episode_step = 0
        self.episode_return = 0

        self.data = data
        self.order_generator = OrderGenerator(args.env.order_gen_args)
        self.order, self.order_data, self.start_index = self._new_order()
        self.remaining_qty = self.order.qty

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Any, ...]:
        super().reset(seed=seed)
        self.tracker(self)
        self.order, self.order_data, self.start_index = self._new_order()
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[Any, ...]:
        self.remaining_qty -= action
        terminated = self.episode_step >= self.order.end_time or self.remaining_qty == 0
        truncated = False
        reward = 0 if terminated else -self.remaining_qty
        obs, info = self._get_obs(), self._get_info()

        self.global_step += 1
        self.episode_step += 1
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> Any:
        return self.order_data[self.start_index + self.episode_step]

    def _get_info(self) -> Dict[Any, Any]:
        return {
            'global_step': self.global_step,
            'max_global_step': self.max_global_step,
        }

    def _new_order(self) -> Tuple[Order, pd.Dataframe, int]:
        self.episode += 1
        self.episode_step = self.episode_return = 0
        order = self.order_generator()
        order_data, start_index = self.data.get_order_data(
            order.start_time, order.end_time
        )
        logging.debug(f'New order: {order}')
        return order, order_data, start_index
