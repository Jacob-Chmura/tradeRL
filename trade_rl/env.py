import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym

from trade_rl.order import Order, OrderGenerator
from trade_rl.util.args import EnvironmentArgs


class TradingEnvironment(gym.Env):
    def __init__(self, config: EnvironmentArgs) -> None:
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)  # Skip or Take
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,))  # TODO
        logging.info(f'Created Environment')

        self.max_global_step = config.max_global_steps
        self.global_step = 0

        self._order_generator = OrderGenerator(config.order_gen_args)
        self._step, self.order, self.remaining_qty = self._new_order()

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Any, ...]:
        super().reset(seed=seed)
        self._step, self.order, self.remaining_qty = self._new_order()
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[Any, ...]:
        self.remaining_qty -= action
        terminated = self._step >= self.order.end_time
        truncated = False
        reward = 1 if terminated else 0
        obs, info = self._get_obs(), self._get_info()

        self.global_step += 1
        self._step += 1
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> Any:
        return self.observation_space.sample()

    def _get_info(self) -> Dict[Any, Any]:
        return {
            'global_step': self.global_step,
            'max_global_step': self.max_global_step,
        }

    def _new_order(self) -> Tuple[int, Order, int]:
        order = self._order_generator()
        logging.debug(f'New order: {order}')
        return 0, order, order.qty
