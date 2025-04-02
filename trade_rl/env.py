import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym

from trade_rl.order import OrderGenerator
from trade_rl.util.args import EnvironmentArgs


class TradingEnvironment(gym.Env):
    def __init__(self, config: EnvironmentArgs) -> None:
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)  # Skip or Take
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,))  # TODO
        logging.info(f'Created Environment')

        self._step = 0
        self._order_generator = OrderGenerator(config.order_gen_args)
        self.order = self._order_generator()
        logging.info(f'New order: {self.order}')

        self.remaining_shares_to_buy = self.order.qty

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Any, ...]:
        super().reset(seed=seed)
        self._step = 0
        self.order = self._order_generator()
        self.remaining_shares_to_buy = self.order.qty
        logging.info(f'New order: {self.order}')

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[Any, ...]:
        self.remaining_shares_to_buy -= action
        terminated = self._step >= self.order.end_time
        truncated = False
        reward = 1 if terminated else 0
        obs = self._get_obs()
        info = self._get_info()

        self._step += 1
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> Any:
        return self.observation_space.sample()

    def _get_info(self) -> Dict[Any, Any]:
        return {}
