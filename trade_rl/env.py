from typing import Any, Dict, Optional, Tuple

import gymnasium as gym

from trade_rl.order import Order


class TradingEnvironment(gym.Env):
    def __init__(self, order: Order) -> None:
        self.action_space = gym.spaces.Discrete(2)  # Skip or Take
        self.observation_space = gym.spaces.Discrete(1)  # TODO

        self._order = order
        self._step = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Any, ...]:
        super().reset(seed=seed)
        self._step = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[Any, ...]:
        terminated = self._step >= self._order.end_time
        truncated = False
        reward = 1 if terminated else 0
        obs = self._get_obs()
        info = self._get_info()

        self._step += 1
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> Any:
        return 0

    def _get_info(self) -> Dict[Any, Any]:
        return {}
