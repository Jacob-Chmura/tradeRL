from typing import Any

import gymnasium as gym

from trade_rl.agents.base import TradingAgent


class RandomTradingAgent(TradingAgent):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def get_action(self, obs: Any) -> int:
        return self.env.action_space.sample()

    def update(
        self, obs: Any, action: int, reward: float, terminated: bool, next_obs: Any
    ) -> None:
        pass
