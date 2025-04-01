from typing import Any

import gymnasium as gym

from trade_rl.agents.base import TradingAgent


# TODO: Implement heuristic agents
# Heuristic agents are simple rule-based agents that do not learn from experience.
class BuyStartAgent(TradingAgent):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def get_action(self, obs: Any) -> int:
        return 0

    def update(
        self, obs: Any, action: int, reward: float, terminated: bool, next_obs: Any
    ) -> None:
        pass


class BuyLastAgent(TradingAgent):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def get_action(self, obs: Any) -> int:
        return 0

    def update(
        self, obs: Any, action: int, reward: float, terminated: bool, next_obs: Any
    ) -> None:
        pass


class LinearAgent(TradingAgent):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def get_action(self, obs: Any) -> int:
        return 0

    def update(
        self, obs: Any, action: int, reward: float, terminated: bool, next_obs: Any
    ) -> None:
        pass


class BuyBelowArrivalAgent(TradingAgent):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def get_action(self, obs: Any) -> int:
        return 0

    def update(
        self, obs: Any, action: int, reward: float, terminated: bool, next_obs: Any
    ) -> None:
        pass
