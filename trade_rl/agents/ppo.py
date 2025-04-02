from typing import Any

from trade_rl.agents.base import TradingAgent
from trade_rl.env import TradingEnvironment


# TODO: Implement PPO agent
class PPOAgent(TradingAgent):
    def __init__(self, env: TradingEnvironment) -> None:
        super().__init__(env)

    def get_action(self, obs: Any) -> int:
        return 0

    def update(
        self, obs: Any, action: int, reward: float, terminated: bool, next_obs: Any
    ) -> None:
        pass
