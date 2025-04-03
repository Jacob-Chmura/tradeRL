from typing import Any

from trade_rl.agents.base import TradingAgent
from trade_rl.env import TradingEnvironment


class RandomTradingAgent(TradingAgent):
    def __init__(self, env: TradingEnvironment) -> None:
        super().__init__(env)

    def get_action(self, obs: Any) -> int:
        return self.env.action_space.sample()
