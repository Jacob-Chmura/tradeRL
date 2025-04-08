from typing import Any

from trade_rl.agents.base import TradingAgent


class RandomAgent(TradingAgent):
    def get_action(self, obs: Any) -> int:
        return self.env.action_space.sample()
