import random
from typing import Any

from trade_rl.agents.base import TradingAgent


class RandomAgent(TradingAgent):
    def get_action(self, obs: Any) -> int:
        buy_prob = self.env.info.order_qty / self.env.info.order_duration
        return random.random() < buy_prob * 1.1


class BuyStartAgent(TradingAgent):
    def get_action(self, obs: Any) -> int:
        return 1


class BuyLastAgent(TradingAgent):
    def get_action(self, obs: Any) -> int:
        start_to_buy_time = self.env.info.order_duration - self.env.info.order_qty
        return self.env.info.step >= start_to_buy_time


class BuyBelowArrivalAgent(TradingAgent):
    def get_action(self, obs: Any) -> int:
        return self.env.current['open'] < self.env.order_arrival['open']
