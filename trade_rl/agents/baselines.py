from typing import Any

from trade_rl.agents.base import TradingAgent


class RandomAgent(TradingAgent):
    def get_action(self, obs: Any) -> int:
        return self.env.action_space.sample()


class BuyStartAgent(TradingAgent):
    def get_action(self, obs: Any) -> int:
        return 1


class BuyLastAgent(TradingAgent):
    def get_action(self, obs: Any) -> int:
        start_to_buy_time = self.env.info.order_duration - self.env.info.order_qty
        return self.env.info.step >= start_to_buy_time


class BuyLinearScheduleAgent(TradingAgent):
    def get_action(self, obs: Any) -> int:
        interval = self.env.info.order_duration // self.env.info.order_qty
        return self.env.info.step % interval == 0


class BuyBelowArrivalAgent(TradingAgent):
    def get_action(self, obs: Any) -> int:
        if self.env.info.step == 0:
            self.arrival_px = self.env.current_market['open']
        return self.env.current_market['open'] < self.arrival_px
