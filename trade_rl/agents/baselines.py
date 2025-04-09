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
        return self.env.info.step >= self.env.order.duration - self.env.order.qty


class BuyLinearScheduleAgent(TradingAgent):
    def get_action(self, obs: Any) -> int:
        interval = self.env.order.duration // self.env.order.qty
        return self.env.info.step % interval == 0


class BuyBelowArrivalAgent(TradingAgent):
    def get_action(self, obs: Any) -> int:
        if self.env.info.step == 0:
            self.arrival_px = self.env.current_market_data['open']
        return self.env.current_market_data['open'] < self.arrival_px
