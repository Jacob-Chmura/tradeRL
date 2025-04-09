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
        # The agent will buy shares at the end of the episode
        if self.env.episode_step >= self.env.order.end_time - self.env.order.qty:
            self.logger.info(f'BUY: step {self.env.episode_step}')
            return 1
        return 0


class BuyLinearScheduleAgent(TradingAgent):
    def get_action(self, obs: Any) -> int:
        # The agent will buy shares linearly
        interval = self.env.order.end_time // self.env.order.qty
        if self.env.episode_step % interval == 0:
            self.logger.info(f'BUY: step {self.env.episode_step}')
            return 1
        return 0


class BuyBelowArrivalAgent(TradingAgent):
    def get_action(self, obs: Any) -> int:
        # At first step, the agent will set the arrival price
        if self.env.episode_step == 0:
            self.arrival_px = self.env.order_data['open'][self.env.start_index]
            return 0

        px = self.env.order_data['open'][self.env.start_index + self.env.episode_step]
        if px < self.arrival_px:
            self.logger.info(f'BUY: step {self.env.episode_step}')
            return 1
        return 0
