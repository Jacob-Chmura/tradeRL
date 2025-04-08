from typing import Any

from trade_rl.agents.base import TradingAgent
from trade_rl.env import TradingEnvironment


class BuyStartAgent(TradingAgent):
    def get_action(self, obs: Any) -> int:
        current_step = self.env.episode_step
        orderQuantity = self.env.order.qty

        # The agent will buy shares at the start of the episode
        if current_step < orderQuantity:
            self.logger.info(f'BUY a share at start step {current_step}')
            return 1

        # Skip if the order is completed
        else:
            self.logger.info(f'Order complete: SKIP at step {current_step}')
            return 0


class BuyLastAgent(TradingAgent):
    def get_action(self, obs: Any) -> int:
        current_step = self.env.episode_step
        orderQuantity = self.env.order.qty
        finalTimeStep = self.env.order.end_time

        # The agent will buy shares at the end of the episode
        if current_step >= finalTimeStep - orderQuantity:
            self.logger.info(f'BUY a share at last step {current_step}')
            return 1

        # Skip if the order is completed
        else:
            self.logger.info(f'Not the end: SKIP at step {current_step}')
            return 0


class LinearAgent(TradingAgent):
    def get_action(self, obs: Any) -> int:
        # Skip if the order is completed
        if self.env.remaining_qty <= 0:
            self.logger.info(f'Order complete: SKIP at step {self.env.episode_step}')
            return 0

        current_step = self.env.episode_step
        orderQuantity = self.env.order.qty
        finalTimeStep = self.env.order.end_time

        interval = finalTimeStep // orderQuantity

        # The agent will buy shares linearly
        if current_step % interval == 0:
            self.logger.info(f'BUY a share at linear interval at step {current_step}')
            return 1
        return 0


class BuyBelowArrivalAgent(TradingAgent):
    def __init__(self, env: TradingEnvironment) -> None:
        super().__init__(env)
        self.arrival_price = 0.0

    def get_action(self, obs: Any) -> int:
        # At first step, the agent will set the arrival price
        if self.env.episode_step == 0:
            self.arrival_price = self.env.order_data['open'][self.env.start_index]
            return 0

        # Skip if the order is completed
        if self.env.remaining_qty <= 0:
            self.logger.info(f'Order complete: SKIP at step {self.env.episode_step}')
            return 0

        current_index = self.env.start_index + self.env.episode_step
        price = self.env.order_data['open'][current_index]
        if price < self.arrival_price:
            self.logger.info(f'BUY, price below arrival step {self.env.episode_step}')
            return 1
        else:
            self.logger.info(f'SKIP, price above arrival step {self.env.episode_step}')
            return 0
