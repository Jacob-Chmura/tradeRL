import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd

from trade_rl.data import Data
from trade_rl.order import Order, OrderGenerator
from trade_rl.reward_manager import RewardManager
from trade_rl.util.args import Args
from trade_rl.util.perf import PerfTracker


@dataclass(slots=True)
class Info:
    # Global Info
    max_global_steps: int = 0
    global_step: int = 0

    # Episode Info
    episode: int = 0
    step: int = 0
    order_id: str = ''
    order_start_time: int = -1
    order_duration: int = -1
    order_qty: int = -1
    order_symbol: str = ''
    order_date: str = ''
    remaining_qty: int = -1

    # Performance Info
    portfolio: Optional[List[Tuple[float, int]]] = None
    returns: float = 0
    agent_vwap: float = 0
    market_vwap: float = 0
    arrival_slippage: float = 0
    vwap_slippage: float = 0
    oracle_slippage: float = 0

    def new_episode(self, order: Order) -> None:
        self.episode += 1
        self.step = 0
        self.order_id = order.order_id
        self.order_start_time = order.start_time
        self.order_duration = order.duration
        self.order_qty = order.qty
        self.order_symbol = order.sym
        self.order_date = ''  # TODO
        self.remaining_qty = order.qty
        self.portfolio = []
        self.returns = 0
        self.agent_vwap = 0
        self.market_vwap = 0
        self.arrival_slippage = 0
        self.vwap_slippage = 0
        self.oracle_slippage = 0

    def new_step(self, action: int, order_data: pd.DataFrame) -> None:
        if action:
            self.remaining_qty -= 1
            self.portfolio.append((order_data['close'][self.step], self.step))  # type: ignore
        self.global_step += 1
        self.step += 1

        # TODO: Update performance


class TradingEnvironment(gym.Env):
    def __init__(self, args: Args, data: Data) -> None:
        super().__init__()
        self.data = data
        self.info = Info()
        self.action_space = gym.spaces.Discrete(2)  # Skip or Take
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,))  # TODO

        self.reward_manager = RewardManager(self, args.env.reward_args)
        self.order_generator = OrderGenerator(args.env.order_gen_args)
        self.tracker = PerfTracker(list(asdict(self.info)), args)

        self.order, self.day_data, self.order_data = self._new_order()
        logging.info(f'Created Environment')

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Any, ...]:
        super().reset(seed=seed)
        self.tracker(asdict(self.info))
        self.order, self.day_data, self.order_data = self._new_order()
        return self._get_obs(), asdict(self.info)

    def step(self, action: int) -> Tuple[Any, ...]:
        self.info.new_step(action, self.order_data)
        obs, info = self._get_obs(), asdict(self.info)
        print(info)
        print(np.round(obs, 2))
        input()
        terminated = (
            self.info.step >= self.info.order_duration or self.info.remaining_qty == 0
        )
        truncated = False
        reward = self.reward_manager(terminated)
        return obs, reward, terminated, truncated, info

    @property
    def current_market_data(self) -> Dict[str, Any]:
        return self.order_data.iloc[self.info.step].to_dict()

    @property
    def previous_market_data(self) -> Dict[str, Any]:
        return self.order_data.iloc[self.info.step - 1].to_dict()

    @property
    def order_arrival_market_data(self) -> Dict[str, Any]:
        return self.order_data.iloc[0].to_dict()

    @property
    def market_open_market_data(self) -> Dict[str, Any]:
        return self.day_data.iloc[0].to_dict()

    def _new_order(self) -> Tuple[Order, pd.DataFrame, pd.DataFrame]:
        order = self.order_generator()
        logging.debug(f'New order: {order}')
        self.info.new_episode(order)

        day_data = self.data.get_random_day_of_data()
        order_end_time = order.start_time + order.duration
        order_mask = day_data.market_second.between(order.start_time, order_end_time)
        order_data = day_data[order_mask].reset_index(drop=True)
        return order, day_data, order_data

    def _get_obs(self) -> np.ndarray:
        # Price Features
        current_px = self.current_market_data['open']
        arrival_px = self.order_arrival_market_data['open']
        market_open_px = self.market_open_market_data['open']
        previous_px = self.previous_market_data['open']
        all_pxs = self.day_data['open'][: self.order.start_time + self.info.step + 1]
        max_day_px, min_day_px = max(all_pxs), min(all_pxs)

        agent_vwap = np.mean([x[0] for x in self.info.portfolio])  # type: ignore
        market_vwap = self.previous_market_data['vwap']

        # Time features
        market_second = self.current_market_data['market_second']

        # Volume features
        prev_volume = self.previous_market_data['volume']
        volume_sma = self.previous_market_data['volume_sma']

        get_return = lambda prev, curr: (curr - prev) / prev

        return np.array(
            [
                self.info.remaining_qty / self.order.qty,
                self.info.step / self.order.duration,
                get_return(previous_px, current_px),
                get_return(market_open_px, current_px),
                get_return(arrival_px, current_px),
                get_return(max_day_px, current_px),
                get_return(min_day_px, current_px),
                market_second / 23400,
                agent_vwap / market_vwap if market_vwap != 0 else 0,
                prev_volume / volume_sma if volume_sma != 0 else 0,
                self.order_data['sma_return_short'][self.info.step],
                self.order_data['sma_return_long'][self.info.step],
                self.order_data['ema_return_short'][self.info.step],
                self.order_data['ema_return_long'][self.info.step],
                self.order_data['macd'][self.info.step],
                self.order_data['signal'][self.info.step],
                self.order_data['volatility'][self.info.step],
                self.order_data['rsi'][self.info.step],
                self.order_data['bollinger_percentage'][self.info.step],
                self.order_data['stoch_k'][self.info.step],
            ]
        ).clip(-1, 1)


# TODO: Move somewhere
def linear_schedule(start: float, end: float, duration: float, t: int) -> float:
    slope = (end - start) / duration
    return max(slope * t + start, end)
