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
    qty_left: int = -1

    # Performance Info
    portfolio: Optional[List[Tuple[float, int]]] = None
    total_reward: float = 0
    agent_vwap: float = 0
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
        self.qty_left = order.qty
        self.portfolio = []
        self.total_reward = 0
        self.agent_vwap = 0
        self.arrival_slippage = 0
        self.vwap_slippage = 0
        self.oracle_slippage = 0

    def new_step(self, action: int, current_market: Dict[str, Any]) -> None:
        if action:
            self.qty_left -= 1
            self.portfolio.append((current_market['close'], self.step))  # type: ignore
            self.agent_vwap = np.mean([x[0] for x in self.portfolio])  # type: ignore
        self.global_step += 1
        self.step += 1

    def update_perf(self, slippages: Dict[str, float], reward: float) -> None:
        self.arrival_slippage = slippages['arrival']
        self.vwap_slippage = slippages['vwap']
        self.oracle_slippage = slippages['oracle']
        self.total_reward += reward

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop('portfolio')  # Don't serialized this
        return d


class TradingEnvironment(gym.Env):
    def __init__(self, args: Args, data: Data) -> None:
        super().__init__()
        self.data = data
        self.info = Info()
        self.action_space = gym.spaces.Discrete(2)  # Skip or Take
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,))  # TODO

        self.reward_manager = RewardManager(self, args.env.reward_args)
        self.order_generator = OrderGenerator(args.env.order_gen_args)
        self.tracker = PerfTracker(list(self.info.to_dict().keys()), args)
        self.day_data = self._new_order()

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Any, ...]:
        super().reset(seed=seed)
        self.tracker(self.info.to_dict())
        self.day_data = self._new_order()
        return self._get_obs(), self.info.to_dict()

    def step(self, action: int) -> Tuple[Any, ...]:
        self.info.new_step(action, self.current_market)
        done = self.info.step >= self.info.order_duration or self.info.qty_left == 0
        truncated = False
        slippages, reward = self.reward_manager(done)
        self.info.update_perf(slippages, reward)
        obs, info = self._get_obs(), self.info.to_dict()
        logging.debug(info)
        return obs, reward, done, truncated, info

    @property
    def current_market(self) -> Dict[str, Any]:
        return self._get_market_data(self.info.order_start_time + self.info.step)

    @property
    def previous_market(self) -> Dict[str, Any]:
        return self._get_market_data(self.info.order_start_time + self.info.step - 1)

    @property
    def order_arrival_market(self) -> Dict[str, Any]:
        return self.day_data.iloc[self.info.order_start_time].to_dict()

    @property
    def market_open(self) -> Dict[str, Any]:
        return self._get_market_data(i=0)

    @property
    def order_duration_market(self) -> pd.DataFrame:
        order_end_time = self.info.order_start_time + self.info.order_duration
        order_mask = self.day_data.market_second.between(
            self.info.order_start_time, order_end_time
        )
        return self.day_data[order_mask].reset_index(drop=True)

    def _new_order(self) -> pd.DataFrame:
        order = self.order_generator()
        logging.debug(f'New order: {order}')
        self.info.new_episode(order)
        self.info.order_date, day_data = self.data.get_random_day_of_data()
        return day_data

    def _get_obs(self) -> np.ndarray:
        current_market = self.current_market
        previous_market = self.previous_market
        order_arrival_market = self.order_arrival_market
        market_open = self.market_open

        day_pxs = self.day_data['open'][: self.info.order_start_time + self.info.step]
        get_return = lambda prev, curr: (curr - prev) / prev

        obs = np.array(
            [
                self.info.qty_left / self.info.order_qty,
                self.info.step / self.info.order_duration,
                get_return(previous_market['open'], current_market['open']),
                get_return(market_open['open'], current_market['open']),
                get_return(order_arrival_market['open'], current_market['open']),
                get_return(min(day_pxs), current_market['open']),
                get_return(max(day_pxs), current_market['open']),
                self.info.agent_vwap / previous_market['vwap']
                if previous_market['vwap'] != 0
                else 0,
                previous_market['volume'] / previous_market['volume_sma']
                if previous_market['volume_sma'] != 0
                else 0,
                current_market['market_second'] / 23400,
                current_market['sma_return_short'],
                current_market['sma_return_long'],
                current_market['ema_return_short'],
                current_market['ema_return_long'],
                current_market['macd'],
                current_market['signal'],
                current_market['volatility'],
                current_market['rsi'],
                current_market['bollinger_percentage'],
                current_market['stoch_k'],
            ]
        ).clip(-3, 3)
        obs[np.isnan(obs)] = 0.0
        return obs

    def _get_market_data(self, i: int) -> Dict[str, Any]:
        return self.day_data.iloc[i].to_dict()
