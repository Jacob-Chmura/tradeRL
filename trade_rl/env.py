import logging
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd

from trade_rl.data import Data
from trade_rl.order import Order, OrderGenerator
from trade_rl.reward_manager import RewardManager
from trade_rl.util.args import Args


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

    def new_step(self, action: int, current: Dict[str, Any]) -> None:
        if action:
            self.qty_left -= 1
            self.portfolio.append((current['close'], self.step))  # type: ignore
            self.agent_vwap = np.mean([x[0] for x in self.portfolio])  # type: ignore
        self.global_step += 1
        self.step += 1

    def update_perf(self, slippages: Dict[str, float], reward: float) -> None:
        self.arrival_slippage = slippages['arrival']
        self.vwap_slippage = slippages['vwap']
        self.oracle_slippage = slippages['oracle']
        self.total_reward += reward

    def to_dict(self) -> Dict[str, Any]:
        return {field: getattr(self, field) for field in self.get_fields()}

    @classmethod
    def get_fields(cls) -> List[str]:
        skip_fields = ['portfolio']  # Don't serialized this
        return [v.name for v in fields(cls) if v.name not in skip_fields]


class TradingEnvironment(gym.Env):
    OBS_DIM = 20

    def __init__(self, args: Args, data: Data) -> None:
        super().__init__()
        self.data = data
        self.info = Info()
        self.action_space = gym.spaces.Discrete(2)  # Skip or Take
        self.observation_space = gym.spaces.Box(low=-3, high=3, shape=(self.OBS_DIM,))

        self.reward_manager = RewardManager(self, args.env.reward_args)
        self.order_generator = OrderGenerator(args.env.order_gen_args)
        self.day_data = self._new_order()

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Any, ...]:
        super().reset(seed=seed)
        self.day_data = self._new_order()
        return self._get_obs(), self.info.to_dict()

    def step(self, action: int) -> Tuple[Any, ...]:
        self.info.new_step(action, self.current)
        done = self.info.step >= self.info.order_duration or self.info.qty_left == 0
        truncated = False
        slippages, reward = self.reward_manager(done)
        self.info.update_perf(slippages, reward)
        obs, info = self._get_obs(), self.info.to_dict()
        logging.debug(info)
        return obs, reward, done, truncated, info

    @property
    def current(self) -> Dict[str, Any]:
        return self._get_market_data(self.info.order_start_time + self.info.step)

    @property
    def previous(self) -> Dict[str, Any]:
        return self._get_market_data(self.info.order_start_time + self.info.step - 1)

    @property
    def order_arrival(self) -> Dict[str, Any]:
        return self._get_market_data(self.info.order_start_time)

    @property
    def market_open(self) -> Dict[str, Any]:
        return self._get_market_data(i=0)

    @property
    def order_duration_market(self) -> pd.DataFrame:
        order_start_time = self.info.order_start_time
        order_end_time = order_start_time + self.info.order_duration
        mask = self.day_data.market_second.between(order_start_time, order_end_time)
        return self.day_data[mask].reset_index(drop=True)

    def _new_order(self) -> pd.DataFrame:
        order = self.order_generator()
        logging.debug(f'New order: {order}')
        self.info.new_episode(order)
        self.info.order_date, day_data = self.data.get_random_day_of_data()
        return day_data

    def _get_obs(self) -> np.ndarray:
        current = self.current
        previous = self.previous
        order_arrival = self.order_arrival
        market_open = self.market_open

        day_pxs = self.day_data['open'][: self.info.order_start_time + self.info.step]
        safe_divide = lambda num, denom: num / denom if denom != 0 else 0
        get_return = lambda prev, curr: safe_divide(curr - prev, prev)

        obs = np.zeros(self.OBS_DIM)
        obs[0] = self.info.qty_left / self.info.order_qty
        obs[1] = self.info.step / self.info.order_duration
        obs[2] = get_return(previous['open'], current['open'])
        obs[3] = get_return(market_open['open'], current['open'])
        obs[4] = get_return(order_arrival['open'], current['open'])
        obs[5] = get_return(min(day_pxs), current['open'])
        obs[6] = get_return(max(day_pxs), current['open'])
        obs[7] = safe_divide(self.info.agent_vwap, previous['vwap'])
        obs[8] = safe_divide(previous['volume'], previous['volume_sma'])
        obs[9] = current['market_second'] / 23400
        obs[10] = current['sma_return_short']
        obs[11] = current['sma_return_long']
        obs[12] = current['ema_return_short']
        obs[13] = current['ema_return_long']
        obs[14] = current['macd']
        obs[15] = current['signal']
        obs[16] = current['volatility']
        obs[17] = current['rsi']
        obs[18] = current['bollinger_percentage']
        obs[19] = current['stoch_k']

        obs = obs.clip(-3, 3)
        obs[np.isnan(obs)] = 0.0
        return obs

    def _get_market_data(self, i: int) -> Dict[str, Any]:
        return self.day_data.iloc[i].to_dict()
