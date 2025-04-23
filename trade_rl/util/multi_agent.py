from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym

from trade_rl.data import Data
from trade_rl.env import TradingEnvironment
from trade_rl.order import OrderGenerator
from trade_rl.util.args import Args


class MultiAgentTradingEnv(gym.Env):
    def __init__(self, args: Args, data: Data, num_agents: int):
        super().__init__()
        self.num_agents = num_agents
        self._shared_order_gen = OrderGenerator(args.env.order_gen_args)
        self._shared_data = data
        self._agent_done = [False] * num_agents
        self._last_obs = [None] * num_agents
        self._last_info = [None] * num_agents

        self.envs = [TradingEnvironment(args, data) for _ in range(num_agents)]

        self.action_space = gym.spaces.Tuple([env.action_space for env in self.envs])
        self.observation_space = gym.spaces.Tuple(
            [env.observation_space for env in self.envs]
        )

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Any, ...]:
        order = self._shared_order_gen()
        _, day_data = self._shared_data.get_random_day_of_data()
        last = day_data.market_second.max()
        if order.start_time + order.duration >= last:
            order.duration = int(last - order.start_time - 1)

        obs_list, info_list = [], []
        for env in self.envs:
            env.day_data = day_data
            env.info = type(env.info)()
            env.info.new_episode(order)
            o = env._get_obs()
            obs_list.append(o)
            info_list.append(env.info.to_dict())

        return obs_list, info_list

    def step(self, actions: List[int]) -> Tuple[List[Any], ...]:  # type: ignore
        obs: List[List[float]] = []
        rews: List[float] = []
        dones: List[bool] = []
        truns: List[bool] = []
        infos: List[Dict[Any, Any]] = []

        for i, (env, a) in enumerate(zip(self.envs, actions)):
            if self._agent_done[i]:
                obs.append(self._last_obs[i])  # type: ignore
                rews.append(0.0)
                dones.append(True)
                truns.append(False)
                infos.append(self._last_info[i])  # type: ignore
            else:
                o, r, d, t, info = env.step(a)
                obs.append(o)
                rews.append(r)
                dones.append(d)
                truns.append(t)
                infos.append(info)
                if d:
                    self._agent_done[i] = True

                self._last_obs[i] = o
                self._last_info[i] = info

        return obs, rews, dones, truns, infos
