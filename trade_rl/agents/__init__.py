import gymnasium as gym
from trade_rl.agents.base import TradingAgent
from trade_rl.agents.random import RandomAgent
from trade_rl.agents.reinforce import ReinforceAgent
from trade_rl.agents.dqn import DQNAgent
from trade_rl.agents.ppo import PPOAgent
from trade_rl.agents.heuristic import (
    BuyStartAgent,
    BuyLastAgent,
    BuyBelowArrivalAgent,
    LinearAgent,
)


def agent_from_env(env: gym.Env, agent_type: str) -> TradingAgent:
    agent_type_to_class = {
        'buy_below_arrival': BuyBelowArrivalAgent,
        'buy_last': BuyLastAgent,
        'buy_start': BuyStartAgent,
        'dqn': DQNAgent,
        'linear': LinearAgent,
        'ppo': PPOAgent,
        'random': RandomAgent,
        'reinforce': ReinforceAgent,
    }
    agent_type = agent_type.lower().strip()
    if agent_type not in agent_type_to_class:
        raise ValueError(
            f'Unknown agent: {agent_type}, expected one of: {agent_type_to_class.keys()}'
        )
    return agent_type_to_class[agent_type](env)  # type: ignore
