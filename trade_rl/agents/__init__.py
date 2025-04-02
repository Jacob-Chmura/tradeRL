import gymnasium as gym
from trade_rl.agents.base import TradingAgent
from trade_rl.agents.random import RandomTradingAgent
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
        'random': RandomTradingAgent,
        'dqn': DQNAgent,
        'ppo': PPOAgent,
        'buy_start': BuyStartAgent,
        'buy_last': BuyLastAgent,
        'buy_below_arrival': BuyBelowArrivalAgent,
        'linear': LinearAgent,
    }
    agent_type = agent_type.lower().strip()
    if agent_type not in agent_type_to_class:
        raise ValueError(f'Unknown agent type: {agent_type}')
    return agent_type_to_class[agent_type](env)  # type: ignore
