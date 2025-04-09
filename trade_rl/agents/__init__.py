import gymnasium as gym
from trade_rl.agents.base import TradingAgent
from trade_rl.agents.reinforce import ReinforceAgent
from trade_rl.agents.dqn import DQNAgent
from trade_rl.agents.baselines import (
    BuyStartAgent,
    BuyLastAgent,
    BuyBelowArrivalAgent,
    LinearAgent,
    RandomAgent,
)
from trade_rl.util.args import AgentArgs


def agent_from_env(env: gym.Env, agent_args: AgentArgs) -> TradingAgent:
    agent_type_to_class = {
        'buy_below_arrival': BuyBelowArrivalAgent,
        'buy_last': BuyLastAgent,
        'buy_start': BuyStartAgent,
        'dqn': lambda env: DQNAgent(env, agent_args.dqn_args),
        'linear': LinearAgent,
        'random': RandomAgent,
        'reinforce': lambda env: ReinforceAgent(env, agent_args.reinforce_args),
    }
    agent_type = agent_args.agent_type.lower().strip()
    if agent_type not in agent_type_to_class:
        raise ValueError(
            f'Unknown agent: {agent_type}, expected one of: {agent_type_to_class.keys()}'
        )
    return agent_type_to_class[agent_type](env)  # type: ignore
