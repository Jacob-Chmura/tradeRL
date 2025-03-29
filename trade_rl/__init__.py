"""Not the smartest trading agent, but a trading agent nonetheless."""

import gymnasium as gym
from trade_rl.env import TradingEnvironment

gym.register(
    id='trade_rl/TradingEnvironment-v0',
    entry_point=TradingEnvironment,  # type: ignore
)

__version__ = '0.1.0'

__all__ = [
    '__version__',
]
