import logging

import gymnasium as gym
from tqdm import tqdm

from trade_rl.order import Order


def run() -> None:
    order = Order(sym='TSLA', qty=100, end_time=1337)
    logging.info(f'Order: {order}')

    env = gym.make('trade_rl/TradingEnvironment-v0', order=order)
    logging.info(f'Created Environment')

    n_episodes = 100
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        while not done:
            action = 0
            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
