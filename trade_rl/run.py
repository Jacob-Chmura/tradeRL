import gymnasium as gym
from tqdm import tqdm

from trade_rl.agents import RandomTradingAgent


def run() -> None:
    env = gym.make('trade_rl/TradingEnvironment-v0')
    agent = RandomTradingAgent(env=env)

    n_episodes = 5
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            agent.update(obs, action, reward, terminated, next_obs)  # type: ignore

            done = terminated or truncated
            obs = next_obs
