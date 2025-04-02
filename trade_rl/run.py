from tqdm import tqdm

from trade_rl.agents import RandomTradingAgent
from trade_rl.env import TradingEnvironment
from trade_rl.util.args import EnvironmentArgs


def run(env_args: EnvironmentArgs) -> None:
    env = TradingEnvironment(config=env_args)
    agent = RandomTradingAgent(env=env)

    n_episodes = 5
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            agent.update(obs, action, reward, terminated, next_obs)

            done = terminated or truncated
            obs = next_obs
