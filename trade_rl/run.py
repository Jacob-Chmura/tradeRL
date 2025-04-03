from tqdm import tqdm

from trade_rl.agents import agent_from_env
from trade_rl.env import TradingEnvironment
from trade_rl.util.args import EnvironmentArgs


def run(env_args: EnvironmentArgs) -> None:
    env = TradingEnvironment(config=env_args)
    agent = agent_from_env(env, agent_type='random')
    with tqdm(total=env_args.max_global_steps) as pbar:
        while env.global_step < env_args.max_global_steps:
            obs, info = env.reset()
            done = False
            while not done:
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                agent.update(obs, action, reward, terminated, next_obs)
                done = terminated or truncated
                obs = next_obs

                pbar.update(1)
