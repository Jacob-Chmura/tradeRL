from tqdm import tqdm

from trade_rl.agents import agent_from_env
from trade_rl.env import TradingEnvironment
from trade_rl.util.args import Args
from trade_rl.util.data import Data


def run(args: Args) -> None:
    train_data = Data('data/TSLA/OCHLV/train.parquet')
    env = TradingEnvironment(args, train_data)
    agent = agent_from_env(env, agent_type='random')
    with tqdm(total=args.env.max_train_steps) as pbar:
        while env.global_step < args.env.max_train_steps:
            obs, info = env.reset()
            done = False
            while not done:
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                agent.update(obs, action, reward, terminated, next_obs)
                done = terminated or truncated
                obs = next_obs

                pbar.update(1)

    # TODO: Make sure agent is in 'eval' mode
    # TODO: Ensure perf tracker is aware of the fact we are in eval mode
    # TODO: Load agent model state dict
    test_data = Data('data/TSLA/OCHLV/test.parquet')
    env = TradingEnvironment(args, test_data)
    agent = agent_from_env(env, agent_type='random')
    with tqdm(total=args.env.max_test_steps) as pbar:
        while env.global_step < args.env.max_test_steps:
            obs, info = env.reset()
            done = False
            while not done:
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                obs = next_obs

                pbar.update(1)
