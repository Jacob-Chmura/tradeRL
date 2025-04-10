from tqdm import tqdm

from trade_rl.agents import agent_from_env
from trade_rl.data import Data
from trade_rl.env import TradingEnvironment
from trade_rl.util.args import Args
from trade_rl.util.seed import seed_everything


def run(args: Args) -> None:
    train_data = Data(args.env.feature_args, train=True)
    test_data = Data(args.env.feature_args, train=False)
    for run_number in range(args.meta.num_runs):  # TODO: Concurrency
        args.meta.global_seed += run_number
        seed_everything(args.meta.global_seed)
        train(train_data, args)
        test(test_data, args)


def train(data: Data, args: Args) -> None:
    env = TradingEnvironment(args, data)
    agent = agent_from_env(env, args.agent)

    with tqdm(total=args.env.max_train_steps) as pbar:
        while env.info.global_step < args.env.max_train_steps:
            obs, info = env.reset()
            done = False
            while not done:
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                agent.update(obs, action, reward, terminated, next_obs)
                done = terminated or truncated
                obs = next_obs
                pbar.update(1)
    agent.save_model(env.tracker.log_dir)


def test(data: Data, args: Args) -> None:
    # TODO: Ensure perf tracker is aware of the fact we are in eval mode
    env = TradingEnvironment(args, data)
    agent = agent_from_env(env, args.agent)
    agent.load_model('TODO')

    with tqdm(total=args.env.max_test_steps) as pbar:
        while env.info.global_step < args.env.max_test_steps:
            obs, info = env.reset()
            done = False
            while not done:
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                obs = next_obs
                pbar.update(1)
