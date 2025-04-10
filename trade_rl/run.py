from tqdm import tqdm

from trade_rl.agents import agent_from_env
from trade_rl.data import Data
from trade_rl.env import Info, TradingEnvironment
from trade_rl.util.args import Args
from trade_rl.util.perf import PerfTracker
from trade_rl.util.seed import seed_everything


def run(args: Args) -> None:
    train_data = Data(args.env.feature_args, train=True)
    test_data = Data(args.env.feature_args, train=False)
    for run_number in range(args.meta.num_runs):  # TODO: Concurrency
        args.meta.global_seed += run_number
        seed_everything(args.meta.global_seed)
        tracker = PerfTracker(Info.get_fields(), args)
        train(train_data, tracker, args)
        test(test_data, tracker, args)


def train(data: Data, tracker: PerfTracker, args: Args) -> None:
    tracker.train()
    env = TradingEnvironment(args, data)
    agent = agent_from_env(env, args.agent)

    with tqdm(total=env.max_train_steps) as pbar:
        while env.info.global_step < env.max_train_steps:
            obs, info = env.reset()
            done = False
            while not done:
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                agent.update(obs, action, reward, terminated, next_obs, info)
                done = terminated or truncated
                obs = next_obs
                pbar.update(1)
            tracker(info)
    agent.save_model(tracker.log_dir)


def test(data: Data, tracker: PerfTracker, args: Args) -> None:
    tracker.eval()
    env = TradingEnvironment(args, data)
    agent = agent_from_env(env, args.agent)
    agent.load_model(tracker.log_dir)

    with tqdm(total=env.max_test_steps) as pbar:
        while env.info.global_step < env.max_test_steps:
            obs, info = env.reset()
            done = False
            while not done:
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                obs = next_obs
                pbar.update(1)
            tracker(info)
