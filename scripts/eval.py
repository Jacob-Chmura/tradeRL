import argparse
from pathlib import Path

from trade_rl.data import Data
from trade_rl.env import Info
from trade_rl.run import test
from trade_rl.util.args import parse_json
from trade_rl.util.perf import PerfTracker
from trade_rl.util.seed import seed_everything


def find_all_configs(base_dir: Path):
    return list(base_dir.rglob('config.json'))


def run(args, config_path):
    # Some configs pointed to .csv raw data?
    args.env.feature_args.test_data_path = args.env.feature_args.test_data_path.replace(
        '.csv', '.parquet'
    )
    test_data = Data(args.env.feature_args, train=False)
    seed_everything(args.meta.global_seed)
    tracker = PerfTracker(Info.get_fields(), args, run_id=config_path.parent)
    test(test_data, tracker, args)


def main(config_paths):
    for config_path in config_paths:
        args = parse_json(str(config_path))
        run(args, config_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='TradeRL',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        help='Path to runs folder you want to rerun',
    )

    args_ = parser.parse_args()
    config_paths = find_all_configs(Path(args_.base_dir))
    main(config_paths)
