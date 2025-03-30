import argparse

from trade_rl.run import run
from trade_rl.util.args import parse_args
from trade_rl.util.logging import setup_basic_logging
from trade_rl.util.path import get_root_dir
from trade_rl.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TradeRL',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file',
    type=str,
    default='config/default.yaml',
    help='Path to yaml configuration file to use',
)


def main() -> None:
    args = parser.parse_args()
    config_file_path = get_root_dir() / args.config_file
    meta_args, env_args, *_ = parse_args(config_file_path)
    setup_basic_logging(meta_args.log_file_path)
    seed_everything(meta_args.global_seed)
    run(env_args)


if __name__ == '__main__':
    main()
