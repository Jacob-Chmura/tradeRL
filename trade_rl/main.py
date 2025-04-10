import argparse

from trade_rl.run import run
from trade_rl.util.args import parse_args
from trade_rl.util.logging import setup_basic_logging
from trade_rl.util.path import get_root_dir

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
    args_ = parser.parse_args()
    config_file_path = get_root_dir() / args_.config_file
    args = parse_args(config_file_path)
    setup_basic_logging(args.meta.log_file_path)
    run(args)


if __name__ == '__main__':
    main()
