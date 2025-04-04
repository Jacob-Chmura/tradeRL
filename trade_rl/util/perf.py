import csv
import json
import time
from dataclasses import asdict

from trade_rl.util.args import Args
from trade_rl.util.path import get_root_dir, get_run_id

FIELDS = ['time', 'order_id', 'global_step', 'episode', 'return']


class PerfTracker:
    def __init__(self, args: Args) -> None:
        dir = get_root_dir() / 'runs' / get_run_id(args.meta.experiment_name)
        dir.mkdir(parents=True, exist_ok=True)
        with open(dir / 'config.json', 'w') as f:
            json.dump(asdict(args), f)

        self.fp = open(dir / 'results.csv', 'a+', encoding='utf8')
        self.writer = csv.DictWriter(self.fp, fieldnames=FIELDS)
        self.writer.writeheader()

    def __del__(self) -> None:
        self.fp.close()

    def __call__(self, env: 'TradingEnvironment') -> None:  # type: ignore
        data = {
            'time': time.time_ns(),
            'order_id': env.order.order_id,
            'global_step': env.global_step,
            'episode': env.episode,
            'return': env.episode_return,
        }
        self.writer.writerow(data)
