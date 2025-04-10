import csv
import json
import time
from dataclasses import asdict
from typing import Any, Dict, List

from trade_rl.util.args import Args
from trade_rl.util.path import get_root_dir, get_run_id


class PerfTracker:
    def __init__(self, fields: List[str], args: Args) -> None:
        self.log_dir = get_root_dir() / 'runs' / get_run_id(args.meta.experiment_name)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_dir / 'config.json', 'w') as f:
            json.dump(asdict(args), f)
        fields += ['time']

        self.fps = {
            'train': open(self.log_dir / 'train_results.csv', 'a+', encoding='utf8'),
            'eval': open(self.log_dir / 'eval_results.csv', 'a+', encoding='utf8'),
        }
        self.writers = {}
        for mode, fp in self.fps.items():
            self.writers[mode] = csv.DictWriter(fp, fields, lineterminator='\n')
            self.writers[mode].writeheader()

        self.train()

    def __del__(self) -> None:
        for fp in self.fps.values():
            fp.close()

    def __call__(self, info: Dict[str, Any]) -> None:
        info['time'] = time.time_ns()
        self.writers[self.mode].writerow(info)

    def train(self) -> None:
        self.mode = 'train'

    def eval(self) -> None:
        self.mode = 'eval'
