import csv
import json
import time
from dataclasses import asdict
from typing import Any, Dict, List

from trade_rl.util.args import Args
from trade_rl.util.path import get_root_dir, get_run_id


class PerfTracker:
    def __init__(self, fields: List[str], args: Args) -> None:
        log_dir = get_root_dir() / 'runs' / get_run_id(args.meta.experiment_name)
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / 'config.json', 'w') as f:
            json.dump(asdict(args), f)
        fields += ['time']
        self.fp = open(log_dir / 'results.csv', 'a+', encoding='utf8')
        self.writer = csv.DictWriter(self.fp, fieldnames=fields, lineterminator='\n')
        self.writer.writeheader()

    def __del__(self) -> None:
        self.fp.close()

    def __call__(self, info: Dict[str, Any]) -> None:
        info['time'] = time.time_ns()
        self.writer.writerow(info)
