import csv
import json
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from trade_rl.util.args import Args
from trade_rl.util.path import get_root_dir, get_run_id


class PerfTracker:
    BAR_FORMAT = '{desc}|{bar:20}| {n_fmt}/{total_fmt}, {rate_fmt}'

    # TODO: Fix duplicate heading when rerunning past config for eval
    def __init__(
        self, fields: List[str], args: Args, rerun: Optional[str] = None
    ) -> None:
        if rerun:
            self.log_dir = get_root_dir() / rerun
        else:
            self.log_dir = (
                get_root_dir() / 'runs' / get_run_id(args.meta.experiment_name)
            )
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

        self.stats: Dict[str, Any] = {
            'episode': 0,
            'symbol': '',
            'vwap': OnlineEMA(),
            'ap': OnlineEMA(),
            'urgency': OnlineEMA(),
            'reward': OnlineEMA(),
        }
        self.train()

    def __del__(self) -> None:
        for fp in self.fps.values():
            fp.close()

    def __call__(self, info: Dict[str, Any]) -> None:
        info['time'] = time.time_ns()
        self.writers[self.mode].writerow(info)

        self.stats['episode'] = info['episode']
        self.stats['symbol'] = info['order_symbol']
        self.stats['vwap'].update(info['vwap_slippage'])
        self.stats['ap'].update(info['arrival_slippage'])
        self.stats['urgency'].update(100 * (1 - info['step'] / info['order_duration']))
        self.stats['reward'].update(info['total_reward'])

    def train(self) -> None:
        self.mode = 'train'

    def eval(self) -> None:
        self.mode = 'eval'

    def pretty(self) -> str:
        symbol = self.stats['symbol']
        episode = self.stats['episode']
        ap = self.stats['ap'].ema
        vwap = self.stats['vwap'].ema
        urgency = self.stats['urgency'].ema
        reward = self.stats['reward'].ema
        s = ''
        s += self._color(f'Episode [{str(episode).zfill(6)}]', 'yellow')
        s += '      '
        s += self._color(f'VWAP: {vwap:3.2f} bps', 'red' if vwap > 0 else 'green')
        s += '      '
        s += self._color(f'AP: {ap:3.2f} bps', 'red' if ap > 0 else 'green')
        s += '      '
        s += self._color(f'Reward: {reward:3.2f}', 'cyan')
        s += '      '
        s += self._color(symbol, 'black')
        s += '      '
        s += self._color(f'Urgency: {urgency:3.2f}%', 'white')
        s += '      '
        s += 'Global Step'
        return s

    @staticmethod
    def _color(s: str, color: str, background: bool = False) -> str:
        colors = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']
        return f'\u001b[{10 * background + 60 * (color.upper() == color) + 30 + colors.index(color.lower())}m{s}\u001b[0m'


class OnlineEMA:
    def __init__(self, alpha: float = 0.0005) -> None:
        self.alpha = alpha
        self.ema = 0.0

    def update(self, x: float) -> None:
        self.ema = self.alpha * x + (1 - self.alpha) * self.ema
