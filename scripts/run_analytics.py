import argparse
import json
import pathlib
from typing import Any, Dict

import pandas as pd
from tabulate import tabulate

from trade_rl.util.path import get_root_dir

parser = argparse.ArgumentParser(
    description='Get Analytics Results for a TradeRL experiment',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--experiment-name',
    type=str,
    help='Name of experiment to run analytics for',
)
parser.add_argument(
    '--artifacts-dir',
    type=str,
    default='artifacts',
    help='Path to artifact directory containing plots',
)


def main() -> None:
    args = parser.parse_args()
    results_dir = get_root_dir() / 'runs' / args.experiment_name
    if not results_dir.is_dir():
        raise FileNotFoundError(f'Results directory: {results_dir.resolve()}')

    df = parse_results_dir(results_dir)

    perf_cols = ['total_reward', 'arrival_slippage', 'vwap_slippage', 'oracle_slippage']
    eval_data = (
        df[df.split == 'eval'].groupby(['config-agent-agent_type'])[perf_cols].mean()
    ).reset_index()
    headers = ['Agent', 'Reward', 'Arrival (bps)', 'VWAP (bps)', 'Oracle (bps)']
    print(tabulate(eval_data, headers=headers, tablefmt='fancy_grid'))

    # TODO: Add actual analytics


def save_latex_tables(latex_tables: Dict[str, str], artifacts_dir_str: str) -> None:
    artifacts_dir = pathlib.Path(artifacts_dir_str)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for table_name, table_latex in latex_tables.items():
        with open(artifacts_dir / f'{table_name}.txt', 'w') as f:
            f.write(table_latex)


def parse_results_dir(results_dir: pathlib.Path) -> pd.DataFrame:
    dfs = []
    for result_dir in results_dir.iterdir():
        print(f'Reading results from: {result_dir}')
        with open(result_dir / 'config.json') as f:
            config = json.load(f)

        train_results = pd.read_csv(result_dir / 'train_results.csv')
        eval_results = pd.read_csv(result_dir / 'eval_results.csv')

        train_results['split'] = 'train'
        eval_results['split'] = 'eval'
        results = pd.concat([train_results, eval_results])

        def _recurse_config(config: Dict[str, Any], prefix: str = 'config') -> None:
            for k, v in config.items():
                if isinstance(v, dict):
                    _recurse_config(v, f'{prefix}-{k}')
                elif isinstance(v, list):
                    results[f'{prefix}-{k}'] = ''.join([str(x) for x in v])
                else:
                    results[f'{prefix}-{k}'] = v

        _recurse_config(config)
        dfs.append(results)
    df = pd.concat(dfs)
    return df


if __name__ == '__main__':
    main()
