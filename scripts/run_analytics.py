import argparse
import json
import pathlib
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

    artifacts_dir = pathlib.Path(args.artifacts_dir) / args.experiment_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = parse_results_dir(results_dir)
    train, test = preprocess_results(df)

    plot_learning_curves(train, artifacts_dir)
    plot_test_performance_table(test, artifacts_dir)

    test_performance_specs = [
        {'by': 'order_duration', 'label': 'Order Duration (s)'},
        {'by': 'order_start_time', 'label': 'Order Time'},
    ]
    for test_spec in test_performance_specs:
        plot_test_performance_by(test, artifacts_dir, **test_spec)


def plot_learning_curves(
    df_: pd.DataFrame, artifacts_dir: pathlib.Path, roll: int = 100
) -> None:
    df = df_.copy()
    smooth_series = lambda x: x.rolling(roll).mean().fillna(0)
    df['arrival_slippage'] = smooth_series(df['arrival_slippage'])
    df['vwap_slippage'] = smooth_series(df['vwap_slippage'])
    df['oracle_slippage'] = smooth_series(df['oracle_slippage'])
    df['reward_per_step'] = smooth_series(df['reward_per_step'])
    df['urgency'] = smooth_series(df['urgency'])
    df['time_left'] = smooth_series(df['time_left'])

    f, ax = plt.subplots(2, 3, figsize=(12, 6))
    sns.lineplot(df, x='episode', hue='reward_type', ax=ax[0][0], y='arrival_slippage')
    sns.lineplot(df, x='episode', hue='reward_type', ax=ax[0][1], y='vwap_slippage')
    sns.lineplot(df, x='episode', hue='reward_type', ax=ax[0][2], y='oracle_slippage')
    sns.lineplot(df, x='episode', hue='reward_type', ax=ax[1][0], y='reward_per_step')
    sns.lineplot(df, x='episode', hue='reward_type', ax=ax[1][1], y='urgency')
    sns.lineplot(df, x='episode', hue='reward_type', ax=ax[1][2], y='time_left')

    titles = [
        'Arrival Slippage',
        'VWAP Slippage',
        'Oracle Slippage',
        'Reward / Step',
        'Urgency',
        'Time Left',
    ]
    for i, a in enumerate(ax.flatten()):
        a.spines[['top', 'right']].set_visible(False)
        a.grid(0.3)
        a.get_legend().remove()
        a.set_title(titles[i])
        a.set_xlabel('Episode')
        ylabel = 'Slippage (bps)' if i == 0 else 'Value' if i == 3 else ''
        a.set_ylabel(ylabel)

    handles, labels = ax[0, 0].get_legend_handles_labels()
    f.legend(
        handles,
        labels,
        loc='lower center',
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.1),
        frameon=False,
    )
    plt.subplots_adjust(hspace=0.5)

    save_path = artifacts_dir / 'learning_curves.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Learning curves saved to: {save_path}')


def plot_test_performance_table(
    test: pd.DataFrame, artifacts_dir: pathlib.Path
) -> None:
    perf_cols = [
        'reward_per_step',
        'arrival_slippage',
        'vwap_slippage',
        'oracle_slippage',
    ]
    groups = ['agent', 'reward_type']
    perf_mu = test.groupby(groups)[perf_cols].mean().reset_index()
    perf_std = test.groupby(groups)[perf_cols].std().reset_index()
    perf = perf_mu[groups].copy()
    for perf_col in perf_cols:
        mu = perf_mu[perf_col].round(3).astype(str)
        std = perf_std[perf_col].round(3).astype(str)
        perf[perf_col] = mu.str.cat(std, sep=' ± ')

    headers = [
        'Agent',
        'Reward Type',
        'Reward',
        'Arrival (bps)',
        'VWAP (bps)',
        'Oracle (bps)',
    ]
    print(tabulate(perf, headers=headers, tablefmt='fancy_grid'))
    save_latex_tables(
        {'performance': tabulate(perf, headers=headers, tablefmt='latex')},
        artifacts_dir,
    )


def plot_test_performance_by(
    df: pd.DataFrame, artifacts_dir: pathlib.Path, by: str, label: str
) -> None:
    f, ax = plt.subplots(1, 3, figsize=(12, 3))
    sns.barplot(data=df, x=by, hue='reward_type', ax=ax[0], y='arrival_slippage')
    sns.barplot(data=df, x=by, hue='reward_type', ax=ax[1], y='vwap_slippage')
    sns.barplot(data=df, x=by, hue='reward_type', ax=ax[2], y='oracle_slippage')
    titles = ['Arrival Slippage', 'VWAP Slippage', 'Oracle Slippage']
    for i, a in enumerate(ax.flatten()):
        a.spines[['top', 'right']].set_visible(False)
        a.set_axisbelow(True)
        a.grid(0.1)
        a.get_legend().remove()
        a.set_title(titles[i])
        a.set_xlabel(label)
        ylabel = 'Slippage (bps)' if i == 0 else ''
        a.set_ylabel(ylabel)

    handles, labels = ax[0].get_legend_handles_labels()
    f.legend(
        handles,
        labels,
        loc='lower center',
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.2),
        frameon=False,
    )

    save_path = artifacts_dir / f'test_performance_by_{by}.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Test Performance by {by} saved to: {save_path}')


def preprocess_results(df: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    keep_cols = [
        'split',
        'global_step',
        'episode',
        'agent',
        'reward_type',
        'terminal_penalty',
        'order_start_time',
        'order_duration',
        'order_qty',
        'order_date',
        'qty_left',
        'time_left',
        'urgency',
        'reward_per_step',
        'arrival_slippage',
        'vwap_slippage',
        'oracle_slippage',
    ]

    df['step'] -= 1  # TODO: Off-by-one in our info update loop
    df['qty_left'] /= df['order_qty']
    df['time_left'] = (df['order_duration'] - df['step']) / df['order_duration']
    df['qty_filled'] = df['order_qty'] - df['qty_left']
    df['urgency'] = (
        df['qty_filled'] / df['order_qty'] * (1 - (df['step'] / df['order_duration']))
    )
    df = df.rename(
        {
            'config-env-reward_args-reward_type': 'reward_type',
            'config-env-reward_args-termination_px_cost_multiplier': 'terminal_penalty',
            'config-agent-agent_type': 'agent',
        },
        axis=1,
    )
    df['order_start_time'] = (
        pd.to_datetime('09:30:00') + pd.to_timedelta(df['order_start_time'], unit='s')
    ).dt.time
    df['reward_per_step'] = df['total_reward'] / df['step']
    df = df[keep_cols]
    train, test = df[df.split == 'train'].copy(), df[df.split == 'eval'].copy()
    train.drop('split', inplace=True, axis=1)
    test.drop('split', inplace=True, axis=1)
    return train, test


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


def save_latex_tables(
    latex_tables: Dict[str, str], artifacts_dir: pathlib.Path
) -> None:
    for table_name, table_latex in latex_tables.items():
        save_path = artifacts_dir / f'{table_name}.txt'
        with open(save_path, 'w') as f:
            f.write(table_latex)
        print(f'Performance Table saved to: {save_path}')


if __name__ == '__main__':
    main()
