import argparse
import json
import pathlib
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

NUMERIC_COLS = [
    "global_step", "episode", "step",
    "order_start_time", "order_duration", "order_qty",
    "qty_left", "total_reward",
    "agent_vwap", "arrival_slippage", "vwap_slippage",
    "oracle_slippage", "time",
]

def parse_results_dir(results_dir: pathlib.Path) -> pd.DataFrame:
    dfs = []
    for result_dir in results_dir.iterdir():
        print(f'Reading results from: {result_dir}')
        with open(result_dir / 'config.json') as f:
            config = json.load(f)

        train_results = pd.read_csv(result_dir / 'train_results.csv')
        train_results['split'] = 'train'
        results = train_results

        results = results[results["global_step"] != "global_step"]
        existing = [c for c in NUMERIC_COLS if c in results.columns]
        results.loc[:, existing] = results[existing].apply(pd.to_numeric)

        def _recurse_config(config: Dict[str, Any], prefix: str = 'config') -> None:
            for k, v in config.items():
                col_name = f'{prefix}-{k}'
                if isinstance(v, dict):
                    _recurse_config(v, col_name)
                elif isinstance(v, list):
                    results.loc[:, col_name] = ''.join([str(x) for x in v])
                else:
                    results.loc[:, col_name] = v

        _recurse_config(config)
        dfs.append(results)
    return pd.concat(dfs)


def preprocess_train(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[:, 'step'] -= 1
    df.loc[:, 'qty_left'] /= df['order_qty']
    df.loc[:, 'time_left'] = (df['order_duration'] - df['step']) / df['order_duration']
    df.loc[:, 'qty_filled'] = df['order_qty'] - df['qty_left']
    df.loc[:, 'urgency'] = (
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
    df.loc[:, 'reward_per_step'] = df['total_reward'] / df['step']
    return df[df.split == 'train'].copy()


def extract_agent_and_stock(run_dir: pathlib.Path) -> str:

    # Modify depending on used naming convention
    full = str(run_dir).split('/')[1].split('_')
    agent = full[0].upper()
    stock = full[-1].upper()
    px = full[-2]
    return f"{agent} | {stock} | penalty: {px}"


def plot_2x3_learning_curves(df1: pd.DataFrame, df2: pd.DataFrame,
                             run1_dir: pathlib.Path, run2_dir: pathlib.Path,
                             out_dir: pathlib.Path) -> None:
    roll = 1000
    smooth = lambda x: x.rolling(roll).mean().fillna(0)

    for df in (df1, df2):
        df.loc[:, 'arrival_slippage'] = smooth(df['arrival_slippage'])
        df.loc[:, 'vwap_slippage'] = smooth(df['vwap_slippage'])
        df.loc[:, 'reward_per_step'] = smooth(df['reward_per_step'])

    fig, ax = plt.subplots(
        2, 3, figsize=(12, 7), sharex='col'
    )

    name1 = extract_agent_and_stock(run1_dir)
    name2 = extract_agent_and_stock(run2_dir)

    def plot_row(df, row_idx, title):
        sns.lineplot(data=df, x='episode', hue='reward_type',
                     y='arrival_slippage', ax=ax[row_idx][0])
        sns.lineplot(data=df, x='episode', hue='reward_type',
                     y='vwap_slippage',    ax=ax[row_idx][1])
        sns.lineplot(data=df, x='episode', hue='reward_type',
                     y='reward_per_step',  ax=ax[row_idx][2])
        for j in range(3):
            a = ax[row_idx][j]
            a.spines[['top', 'right']].set_visible(False)
            a.grid(0.3)
            a.set_ylabel('' if j > 0 else ('Slippage (bps)' if j < 2 else 'Value'))
            a.set_xlabel('Episode' if row_idx == 1 else '')
            if row_idx == 0:
                a.set_title(['Arrival Slippage', 'VWAP Slippage', 'Reward / Step'][j])
            a.get_legend().remove()

        y_offset = 1.2 if row_idx == 0 else 1.1
        ax[row_idx][0].annotate(title, xy=(0, y_offset), xycoords='axes fraction',
                                fontsize=12, weight='bold')
    plot_row(df1, 0, name1)
    plot_row(df2, 1, name2)

    handles, labels = ax[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05), frameon=False, prop={'size': 14})

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / 'comparison.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Learning curves saved to: {save_path}')

    # Plot combined urgency curve across run1 and run2
    df1_copy = df1.copy()
    df2_copy = df2.copy()

    agent1 = run1_dir.parts[-2].upper()
    agent2 = run2_dir.parts[-2].upper()
    df1_copy['agent'] = agent1
    df2_copy['agent'] = agent2

    urgency_df = pd.concat([df1_copy, df2_copy])
    urgency_df['reward_type'] = urgency_df['reward_type'].astype(str).str.strip()
    allowed_reward_types = ['vwap_sparse', 'vwap_dense']
    urgency_df = urgency_df[urgency_df['reward_type'].isin(allowed_reward_types)]

    urgency_df['urgency'] = urgency_df['urgency'].rolling(roll).mean().fillna(0)
    urgency_df['label'] = urgency_df['agent'] + ' | ' + urgency_df['reward_type']

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.lineplot(
        data=urgency_df,
        x='episode',
        y='urgency',
        hue='label',
        ax=ax2
    )
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.grid(0.3)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('')
    ax2.set_title('Urgency on GTLB')

    # Legend under the plot
    ax2.legend(
        # title='Run',
        prop={'size': 12},
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False
    )

    urgency_save_path = out_dir / 'urgency_comparison.png'
    fig2.tight_layout()
    fig2.savefig(urgency_save_path, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f'Urgency comparison saved to: {urgency_save_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run1', required=True, help='First experiment (e.g. dqn/dqn_agent_spy_v2)')
    parser.add_argument('--run2', required=True, help='Second experiment (e.g. dqn/dqn_agent_tsla_v2)')
    parser.add_argument('--base-dir', default='runs', help='Base directory for experiment results')
    parser.add_argument('--output-dir', default='artifacts/comparison', help='Output directory for plot')
    args = parser.parse_args()

    run1_path = pathlib.Path(args.base_dir) / args.run1
    run2_path = pathlib.Path(args.base_dir) / args.run2

    df1 = preprocess_train(parse_results_dir(run1_path))
    df2 = preprocess_train(parse_results_dir(run2_path))

    plot_2x3_learning_curves(df1, df2, run1_path, run2_path, pathlib.Path(args.output_dir))


if __name__ == '__main__':
    main()
