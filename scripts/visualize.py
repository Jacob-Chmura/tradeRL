import argparse
from dataclasses import asdict

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import gridspec

from trade_rl.agents import agent_from_env
from trade_rl.data import Data
from trade_rl.util.args import AgentArgs, parse_args
from trade_rl.util.logging import setup_basic_logging
from trade_rl.util.multi_agent import MultiAgentTradingEnv
from trade_rl.util.path import get_root_dir


def collect_episode(args, data, agents, weights):
    num_agents = len(agents)
    agent_args_list = [
        AgentArgs(
            agent_type=agent_type,
            dqn_args=asdict(args.agent.dqn_args),
            reinforce_args=asdict(args.agent.reinforce_args),
        )
        for agent_type in agents
    ]
    env = MultiAgentTradingEnv(args, data, num_agents=num_agents)
    loaded_agents = []
    for i, e in enumerate(env.envs):
        agent = agent_from_env(e, agent_args_list[i])
        if agents[i] == 'dqn':
            agent.load_model(weights['path_dqn'])
        elif agents[i] == 'reinforce':
            agent.load_model(weights['path_reinforce'])
        loaded_agents.append(agent)

    obs, _ = env.reset()
    df = env.envs[0].day_data
    done = [False] * num_agents

    stats = [
        {
            'step': [],
            'oracle_slips': [],
            'vwap_slips': [],
            'qtys_left': [],
            'portfolio_avg': [0] * 600,
        }
        for _ in range(num_agents)
    ]

    t = 0
    while not all(done):
        actions = [agent.get_action(obs_i) for agent, obs_i in zip(loaded_agents, obs)]
        obs, _, done, _, infos = env.step(actions)
        for i, info in enumerate(infos):
            stats[i]['step'].append(t)
            stats[i]['oracle_slips'].append(info['oracle_slippage'])
            stats[i]['vwap_slips'].append(info['vwap_slippage'])
            stats[i]['qtys_left'].append(info['qty_left'])
            stats[i]['portfolio_avg'].append(
                np.mean([v[0] for v in info['portfolio']]) if info['portfolio'] else 0
            )
        t += 1
    final_infos = [e.info for e in env.envs]
    return df, final_infos, stats


def animate_trading(day_data, infos, agents, stats, interval=1):
    window_pad = 600
    # ignore, for debugging
    for i, info in enumerate(infos):
        print(agents[i], info.take_actions)

    all_actions_seconds = []
    marker_styles = ['o', 'X', 'P', '*', 'v', 'x', 'D']
    markers = []

    for i, info in enumerate(infos):
        action_seconds = set(info.order_start_time + s for s in info.take_actions)
        day_data[f'action_{i}'] = day_data['market_second'].isin(action_seconds)
        start_plot = max(0, info.order_start_time - window_pad)
        end_plot = min(info.order_start_time + info.order_duration + window_pad, 23400)
        all_actions_seconds.append(action_seconds)

    day_data = day_data[
        day_data['market_second'].between(start_plot, end_plot)
    ].reset_index(drop=True)

    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.75)
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(gs[0])
    bar_ax = fig.add_subplot(gs[1])
    fig.subplots_adjust(wspace=0.25, right=0.9, bottom=0.3)

    ax.axvline(
        x=start_plot + window_pad,
        linestyle=':',
        color='gray',
        linewidth=1,
        label='Order start/end',
        zorder=0,
    )
    ax.axvline(
        x=end_plot - window_pad, linestyle=':', color='gray', linewidth=1, zorder=0
    )
    (line,) = ax.plot([], [], lw=2, label='Close Price', zorder=1)
    for i in range(len(infos)):
        style = marker_styles[i]
        (marker_line,) = ax.plot(
            [],
            [],
            linestyle='',
            marker=style,
            label=f'{agents[i]}',
            alpha=0.7,
            markersize=1 + len(infos) - i,
        )
        markers.append(marker_line)

    name_texts, qleft_texts, oracle_texts, vwap_texts = [], [], [], []
    top_margin = 0.95
    bottom_margin = 0.05
    usable = top_margin - bottom_margin
    block_h = usable / len(infos)
    line_spacing = block_h / 4
    for i in range(len(infos)):
        block_top = top_margin - i * block_h

        name_y = block_top
        q_y = block_top - line_spacing
        orc_y = block_top - 2 * line_spacing
        vwap_y = block_top - 3 * line_spacing

        name = ax.text(
            1.02,
            name_y,
            '',
            transform=ax.transAxes,
            fontweight='bold',
            ha='left',
            va='top',
        )
        qtxt = ax.text(1.02, q_y, '', transform=ax.transAxes, ha='left', va='top')
        orct = ax.text(1.02, orc_y, '', transform=ax.transAxes, ha='left', va='top')
        vwpt = ax.text(1.02, vwap_y, '', transform=ax.transAxes, ha='left', va='top')

        name_texts.append(name)
        qleft_texts.append(qtxt)
        oracle_texts.append(orct)
        vwap_texts.append(vwpt)

    ax.set_xlim(day_data['market_second'].min(), day_data['market_second'].max())
    ax.set_ylim(day_data['close'].min(), day_data['close'].max())
    ax.set_xlabel('Market Second')
    ax.set_ylabel('Close Price ($)')
    ax.set_title('SPY')
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(infos) + 2,
        frameon=False,
    )

    x_pos = np.arange(len(infos))
    bars = bar_ax.bar(x_pos, [0] * len(infos), tick_label=agents, alpha=0.7)
    bar_ax.set_title('Avg. Buy Price')

    flat_avgs = []
    for i in range(len(infos)):
        flat_avgs += stats[i]['portfolio_avg']

    global_max = max(flat_avgs) + 0.05
    non_zero = [v for v in flat_avgs if v != 0]
    second_min = min(non_zero) - 0.05
    bar_ax.set_ylim(second_min, global_max)

    def init():
        line.set_data([], [])
        for marker_line in markers:
            marker_line.set_data([], [])
        for txt in name_texts + qleft_texts + oracle_texts + vwap_texts:
            txt.set_text('')
        for bar in bars:
            bar.set_height(0)
        return (
            [line]
            + markers
            + list(bars)
            + name_texts
            + qleft_texts
            + oracle_texts
            + vwap_texts
        )

    def update(frame):
        x = day_data['market_second'].iloc[:frame]
        y = day_data['close'].iloc[:frame]
        line.set_data(x, y)

        for i, marker_line in enumerate(markers):
            action_points = day_data.iloc[:frame][day_data[f'action_{i}'].iloc[:frame]]
            marker_line.set_data(action_points['market_second'], action_points['close'])

        for i in range(len(infos)):
            sec = day_data['market_second'].iloc[frame]
            start = infos[i].order_start_time
            idx = min(max(sec - start, 0), len(stats[i]['step']) - 1)

            q = stats[i]['qtys_left'][idx]
            os = stats[i]['oracle_slips'][idx]
            vs = stats[i]['vwap_slips'][idx]

            name_texts[i].set_text(f'{agents[i]}')
            qleft_texts[i].set_text(f'Quantity Left = {q}')
            oracle_texts[i].set_text(f'Oracle Slip = {os:.1f}bps')
            oracle_texts[i].set_color('green' if os < 0 else 'red')
            vwap_texts[i].set_text(f'VWAP Slip = {vs:.1f}bps')
            vwap_texts[i].set_color('green' if vs < 0 else 'red')

        for i, bar in enumerate(bars):
            idx = min(frame, len(stats[i]['portfolio_avg']) - 1)
            bar.set_height(stats[i]['portfolio_avg'][idx])

        return (
            [line]
            + markers
            + list(bars)
            + name_texts
            + qleft_texts
            + oracle_texts
            + vwap_texts
        )

    ani = animation.FuncAnimation(
        fig, update, frames=len(day_data), init_func=init, blit=False, interval=interval
    )
    return fig, ani


def main(args, agents, weights):
    data = Data(args.env.feature_args, train=False)
    day_data, infos, stats = collect_episode(args, data, agents, weights)
    fig, ani = animate_trading(day_data, infos, agents, stats, interval=1)
    plt.show()


if __name__ == '__main__':
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
    parser.add_argument(
        '--agents', nargs='+', type=str, help='A list of integers separated by spaces'
    )

    args_ = parser.parse_args()
    config_file_path = get_root_dir() / args_.config_file
    args = parse_args(config_file_path)
    setup_basic_logging(args.meta.log_file_path)
    # Custom config file for visualize including paths for extra 'weights' field
    with open(args_.config_file, 'r') as f:
        config = yaml.safe_load(f)

    main(args, args_.agents, config['weights'])
