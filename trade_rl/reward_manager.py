from typing import Dict, Tuple

from trade_rl.util.args import RewardArgs


class RewardManager:
    _REWARD_TYPES = [
        'arrival_sparse',
        'arrival_dense',
        'vwap_sparse',
        'vwap_dense',
        'oracle',
    ]

    def __init__(self, env: 'TradingEnvironment', reward_args: RewardArgs) -> None:  # type: ignore
        reward_type = reward_args.reward_type.lower().strip()
        if reward_type not in RewardManager._REWARD_TYPES:
            raise ValueError(f'Unknown reward type: {reward_type}')

        self.env = env
        self.is_sparse = 'sparse' in reward_args.reward_type
        self.slippage_type = reward_type.split('_')[0]  # arrival_sparse -> arrival
        self.terminal_cost_multiplier = reward_args.termination_px_cost_multiplier

        self.arrival_benchmark = lambda: self.env.order_arrival['open']
        self.vwap_benchmark = lambda: self.env.previous['vwap']
        self.oracle_benchmark = (
            lambda: self.env.order_duration_market['vwap']
            .nsmallest(self.env.info.step)
            .mean()
        )

    def __call__(self, order_done: bool) -> Tuple[Dict[str, float], float]:
        slippages = {'arrival': 0.0, 'vwap': 0.0, 'oracle': 0.0}
        if len(self.env.info.portfolio):
            slippages['arrival'] = self.env.info.agent_vwap - self.arrival_benchmark()
            slippages['vwap'] = self.env.info.agent_vwap - self.vwap_benchmark()
            slippages['oracle'] = self.env.info.agent_vwap - self.oracle_benchmark()

        cost = 0 if self.is_sparse else slippages[self.slippage_type]
        if order_done and self.env.info.qty_left > 0:
            finish_px = self.terminal_cost_multiplier * self.env.current['high']
            cost += finish_px - self.env.order_arrival['open']
        return slippages, -cost
