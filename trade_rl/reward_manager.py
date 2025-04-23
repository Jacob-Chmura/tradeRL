from typing import Dict, Tuple

from trade_rl.util.args import RewardArgs


class RewardManager:
    _REWARD_TYPES = [
        'arrival_sparse',
        'arrival_dense',
        'vwap_sparse',
        'vwap_dense',
    ]

    def __init__(self, env: 'TradingEnvironment', reward_args: RewardArgs) -> None:  # type: ignore
        reward_type = reward_args.reward_type.lower().strip()
        if reward_type not in RewardManager._REWARD_TYPES:
            raise ValueError(f'Unknown reward type: {reward_type}')

        self.env = env
        self.is_sparse = 'sparse' in reward_args.reward_type
        self.slippage_type = reward_type.split('_')[0]  # arrival_sparse -> arrival
        self.terminal_cost_multiplier = reward_args.termination_px_cost_multiplier
        self.benchmarks = {
            'arrival': lambda: self._slippage_bps(self.env.order_arrival['close']),
            'vwap': lambda: self._slippage_bps(self.env.previous['vwap_close']),
            'oracle': lambda: self._slippage_bps(
                self.env.order_duration_market['close']
                .nsmallest(self.env.info.order_qty)
                .mean(),
            ),
        }

    def __call__(self, order_done: bool) -> Tuple[Dict[str, float], float]:
        slippages = {}
        for name, benchmark in self.benchmarks.items():
            slippages[name] = benchmark() if len(self.env.info.portfolio) else 0.0
        cost = 0 if self.is_sparse else slippages[self.slippage_type]
        if order_done and self.env.info.qty_left > 0:
            finish_px = self.terminal_cost_multiplier * self.env.current['high']
            cost += finish_px - self.env.order_arrival['open']
        return slippages, -cost

    def _slippage_bps(self, benchmark_px: float) -> float:
        return 10_000 * (self.env.info.agent_vwap - benchmark_px) / benchmark_px
