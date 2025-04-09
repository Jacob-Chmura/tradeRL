import numpy as np

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
        self.terminal_cost_multiplier = reward_args.termination_px_cost_multiplier

        if 'arrival' in reward_type:
            self.benchmark = lambda: self.env.order_data['open'].iloc[
                self.env.start_index
            ]
        elif 'vwap' in reward_type:
            self.benchmark = lambda: self.env.order_data['vwap'].iloc[
                self.env.start_index + self.env.episode_step - 1
            ]
        elif 'oracle' in reward_type:
            self.benchmark = (
                lambda: self.env.order_data['vwap']
                .nsmallest(self.env.episode_step)
                .mean()
            )

    def __call__(self, terminated: bool) -> float:
        cost = self.get_slippage()
        if terminated:
            cost += self._get_terminal_cost()
        return -cost

    def get_slippage(self) -> float:
        if self.is_sparse or not len(self.env.portfolio):
            return 0
        agent_vwap = np.mean([x[0] for x in self.env.portfolio])
        return agent_vwap - self.benchmark()

    def _get_terminal_cost(self) -> float:
        if self.env.remaining_qty == 0:
            return 0
        px = self.env.order_data['high'].iloc[
            self.env.start_index + self.env.episode_step
        ]
        px *= self.terminal_cost_multiplier  # Additional cost for unfinished qty

        # Just use arrival here for simplicity
        return px - self.env.order_data['open'].iloc[self.env.start_index]
