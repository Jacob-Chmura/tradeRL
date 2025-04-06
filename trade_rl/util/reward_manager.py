class RewardManager:
    def __init__(self, reward_type: str) -> None:
        self.reward_type = reward_type

    def __call__(self, env: 'TradingEnvironment', terminated: bool) -> float:  # type: ignore
        return 0 if terminated else -env.remaining_qty
