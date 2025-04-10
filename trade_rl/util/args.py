import pathlib
from dataclasses import dataclass
from typing import List, Optional

import yaml  # type: ignore

from trade_rl.util.path import get_root_dir


@dataclass(slots=True)
class MetaArgs:
    experiment_name: str
    log_file_path: Optional[str]
    global_seed: int
    num_runs: int

    def __post_init__(self) -> None:
        if self.log_file_path is not None:
            self.log_file_path = str(get_root_dir() / self.log_file_path)


@dataclass(slots=True)
class OrderGenArgs:
    sym_spec: List[str]
    qty_spec: List[int]
    start_time_spec: List[int]
    duration_spec: List[int]


@dataclass(slots=True)
class FeatureArgs:
    train_data_path: str
    test_data_path: str
    short_window: int
    medium_window: int
    long_window: int


@dataclass(slots=True)
class RewardArgs:
    reward_type: str
    termination_px_cost_multiplier: float


@dataclass(slots=True)
class EnvironmentArgs:
    env_name: str
    max_train_steps: int
    max_test_steps: int
    order_gen_args: OrderGenArgs
    feature_args: FeatureArgs
    reward_args: RewardArgs

    def __post_init__(self) -> None:
        self.order_gen_args = OrderGenArgs(**self.order_gen_args)  # type: ignore
        self.feature_args = FeatureArgs(**self.feature_args)  # type: ignore
        self.reward_args = RewardArgs(**self.reward_args)  # type: ignore


@dataclass(slots=True)
class DQNArgs:
    lr: float
    gamma: float
    batch_size: int
    buffer_size: int
    eps_start: float
    eps_end: float


@dataclass(slots=True)
class ReinforceArgs:
    lr: float
    gamma: float
    batch_size: int
    temp_start: float
    temp_end: float


@dataclass(slots=True)
class AgentArgs:
    agent_type: str
    dqn_args: DQNArgs
    reinforce_args: ReinforceArgs

    def __post_init__(self) -> None:
        self.dqn_args = DQNArgs(**self.dqn_args)  # type: ignore
        self.reinforce_args = ReinforceArgs(**self.reinforce_args)  # type: ignore


@dataclass(slots=True)
class Args:
    meta: MetaArgs
    env: EnvironmentArgs
    agent: AgentArgs


def parse_args(config_yaml: str | pathlib.Path) -> Args:
    config = yaml.safe_load(pathlib.Path(config_yaml).read_text())
    return Args(
        meta=MetaArgs(**config['MetaArgs']),
        env=EnvironmentArgs(**config['EnvironmentArgs']),
        agent=AgentArgs(**config['AgentArgs']),
    )
