import pathlib
from dataclasses import dataclass, field
from typing import List, Optional

import yaml  # type: ignore

from trade_rl.util.path import get_root_dir


@dataclass(slots=True)
class MetaArgs:
    experiment_name: str = field(metadata={'help': 'Name of experiment'})
    log_file_path: Optional[str] = field(metadata={'help': 'Path to log file.'})
    global_seed: int = field(default=1337, metadata={'help': 'Random seed.'})

    def __post_init__(self) -> None:
        if self.log_file_path is not None:
            self.log_file_path = str(get_root_dir() / self.log_file_path)


@dataclass(slots=True)
class OrderGenArgs:
    sym_spec: List[str]
    qty_spec: List[int]
    start_time_spec: List[int]
    end_time_spec: List[int]


@dataclass(slots=True)
class EnvironmentArgs:
    env_name: str = field(metadata={'help': 'Gymanasium registered environment name'})
    max_train_steps: int = field(metadata={'help': 'Total train steps to run for'})
    max_test_steps: int = field(metadata={'help': 'Total test steps to run for'})
    order_gen_args: OrderGenArgs = field(metadata={'help': 'Order generator params'})

    def __post_init__(self) -> None:
        self.order_gen_args = OrderGenArgs(**self.order_gen_args)  # type: ignore


@dataclass(slots=True)
class DQNArgs:
    lr: float = field(metadata={'help': 'Learning rate'})
    gamma: float = field(metadata={'help': 'Discount factor'})
    batch_size: int = field(metadata={'help': 'Batch size from replay buffer'})
    buffer_size: int = field(metadata={'help': 'Max experience replay buffer size'})
    eps: float = field(metadata={'help': 'Epsilon random exploration probability'})


@dataclass(slots=True)
class ReinforceArgs:
    lr: float = field(metadata={'help': 'Learning rate'})
    gamma: float = field(metadata={'help': 'Discount factor'})
    batch_size: int = field(metadata={'help': 'Batch size from replay buffer'})
    temp: float = field(metadata={'help': 'Softmax temperature for exploration'})


@dataclass(slots=True)
class AgentArgs:
    agent_type: str = field(metadata={'help': 'Agent type to use'})
    dqn_args: DQNArgs = field(metadata={'help': 'DQN Agent args'})
    reinforce_args: ReinforceArgs = field(metadata={'help': 'Reinforce Agent args'})

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
