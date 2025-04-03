import pathlib
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import yaml  # type: ignore

from trade_rl.util.path import get_root_dir


@dataclass(slots=True)
class MetaArgs:
    log_file_path: Optional[str] = field(metadata={'help': 'Path to log file.'})
    global_seed: int = field(default=1337, metadata={'help': 'Random seed.'})

    def __post_init__(self) -> None:
        if self.log_file_path is not None:
            self.log_file_path = str(get_root_dir() / self.log_file_path)


@dataclass(slots=True)
class OrderGenArgs:
    sym_spec: List[str]
    qty_spec: List[int]
    end_time_spec: List[int]


@dataclass(slots=True)
class EnvironmentArgs:
    env_name: str = field(metadata={'help': 'Gymanasium registered environment name'})
    max_global_steps: int = field(metadata={'help': 'Total number of steps to run for'})
    order_gen_args: OrderGenArgs = field(metadata={'help': 'Order generator params'})

    def __post_init__(self) -> None:
        self.order_gen_args = OrderGenArgs(**self.order_gen_args)  # type: ignore


def parse_args(config_yaml: str | pathlib.Path) -> Tuple[Any, ...]:
    config_dict = yaml.safe_load(pathlib.Path(config_yaml).read_text())

    meta_args = MetaArgs(**config_dict['MetaArgs'])
    env_args = EnvironmentArgs(**config_dict['EnvironmentArgs'])
    return (meta_args, env_args)
