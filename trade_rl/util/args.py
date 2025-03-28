import pathlib
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import yaml  # type: ignore

from trade_rl.util.path import get_root_dir


@dataclass(slots=True)
class MetaArguments:
    log_file_path: Optional[str] = field(metadata={'help': 'Path to log file.'})
    global_seed: int = field(default=1337, metadata={'help': 'Random seed.'})

    def __post_init__(self) -> None:
        if self.log_file_path is not None:
            self.log_file_path = str(get_root_dir() / self.log_file_path)


def parse_args(config_yaml: str | pathlib.Path) -> Tuple[Any, ...]:
    config_dict = yaml.safe_load(pathlib.Path(config_yaml).read_text())

    meta_args = MetaArguments(config_dict['MetaArguments'])
    return (meta_args,)
