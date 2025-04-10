from __future__ import annotations

import logging
import pathlib
from abc import ABC, abstractmethod
from typing import Any, Dict

from trade_rl.env import TradingEnvironment


class TradingAgent(ABC):
    def __init__(self, env: TradingEnvironment) -> None:
        self.logger = logging.Logger(self.__class__.__name__)
        self.env = env

        self.logger.debug('Initialized agent')

    @abstractmethod
    def get_action(self, obs: Any) -> int:
        raise NotImplementedError

    def update(
        self,
        obs: Any,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: Any,
        info: Dict[str, Any],
    ) -> None: ...

    def save_model(self, path: str | pathlib.Path) -> None: ...
    def load_model(self, path: str | pathlib.Path) -> None: ...

    def linear_schedule(self, start: float, end: float, t: int) -> float:
        slope = (end - start) / self.env.max_train_steps
        return max(slope * t + start, end)
