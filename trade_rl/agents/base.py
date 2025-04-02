from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from trade_rl.env import TradingEnvironment


class TradingAgent(ABC):
    def __init__(self, env: TradingEnvironment) -> None:
        self.logger = logging.Logger(self.__class__.__name__)
        self.env = env

        self.logger.debug('Initialized agent')

    @abstractmethod
    def get_action(self, obs: Any) -> int:
        raise NotImplementedError

    @abstractmethod
    def update(
        self, obs: Any, action: int, reward: float, terminated: bool, next_obs: Any
    ) -> None:
        raise NotImplementedError
