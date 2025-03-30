from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from trade_rl.util.args import OrderGenArgs


@dataclass(slots=True, frozen=True)
class Order:
    sym: Literal['TSLA']
    qty: int
    end_time: int


class OrderGenerator:
    def __init__(self, config: OrderGenArgs) -> None:
        self.config = config

    def __call__(self) -> Order:
        return Order(
            sym=random.choice(self.config.sym_spec),  # type: ignore
            qty=random.choice(self.config.qty_spec),
            end_time=random.choice(self.config.end_time_spec),
        )
