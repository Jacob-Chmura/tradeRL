from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Literal

from trade_rl.util.args import OrderGenArgs


@dataclass(slots=True, frozen=True)
class Order:
    order_id: str
    start_time: int
    end_time: int
    sym: Literal['TSLA']
    qty: int


class OrderGenerator:
    def __init__(self, config: OrderGenArgs) -> None:
        self.config = config

    def __call__(self) -> Order:
        return Order(
            order_id=str(uuid.uuid4()),
            # data doesn't always adhere to 1s time steps, less than 23400s in a trading day,
            start_time=random.choice(self.config.start_time_spec),
            end_time=random.choice(self.config.end_time_spec),
            sym=random.choice(self.config.sym_spec),  # type: ignore
            qty=random.choice(self.config.qty_spec),
        )
