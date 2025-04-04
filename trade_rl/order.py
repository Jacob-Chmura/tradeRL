from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Literal

from trade_rl.util.args import OrderGenArgs


@dataclass(slots=True, frozen=True)
class Order:
    order_id: str
    sym: Literal['TSLA']
    qty: int
    end_time: int


class OrderGenerator:
    def __init__(self, config: OrderGenArgs) -> None:
        self.config = config

    def __call__(self) -> Order:
        return Order(
            order_id=str(uuid.uuid4()),
            sym=random.choice(self.config.sym_spec),  # type: ignore
            qty=random.choice(self.config.qty_spec),
            end_time=random.choice(self.config.end_time_spec),
        )
