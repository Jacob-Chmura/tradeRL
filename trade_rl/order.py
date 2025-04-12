from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Literal

from trade_rl.util.args import OrderGenArgs


@dataclass(slots=True)
class Order:
    order_id: str
    start_time: int
    duration: int
    sym: Literal['TSLA']
    qty: int


class OrderGenerator:
    def __init__(self, config: OrderGenArgs) -> None:
        self.config = config

    def __call__(self) -> Order:
        duration = random.choice(self.config.duration_spec)
        return Order(
            order_id=str(uuid.uuid4()),
            start_time=random.choice(self.config.start_time_spec),
            duration=duration,
            sym=random.choice(self.config.sym_spec),  # type: ignore
            qty=int(duration * random.choice(self.config.qty_spec)),
        )
