from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True, frozen=True)
class Order:
    sym: Literal['TSLA']
    qty: int
    end_time: int


class OrderGenerator:
    def __init__(self) -> None:
        self.sym_spec = ['TSLA']
        self.qty_spec = [10, 50]
        self.end_time_spec = [100, 1000]

    def __call__(self) -> Order:
        return Order(
            sym=random.choice(self.sym_spec),  # type: ignore
            qty=random.choice(self.qty_spec),
            end_time=random.choice(self.end_time_spec),
        )
