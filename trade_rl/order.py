from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True, frozen=True)
class Order:
    sym: Literal['TSLA']
    qty: int
    end_time: int
