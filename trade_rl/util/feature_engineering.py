from trade_rl.order import Order


# Incremental features
def update_ma(prev_average: float, new_value: float, n: int) -> float:
    return (prev_average * (n - 1) + new_value) / n


def update_ema(prev_average: float, new_value: float, alpha: float) -> float:
    return alpha * new_value + (1 - alpha) * prev_average


# Agent features
def get_vleft_norm(remaining_qty: float, order: Order) -> float:
    return remaining_qty / order.qty


def get_tleft_norm(current_step: int, order: Order) -> float:
    return (order.end_time - current_step) / order.end_time


# Market features
def getPriceReturn(previous_price: float, current_price: float) -> float:
    return (current_price - previous_price) / previous_price


def linear_schedule(start: float, end: float, duration: float, t: int) -> float:
    slope = (end - start) / duration
    return max(slope * t + start, end)
