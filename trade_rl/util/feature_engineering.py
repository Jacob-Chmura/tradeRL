from trade_rl.order import Order


# Incremental features
def updateMovingAverage(prev_average: float, new_value: float, n: int) -> float:
    return (prev_average * (n - 1) + new_value) / n


def updateExponentialMovingAverage(
    prev_average: float, new_value: float, alpha: float
) -> float:
    return alpha * new_value + (1 - alpha) * prev_average


# Agent features
def getQuanityLeftNormalized(remaining_qty: float, order: Order) -> float:
    return remaining_qty / order.qty


def getTimeLeftNormalized(current_step: int, order: Order) -> float:
    return (order.end_time - current_step) / order.end_time


# Market features
def getPriceReturn(previous_price: float, current_price: float) -> float:
    return (current_price - previous_price) / previous_price
