def MovingAverage(prev_average: float, new_value: float, n: int) -> float:
    return (prev_average * (n - 1) + new_value) / n


def ExponentialMovingAverage(
    prev_average: float, new_value: float, alpha: float
) -> float:
    return alpha * new_value + (1 - alpha) * prev_average
