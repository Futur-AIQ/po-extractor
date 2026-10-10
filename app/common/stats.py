"""Small statistics helpers shared by evaluation and the service metrics."""


def percentile(values: list[float], q: float) -> float | None:
    """q-th percentile (0-100) with linear interpolation; None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)
