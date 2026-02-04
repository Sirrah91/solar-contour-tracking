import numpy as np
from typing import Callable

from scr.utils.filesystem import is_empty


def nanaverage(
        array: np.ndarray,
        weights: np.ndarray | None = None
) -> float:
    if weights is None:
        return np.nan

    mask = np.isfinite(array) & np.isfinite(weights)
    if not np.any(mask):
        return np.nan

    w = weights[mask]
    if np.nansum(w) == 0.0:
        return np.nan

    return float(np.average(array[mask], weights=w))


def weighted_std(
        array: np.ndarray,
        mean: np.ndarray,
        weights: np.ndarray | None = None
) -> float:
    if weights is None or np.nansum(weights) == 0. or not np.isfinite(mean):
        return np.nan

    return np.sqrt(nanaverage(array=(array - mean) ** 2., weights=weights))


def safe_call(
        func: Callable,
        empty_case: bool,
        *args,
        n_outputs: int = 1,
        default: float = np.nan,
        **kwargs
) -> float | tuple[float, ...]:
    """Call func(*args, **kwargs), return default if empty_case=True."""
    if empty_case:
        if n_outputs == 1:
            return default
        return tuple(default for _ in range(n_outputs))
    result = func(*args, **kwargs)
    try:
        return float(result)
    except TypeError:
        try:
            return tuple(map(float, result))
        except TypeError:
            return result


def safe_sum(x: list | np.ndarray) -> float:
    return float(np.nan) if is_empty(x) or np.all(np.isnan(x)) else float(np.nansum(x))
