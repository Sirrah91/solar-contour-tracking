import numpy as np
from typing import Literal, Callable

from scr.utils.types_alias import Stats


def extract_parameter_series(
        stats: Stats,
        which: Literal["penumbra", "umbra", "ratio", "overall"],
        param: str
) -> list[np.ndarray]:
    """
    Extract time series of a single parameter for each sunspot.

    Parameters:
        stats: Output of `compute_sunspot_statistics_evolution()`.
        which: Region type to extract from ("penumbra", "umbra", "ratio", or "overall").
        param: Parameter to extract.

    Returns:
        A list of 1D NumPy arrays, where each array contains values for one sunspot
        ordered by frame number.
    """
    return [
        np.array([stat[which][frame][param] for frame in sorted(stat[which])])
        for stat in stats.values()
    ]


def extract_parameter_series_with_frames(
        stats: Stats,
        which: Literal["penumbra", "umbra", "ratio", "overall"],
        param: str
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Extract time series of a single parameter along with frame indices for each sunspot.

    Parameters:
        stats: Output of `compute_sunspot_statistics_evolution()`.
        which: Region type to extract from ("penumbra", "umbra", "ratio", or "overall").
        param: Parameter to extract.

    Returns:
        A list of (frames, values) tuples  one per sunspot.
            frames: 1D array of frame indices (ints)
            values: 1D array of corresponding parameter values
    """
    return [
        (np.array(sorted(stat[which])),  # x = frame numbers
         np.array([stat[which][frame][param] for frame in sorted(stat[which])]))  # y = values
        for stat in stats.values()
    ]


def aggregate_parameter_across_sunspots(
        stats: Stats,
        func: Callable[[np.ndarray], float],
        which: Literal["penumbra", "umbra", "ratio", "overall"],
        param: str
) -> float:
    """
    Apply an aggregation function to a parameter collected across all sunspots and frames.

    Parameters:
        stats: Output of `compute_sunspot_statistics_evolution()`.
        func: A NumPy-compatible function that reduces a 1D array to a scalar (e.g. np.mean, np.sum).
        which: Region type to extract from ("penumbra", "umbra", "ratio", "overall").
        param: Parameter to aggregate.

    Returns:
        The aggregated result as a float. Returns NaN if no data is present.
    """
    values = np.concatenate(extract_parameter_series(stats=stats, which=which, param=param))
    return func(values) if values.size > 0 else float("nan")
