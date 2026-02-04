import numpy as np
import pandas as pd

from scr.utils.types_alias import Stat


def phase_duration_statistics(
        df: pd.DataFrame,
        value_col: str,
        duration_col: str = "phase_duration",
        percentiles=(95, 98),
) -> Stat:
    """
    Aggregate per (observation, sunspot, phase segment).
    """
    groups = df.groupby(
        ["observation_id", "sunspot_id", duration_col],
        observed=True,
    )

    duration = []
    max_val = []
    perc_vals = {p: [] for p in percentiles}

    for _, g in groups:
        duration.append(np.nanmax(g[duration_col]))
        vals = np.abs(g[value_col])

        max_val.append(np.nanmax(vals))
        for p in percentiles:
            perc_vals[p].append(
                np.nanpercentile(vals, p, method="median_unbiased")
            )

    out = {
        "duration": np.asarray(duration),
        "max": np.asarray(max_val),
    }
    for p in percentiles:
        out[f"p{p}"] = np.asarray(perc_vals[p])

    return out


def lifetime_and_mean(
        df: pd.DataFrame,
        value_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lifetime = number of valid frames.
    """
    groups = df.groupby(["observation_id", "sunspot_id"], observed=True)

    lifetime = []
    mean_val = []

    for _, g in groups:
        mask = np.isfinite(g[value_col])
        lifetime.append(mask.sum())
        mean_val.append(np.nanmean(g.loc[mask, value_col]))

    return np.asarray(lifetime), np.asarray(mean_val)
