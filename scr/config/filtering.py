import numpy as np
from typing import Literal


def gimme_filtering_kwargs(
        mode: Literal["sunspots", "pores", "all_sunspots", "all_pores"]
) -> dict:
    if "all" not in mode:
        # filter out the invalid frames first
        filtering_kwargs = {
            "phase_duration": {
                "stats_key": "Ic",
                "min_value": 3.,
                "max_value": np.inf,
                "mode": "frame-wise"
            },
            "overall_mu_min": {
                "stats_key": "Ic",
                "min_value": 0.4,
                "max_value": 1.0,
                "mode": "frame-wise"
            },
        }
    else:
        filtering_kwargs = {}

    if "sunspots" in mode:
        filtering_kwargs |= {
            "overall": {
                "umbra_lifetime": {
                    "stats_key": "Ic",
                    "min_value": 30.,
                    "max_value": np.inf,
                    "mode": "all"
                },
            },
            "penumbra": {
                "corrected_area": {
                    "stats_key": "Ic",
                    "min_value": 5000.,
                    "max_value": np.inf,
                    "mode": "any"
                },
            },
        }

    elif "pores" in mode:
        filtering_kwargs |= {
            "overall": {
                "umbra_lifetime": {
                    "stats_key": "Ic",
                    "min_value": 8.,
                    "max_value": np.inf,
                    "mode": "all"
                },
            },
            "penumbra": {
                "corrected_area": {
                    "stats_key": "Ic",
                    "min_value": 0.,
                    "max_value": 1000.,
                    "mode": "all"
                },
            },
            "umbra": {
                "corrected_area": {
                    "stats_key": "Ic",
                    "min_value": 100.,
                    "max_value": np.inf,
                    "mode": "any"
                },
            },
        }

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Available options are 'sunspots', 'pores', 'all_sunspots', and 'all_pores'."
        )

    return filtering_kwargs
