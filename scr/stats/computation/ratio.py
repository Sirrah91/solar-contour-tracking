import numpy as np

from scr.utils.types_alias import Stat


def compute_ratio_stats(
        umbra_stats: Stat,
        penumbra_stats: Stat
) -> Stat:
    area_umbra = umbra_stats["corrected_area"]
    area_penumbra = penumbra_stats["corrected_area"]

    length_umbra = umbra_stats["corrected_length"]
    length_penumbra = penumbra_stats["corrected_length"]

    return {
        "area": (
            float(area_penumbra / area_umbra)
            if area_umbra else np.nan
        ),
        "length": (
            float(length_penumbra / length_umbra)
            if length_umbra else np.nan
        ),
    }
