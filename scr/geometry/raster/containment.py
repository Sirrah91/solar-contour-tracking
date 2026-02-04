import numpy as np

from scr.utils.types_alias import Mask


def containment_ratio(
        mask_small: Mask,
        mask_large: Mask
) -> float:
    """
    Compute what fraction of the smaller mask is contained in the larger mask.

    Assumes masks are boolean arrays with the same shape.
    """
    intersection = np.logical_and(mask_small, mask_large).sum()
    area_small = mask_small.sum()

    return intersection / area_small if area_small > 0.0 else 0.0
