import numpy as np

from scr.utils.types_alias import Mask, Masks
from scr.utils.filesystem import is_empty


def overall_mask(
        masks: Masks,
        shape: tuple[int, int],
        dtype: type = np.float32
) -> Mask:
    """Combine a list of masks into a single clipped mask."""
    if not is_empty(masks):
        stacked = np.nansum(np.stack(masks, axis=0), axis=0)
        return np.clip(stacked, 0.0, 1.0).astype(dtype)
    else:
        return np.zeros(shape=shape, dtype=dtype)


def corr_mask(
        mask: Mask,
        mu2d: np.ndarray
) -> Mask:
    corrected_mask = np.zeros_like(mask, dtype=float)
    valid = np.isfinite(mask) & np.isfinite(mu2d) & (mu2d != 0)
    corrected_mask[valid] = mask[valid] / mu2d[valid]

    return corrected_mask
