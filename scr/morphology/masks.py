import numpy as np

from scr.utils.types_alias import Contours, Mask, Masks
from scr.utils.filesystem import is_empty

from scr.geometry.raster.mask import nested_contours_to_mask

from scr.morphology.filling import filling_factor_mask, subtract_filling_masks


def compute_masks(
        contours: Contours,
        shape: tuple[int, int],
        mask_holes: Mask | None = None,
        dtype: type = np.float32,
) -> tuple[Masks, Masks]:
    """
    Build filling-factor masks and 1-pixel border masks for a list of contours.

    Parameters
    ----------
    contours : list of (N,2) arrays or None
        Contours in (row, col) coordinates. May be None or empty.
    shape : (H, W)
        Shape of the output masks.
    mask_holes : 2D array or None, optional
        If provided, each filling mask will have mask_holes subtracted
        (useful to remove umbra from penumbra). Should be same shape.
    dtype : numpy dtype, optional
        dtype for masks (default float32)

    Returns
    -------
    masks : list of 2D float arrays
        Filling-factor masks (values in [0,1])  one per contour (may be empty list).
    masks_border : list of 2D float arrays
        Border (1-px) masks  one per contour (may be empty list).
    """
    # treat None or empty list as empty
    if is_empty(contours):
        return [], []

    # build per-contour filling-factor masks
    masks = [filling_factor_mask(c, shape).astype(dtype) for c in contours]

    # optionally subtract holes / inner mask (mask_holes expected same shape)
    if not is_empty(mask_holes):
        masks = [subtract_filling_masks(m, mask_holes).astype(dtype) for m in masks]

    # border masks (one-pixel borders)
    masks_border = [
        nested_contours_to_mask(c, shape, border_only=True).astype(dtype) for c in contours
    ]

    return masks, masks_border
