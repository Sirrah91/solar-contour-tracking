import numpy as np
from skimage.draw import polygon2mask
from skimage.transform import rescale, resize

from scr.utils.types_alias import Contour, Contours, Mask

from scr.geometry.contours.normalization import normalize_contour_input
from scr.geometry.contours.area import contour_signed_area


def filling_factor_mask(
        contours: Contour | Contours,
        shape: tuple[int, int],
        oversample: int = 5
) -> Mask:
    contours = normalize_contour_input(contours)
    highres_shape = (shape[0] * oversample, shape[1] * oversample)

    # highres_contours = [contour * oversample for contour in contours]
    # mask_highres = nested_contours_to_mask(highres_contours, highres_shape).astype(float)

    mask_highres = np.zeros(highres_shape, dtype=float)
    for contour in contours:
        highres_contour = contour * oversample
        mask_highres += polygon2mask(highres_shape, highres_contour) * np.sign(contour_signed_area(contour))

    filling_factor = rescale(mask_highres, scale=1./oversample, anti_aliasing=True)

    if filling_factor.shape == shape:
        return filling_factor
    return resize(mask_highres, shape, anti_aliasing=True, preserve_range=True)


def subtract_filling_masks(
        mask1: Mask,
        mask2: Mask
) -> Mask:
    return np.clip(mask1 - mask2, a_min=0.0, a_max=1.0)
