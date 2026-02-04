import numpy as np
from skimage.draw import polygon, polygon_perimeter
from skimage.morphology import thin

from scr.utils.types_alias import Contour, Contours, Mask

from scr.geometry.contours.normalization import normalize_contour_input
from scr.geometry.contours.area import contour_signed_area


def contours_to_mask(
        contours: Contour | Contours,
        shape: tuple[int, int],
        border_only: bool = False
) -> Mask:
    """
    Convert one or more contours into a combined binary mask.

    Parameters:
        contours: A single (N, 2) array or a list of such arrays, each representing a contour.
        shape: Shape of the output mask (height, width).
        border_only: If True, only mark the contour edge. If False, fill the interior.

    Returns:
        Binary mask of shape `shape`, with True where the contour(s) are marked.
    """
    contours = normalize_contour_input(contours)
    mask = np.zeros(shape, dtype=bool)

    for contour in contours:
        r, c = contour[:, 0], contour[:, 1]
        if border_only:
            rr, cc = polygon_perimeter(r, c, shape, clip=True)
        else:
            rr, cc = polygon(r, c, shape)
        mask[rr, cc] = True

    return mask


def nested_contours_to_mask(
        contours: Contour | Contours,
        shape: tuple[int, int],
        border_only: bool = False
) -> Mask:
    """
    Convert one or more contours into a combined binary mask.

    Parameters:
        contours: A single (N, 2) array or a list of such arrays, each representing a contour.
        shape: Shape of the output mask (height, width).
        border_only: If True, only mark the contour edge. If False, fill the interior.

    Returns:
        Binary mask of shape `shape`, with True where the contour(s) are marked.
    """
    contours = normalize_contour_input(contours)

    areas = [contour_signed_area(c) for c in contours]

    # Sort by descending absolute area (largest first)
    order = np.argsort([-abs(a) for a in areas])
    contours = [contours[i] for i in order]
    areas = [areas[i] for i in order]

    mask = np.zeros(shape, dtype=bool)

    for i, contour in enumerate(contours):
        r, c = contour[:, 0], contour[:, 1]
        if border_only:
            rr, cc = polygon_perimeter(r, c, shape, clip=True)
            mask[rr, cc] = True
        else:
            rr, cc = polygon(r, c, shape)
            mask[rr, cc] = areas[i] > 0  # CCW = True, CW = False

    if border_only:
        return thin(mask)
    return mask
