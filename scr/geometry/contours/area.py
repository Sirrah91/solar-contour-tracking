import numpy as np

from scr.utils.types_alias import Contour, Contours
from scr.utils.filesystem import is_empty


def contour_signed_area(
        contour: Contour,
        correction: np.ndarray | float = 1.
) -> float:
    """
    Compute the signed area of a contour with optional per-point correction.

    Parameters
    ----------
    contour : (N,2) array
        Ordered coordinates (row=y, col=x).
    correction : float or (N,) array
        Correction factor for each point. If array, per-point corrections are averaged per segment.

    Returns
    -------
    float
        Signed area, corrected.
    """

    x, y = contour[:, 1], contour[:, 0]

    if not np.isscalar(correction):
        correction = 0.5 * (correction + np.roll(correction, -1))  # average per segment

    # shoelace formula with correction applied per segment
    area = 0.5 * np.nansum(correction * (x * np.roll(y, 1) - y * np.roll(x, 1)))

    return area


def contour_area(
        contour: Contour,
        correction: np.ndarray | float = 1.
) -> float:
    return np.abs(contour_signed_area(contour, correction))


def total_contours_area(
        contours: Contours,
        hole_contours: Contours | None = None
) -> float:
    """
    Estimate the total area enclosed by a list of 2D contours using the shoelace formula.

    Parameters:
        contours: List of Nx2 arrays representing ordered (y, x) or (row, col) coordinates of polygons.
        hole_contours: List of inner hole contours.

    Returns:
        Total area of all contours combined.
    """
    if not is_empty(hole_contours):
        return sum(contour_signed_area(contour) for contour in contours) - sum(
            contour_signed_area(contour) for contour in hole_contours)

    return sum(contour_signed_area(contour) for contour in contours)
