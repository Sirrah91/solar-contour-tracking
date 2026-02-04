import numpy as np
from shapely.geometry import Polygon, Point, LineString

from scr.utils.types_alias import Contour, Contours
from scr.utils.filesystem import is_empty

from scr.geometry.contours.normalization import close_contour


def get_intersecting_pairs(
        contours1: Contours,
        contours2: Contours
) -> np.ndarray:
    """
    Return index pairs of contours that intersect.

    Returns:
        An array of shape (N, 2) with indices (i, j) where contours1[i] intersects contours2[j].
    """
    intersecting_indices = [
        (i1, i2)
        for i1, c1 in enumerate(contours1)
        for i2, c2 in enumerate(contours2)
        if do_contours_intersects(c1, c2)
    ]
    return np.array(intersecting_indices, dtype=int)


def group_by_first_index(
        values1: list,
        values2: list,
        index_pairs: np.ndarray
) -> tuple[list, list, np.ndarray]:
    """
    Group items in values2 by the first index in `index_pairs`, and return matching items from values1.

    Parameters:
        values1: List to filter once per unique first index (e.g. contours1).
        values2: List to group based on matching values1 (e.g. contours2).
        index_pairs: Array of shape (N, 2) with (i, j) index pairs for intersecting items.

    Returns:
        - Filtered values1 for each unique i (no duplicates).
        - Grouped lists of values2 corresponding to each i.
        - Filtered index_pairs (only one per i in values1).
    """
    unique_i1, first_idx = np.unique(index_pairs[:, 0], return_index=True)
    grouped_values2 = [
        np.array(values2, dtype=object)[index_pairs[:, 0] == i]
        for i in unique_i1
    ]
    filtered_values1 = list(np.array(values1, dtype=object)[unique_i1])
    return filtered_values1, grouped_values2, index_pairs[first_idx]


def contour_to_shape(
        contour: Contour,
        holes: Contours | None = None,
        close: bool = True
) -> Polygon | LineString | Point:
    """
    Convert a contour and optional holes into a shapely shape.

    Parameters:
        contour: Nx2 array of (y, x) coordinates for the outer boundary.
        holes: Optional list of Nx2 arrays defining holes.
        close: If True, ensure contours are closed polygons.

    Returns:
        Shapely geometry (Polygon, LineString, or Point).
    """
    if close:
        contour = close_contour(contour)
        holes = [close_contour(h) for h in holes] if not is_empty(holes) else []

    if len(contour) == 1:
        return Point(contour[0])
    if len(contour) < 3:
        return LineString(contour)

    try:
        poly = Polygon(shell=contour, holes=holes)
        return poly if poly.is_valid else poly.buffer(0)
    except Exception:
        return Polygon(contour).convex_hull


def do_contours_intersects(
        contour1: Contour,
        contour2: Contour
) -> bool:
    """
    Return True if two contours intersect.
    """
    return contour_to_shape(contour1).intersects(contour_to_shape(contour2))
