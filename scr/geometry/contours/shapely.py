from scr.utils.types_alias import Contour, Contours

from scr.geometry.contours.utils import contour_to_shape


def contour_area_shapely(
        contour: Contour,
        hole_contours: Contours | None = None,
        close: bool = True
) -> float:
    """
    Compute the geometric area of a contour using Shapely.

    Parameters:
        contour: Nx2 array of points.
        hole_contours: List of inner hole contours.
        close: Whether to enforce closed polygons.

    Returns:
        Area of the contour polygon. Returns 0 for non-closed shapes.
    """
    return contour_to_shape(contour=contour, holes=hole_contours, close=close).area


def contour_length_shapely(
        contour: Contour,
        close: bool = True
) -> float:
    """
    Compute the geometric perimeter/length of a contour using Shapely.

    Parameters:
        contour: Nx2 array of points.
        close: Whether to close the contours.

    Returns:
        Perimeter of the shape (polygon or line).
    """
    return contour_to_shape(contour=contour, close=close).length


def total_contours_area_shapely(
        contours: Contours,
        hole_contours: list[Contours] | None = None,
        close: bool = True
) -> float:
    """
    Compute total area from outer contours, optionally subtracting holes.

    Parameters:
        contours: List of outer boundary contours.
        hole_contours: List of lists of inner hole contours for each outer contour (or None).
        close: Whether to enforce closed polygons.

    Returns:
        Total geometric area.
    """
    if hole_contours is None:
        hole_contours = [[] for _ in contours]

    # Valid list of lists
    if not (isinstance(hole_contours, list) and all(isinstance(inner, list) for inner in hole_contours)):
        raise ValueError(
            "hole_contours parameter must be a list of lists of inner hole contours for each outer contour")

    return sum(contour_area_shapely(contour=contour, hole_contours=holes, close=close)
               for contour, holes in zip(contours, hole_contours))


def total_contours_length_shapely(
        contours: Contours,
        close: bool = True
) -> float:
    """
    Compute total perimeter/length from a list of contours.

    Parameters:
        contours: List of Nx2 arrays representing contours.
        close: Whether to close the contours.

    Returns:
        Sum of the lengths of all contours.
    """
    return sum(contour_length_shapely(contour=contour, close=close) for contour in contours)
