import numpy as np

from scr.utils.types_alias import Contour


def densify_contour(
        contour: Contour,
        min_step: float = 0.5
) -> Contour:
    """
    Densify a polyline without changing existing vertices.

    For each segment between contour[i] and contour[i+1], if the Euclidean
    distance is > min_step, insert extra points so that all sub-steps
    are <= min_step.

    Parameters
    ----------
    contour : (N, 2) array
        Polyline vertices (y, x) or (x, y).
    min_step : float
        Maximum allowed spacing between points.

    Returns
    -------
    dense : (M, 2) array
        Densified polyline.
    """
    new_points = [contour[0]]

    for p0, p1 in zip(contour[:-1], contour[1:]):
        seg = p1 - p0
        dist = np.hypot(seg[0], seg[1])

        if dist <= min_step:
            # no densification needed
            new_points.append(p1)
            continue

        # number of intervals; +1 ensures spacing <= min_step
        n_intervals = int(np.ceil(dist / min_step))

        # generate points including both endpoints
        dense_seg = np.linspace(p0, p1, n_intervals + 1)

        # skip the first point to avoid duplicating p0
        for pt in dense_seg[1:]:
            new_points.append(pt)

    return np.vstack(new_points)
