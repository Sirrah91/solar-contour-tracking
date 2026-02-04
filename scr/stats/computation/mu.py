import numpy as np
from scipy.ndimage import center_of_mass

from scr.utils.types_alias import Contour, Mask, Stat
from scr.utils.filesystem import is_empty

from scr.geometry.contours.shapely import contour_to_shape
from scr.geometry.contours.sampling import sample_map_at_contour

from scr.stats.computation.utils import safe_call, nanaverage


def compute_mu_stats(
        mask: Mask,
        mu2d: np.ndarray,
        contour: Contour | None = None,
) -> Stat:
    mask_bin = mask > 0.5
    empty_entry = not np.any(mask)

    # centroid µ
    if not is_empty(contour):
        centroid_coords = np.array(
            contour_to_shape(contour).centroid.coords[0]
        ).reshape(-1, 2)
    elif not empty_entry:
        centroid_coords = center_of_mass(mask)
    else:
        return {"mu_centroid": np.nan, "mu_min": np.nan, "mu_max": np.nan, "mu_mean": np.nan}

    mu_centroid = float(
        sample_map_at_contour(centroid_coords, mu2d, interp=True)[0]
    )

    mu_mean = safe_call(nanaverage, empty_entry, mu2d, weights=mask)

    if mask_bin.any():
        mu_min = float(np.nanmin(mu2d[mask_bin]))
        mu_max = float(np.nanmax(mu2d[mask_bin]))
    else:
        mu_min = mu_max = np.nan

    return {
        "mu_centroid": mu_centroid,
        "mu_min": mu_min,
        "mu_max": mu_max,
        "mu_mean": mu_mean,
    }
