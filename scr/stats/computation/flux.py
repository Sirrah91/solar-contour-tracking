import numpy as np

from scr.utils.types_alias import Contour, Contours, Masks, Mask, Stat
from scr.utils.filesystem import is_empty

from scr.geometry.contours.normalization import normalize_contour_input
from scr.geometry.contours.sampling import sample_map_at_contour, calc_arc_lengths
from scr.geometry.contours.densify import densify_contour

from scr.stats.computation.masks import overall_mask, corr_mask
from scr.stats.computation.utils import safe_call, nanaverage, weighted_std, safe_sum


def compute_flux_area_stats(
        image: np.ndarray,
        masks: Masks,
        shape: tuple[int, int],
        mu2d: np.ndarray | None = None,
        take_abs: bool = False
) -> Stat:
    """
    Compute flux statistics for a list of masks.

    Parameters
    ----------
    image : 2D array
        Map of intensity or magnetic field.
    masks : list of 2D arrays
        Filling-factor masks (values in [0,1]).
    shape : tuple of ints
        Just in case of no masks
    mu2d : 2D array, optional
        Map of cos(theta) for projection correction.
    take_abs : bool
        Whether to take absolute value of image before integration.

    Returns
    -------
    dict
        Flux statistics: total, mean, std, corrected_total, corrected_mean, corrected_std,
        plus per-mask lists.
    """

    def _process_mask(mask: Mask) -> tuple[float, float, float, float, float, float]:
        empty_entry = not np.any(mask)

        # uncorrected
        total = safe_call(safe_sum, empty_entry, values * mask)
        mean = safe_call(nanaverage, empty_entry, values, weights=mask)
        std = safe_call(weighted_std, empty_entry, values, mean, weights=mask)

        # corrected
        if mu2d is None:
            corr_total = corr_mean = corr_std = np.nan
        else:
            corr_weights = corr_mask(mask, mu2d=mu2d)

            corr_total = safe_call(safe_sum, empty_entry, values * corr_weights)
            corr_mean = safe_call(nanaverage, empty_entry, values, weights=corr_weights)
            corr_std = safe_call(weighted_std, empty_entry, values, corr_mean, weights=corr_weights)

        return total, mean, std, corr_total, corr_mean, corr_std

    values = np.abs(image) if take_abs else image

    # ---- Ensure mask list format ----
    if isinstance(masks, np.ndarray):
        masks = [masks]

    # ---- Containers for per-mask values ----
    totals, means, stds = [], [], []
    corr_totals, corr_means, corr_stds = [], [], []

    # ---- Process each mask individually ----
    for mask in masks:
        t, m, s, ct, cm, cs = _process_mask(mask)
        totals.append(t)
        means.append(m)
        stds.append(s)
        corr_totals.append(ct)
        corr_means.append(cm)
        corr_stds.append(cs)

    # global stats
    mask = overall_mask(masks, shape=shape)

    global_total, global_mean, global_std, global_corr_total, global_corr_mean, global_corr_std = _process_mask(
        mask
    )

    out = {
        f"flux_total": global_total,
        f"flux_mean": global_mean,
        f"flux_std": global_std,
        f"corrected_flux_total": global_corr_total,
        f"corrected_flux_mean": global_corr_mean,
        f"corrected_flux_std": global_corr_std,
        f"flux_total_list": totals,
        f"flux_mean_list": means,
        f"flux_std_list": stds,
        f"corrected_flux_total_list": corr_totals,
        f"corrected_flux_mean_list": corr_means,
        f"corrected_flux_std_list": corr_stds,
    }

    return out


def compute_flux_length_stats(
        image: np.ndarray,
        contours: Contours,
        lon2d: np.ndarray,
        lat2d: np.ndarray,
        rsun: float,
        mu2d: np.ndarray | None = None,
        min_step: float = 0.5,
        take_abs: bool = False
) -> Stat:
    """
    Compute flux statistics for a list of borders.

    Parameters
    ----------
    image : 2D array
        Map of intensity or magnetic field.
    contours : list of (N_i, 2) arrays or a single (N, 2) array
        Contour vertex coordinates in pixel units. Empty contours are ignored.
        A list of contours is interpreted as multiple disjoint border segments.
    lon2d, lat2d : ndarray (ny, nx)
        Heliographic longitude and latitude of each pixel (in degrees).
    rsun : float
        Solar radius in pixels.
    mu2d : 2D array, optional
        Map of cos(theta) for projection correction.
    min_step : float, optional
        Minimum step between contour vertices.
    take_abs : bool
        Whether to take absolute value of image before integration.

    Returns
    -------
    dict
        Flux statistics: total, mean, std, corrected_total, corrected_mean, corrected_std,
        plus per-mask lists.
    """

    def _process_contour(contour: Contour | Contours) -> tuple[float, float, float, float, float, float]:
        # Determine if we have a list of contours (Contours)
        if isinstance(contour, list):
            contour_list = [c for c in contour if not is_empty(c)]
        else:
            contour_list = [contour] if not is_empty(contour) else []

        # Check empty entry
        empty_entry = (len(contour_list) == 0)

        if empty_entry:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan  # solve np.concatenate of []

        values_on_contour = np.concatenate([sample_map_at_contour(contour=c, data_map=values, interp=True)
                                            for c in contour_list])
        arc_lengths = np.concatenate(
            [calc_arc_lengths(c, lon2d=lon2d, lat2d=lat2d, rsun=rsun) for c in contour_list])

        # uncorrected
        total = safe_call(safe_sum, empty_entry, values_on_contour * arc_lengths)
        mean = safe_call(nanaverage, empty_entry, values_on_contour, weights=arc_lengths)
        std = safe_call(weighted_std, empty_entry, values_on_contour, mean, weights=arc_lengths)

        # corrected
        if mu2d is None:
            corr_total = corr_mean = corr_std = np.nan
        else:
            weights = np.concatenate([1. / sample_map_at_contour(contour=c, data_map=mu2d, interp=True)
                                      for c in contour_list])

            corr_total = safe_call(safe_sum, empty_entry, values_on_contour * weights * arc_lengths)
            corr_mean = safe_call(nanaverage, empty_entry, values_on_contour, weights=weights * arc_lengths)
            corr_std = safe_call(weighted_std, empty_entry, values_on_contour, corr_mean,
                                  weights=weights * arc_lengths)

        return total, mean, std, corr_total, corr_mean, corr_std

    values = np.abs(image) if take_abs else image

    # ---- Ensure contour list format ----
    contours = normalize_contour_input(contours)

    # ---- Densify each contour ONCE ----
    dense_contours = []
    for c in contours:
        if is_empty(c):
            dense_contours.append(c)  # keep empty
        else:
            dense_contours.append(densify_contour(c, min_step=min_step))

    # ---- Containers for per-mask values ----
    totals, means, stds = [], [], []
    corr_totals, corr_means, corr_stds = [], [], []

    # ---- Process each mask individually ----
    for contour in dense_contours:
        t, m, s, ct, cm, cs = _process_contour(contour)
        totals.append(t)
        means.append(m)
        stds.append(s)
        corr_totals.append(ct)
        corr_means.append(cm)
        corr_stds.append(cs)

    # global stats
    global_total, global_mean, global_std, global_corr_total, global_corr_mean, global_corr_std = _process_contour(
        contour=dense_contours
    )

    out = {
        f"border_flux_total": global_total,
        f"border_flux_mean": global_mean,
        f"border_flux_std": global_std,
        f"corrected_border_flux_total": global_corr_total,
        f"corrected_border_flux_mean": global_corr_mean,
        f"corrected_border_flux_std": global_corr_std,
        f"border_flux_total_list": totals,
        f"border_flux_mean_list": means,
        f"border_flux_std_list": stds,
        f"corrected_border_flux_total_list": corr_totals,
        f"corrected_border_flux_mean_list": corr_means,
        f"corrected_border_flux_std_list": corr_stds,
    }

    return out
