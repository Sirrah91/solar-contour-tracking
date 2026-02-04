import numpy as np
from tqdm import tqdm
from typing import Sequence

from scr.utils.types_alias import Sunspots, Stats, Headers
from scr.utils.filesystem import is_empty

from scr.geometry.contours.sampling import sample_map_at_contour
from scr.geometry.contours.utils import contour_to_shape
from scr.geometry.solar.mu import compute_mu
from scr.geometry.solar.projection import pixel_to_lonlat

from scr.morphology.masks import compute_masks

from scr.stats.computation.geometry import compute_geometry_stats, compute_corrected_total_area
from scr.stats.computation.masks import overall_mask, corr_mask
from scr.stats.computation.flux import compute_flux_area_stats, compute_flux_length_stats
from scr.stats.computation.ratio import compute_ratio_stats
from scr.stats.computation.mu import compute_mu_stats
from scr.stats.computation.utils import nanaverage, safe_call


def compute_sunspot_statistics_evolution(
        sunspots: Sunspots,
        images: Sequence[np.ndarray],
        headers: Headers,
        min_step: float = 0.5,
        take_abs: bool = False
) -> Stats:
    """
    Compute geometric and intensity-based statistics for umbra and penumbra
    regions of sunspots over time. Includes projection correction using mu.

    Statistics include:
    - Raw (pixel-based) area and perimeter
    - Flux-related quantities (sum and mean) within region and on boundary
    - Projection-corrected flux quantities using 1/mu weighting
    - Corrected geometric area and length based on 1/mu correction factor

    Parameters:
        sunspots: Dictionary of contours: {sid: {"outer": {t: [...]}, "inner": {t: [...]}}}
        images: 3D array of (T, H, W), time series of intensity/field maps.
        headers: List of FITS headers, one per frame, used to compute mu map.
        min_step: Maximum distance between contour points.
        take_abs: Whether to take absolute value of the field before flux integration.

    Returns:
        Nested dictionary: {sid: {"penumbra": {t: {...}}, "umbra": {...}, "ratio": {...}, "overall": {...}}}
    """

    stats: Stats = {}

    for sid, group in tqdm(sunspots.items()):
        stats[sid] = {"penumbra": {}, "umbra": {}, "ratio": {}, "overall": {}}

        # frames where either inner or outer exists
        frames = sorted(set(group.get("outer", {}).keys()) | set(group.get("inner", {}).keys()))

        # Precompute lifetime (frames count) for fields
        penumbra_lifetime = len(set(group.get("outer", {}).keys()))
        umbra_lifetime = len(set(group.get("inner", {}).keys()))

        lifetime_stats = {
                "umbra_lifetime": umbra_lifetime,
                "penumbra_lifetime": penumbra_lifetime,
            }

        for t in frames:
            image, header = images[t], headers[t]
            shape = image.shape

            outer_contours = group.get("outer", {}).get(t, []) or []
            inner_contours = group.get("inner", {}).get(t, []) or []

            mu2D = compute_mu(header)
            lon2D, lat2D = pixel_to_lonlat(header)
            rsun = header["RSUN_OBS"] / header["CDELT1"]

            # --- Masks ---
            umbra_masks, umbra_masks_border = compute_masks(
                contours=inner_contours,
                shape=shape,
                mask_holes=None,
                dtype=np.float32
            )

            penumbra_masks, penumbra_masks_border = compute_masks(
                contours=outer_contours,
                shape=shape,
                mask_holes=overall_mask(umbra_masks, shape=shape, dtype=np.float32),
                dtype=np.float32
            )

            spot_mask = overall_mask(umbra_masks + penumbra_masks, shape=shape)

            # --- Geometric stats ---
            umbra_stats = compute_geometry_stats(
                contours=inner_contours,
                masks=umbra_masks,
                masks_border=umbra_masks_border,
                shape=shape,
                mu2d=mu2D,
                lon2d=lon2D,
                lat2d=lat2D,
                rsun=rsun
            )

            penumbra_stats = compute_geometry_stats(
                contours=outer_contours,
                masks=penumbra_masks,
                masks_border=penumbra_masks_border,
                shape=shape,
                mu2d=mu2D,
                lon2d=lon2D,
                lat2d=lat2D,
                rsun=rsun
            )

            # --- Flux stats ---
            umbra_stats.update(compute_flux_area_stats(
                image=image,
                masks=umbra_masks,
                shape=shape,
                mu2d=mu2D,
                take_abs=take_abs)
            )
            umbra_stats.update(compute_flux_length_stats(
                image=image,
                contours=inner_contours,
                lon2d=lon2D,
                lat2d=lat2D,
                rsun=rsun,
                mu2d=mu2D,
                min_step=min_step,
                take_abs=take_abs)
            )

            penumbra_stats.update(compute_flux_area_stats(
                image=image,
                masks=penumbra_masks,
                shape=shape,
                mu2d=mu2D,
                take_abs=take_abs)
            )
            penumbra_stats.update(compute_flux_length_stats(
                image=image,
                contours=outer_contours,
                lon2d=lon2D,
                lat2d=lat2D,
                rsun=rsun,
                mu2d=mu2D,
                min_step=min_step,
                take_abs=take_abs)
            )

            stats[sid]["penumbra"][t] = penumbra_stats
            stats[sid]["umbra"][t] = umbra_stats
            stats[sid]["ratio"][t] = compute_ratio_stats(
                umbra_stats=umbra_stats,
                penumbra_stats=penumbra_stats
            )
            stats[sid]["overall"][t] = (
                    compute_mu_stats(
                        mask=spot_mask,
                        mu2d=mu2D,
                        contour=outer_contours[0] if not is_empty(outer_contours) else None
                    )
                    | lifetime_stats
                    | compute_corrected_total_area(spot_mask=spot_mask, mu2d=mu2D)
            )

            # --- µ statistics ---
            spots_mask = overall_mask(umbra_masks, shape=shape) + overall_mask(penumbra_masks, shape=shape)
            spots_mask_bin = spots_mask > 0.5
            empty_entry = not spots_mask.any()

            # centroid µ
            if not is_empty(outer_contours):
                centroid_coords = np.array(contour_to_shape(outer_contours[0]).centroid.coords[0]).reshape(-1, 2)
                mu_centroid = float(sample_map_at_contour(centroid_coords, mu2D, interp=True)[0])
            else:
                mu_centroid = np.nan

            mu_mean = safe_call(nanaverage, empty_entry, mu2D, weights=spots_mask)

            if spots_mask_bin.any():
                mu_min = float(np.nanmin(mu2D[spots_mask_bin]))
                mu_max = float(np.nanmax(mu2D[spots_mask_bin]))
            else:
                mu_min = mu_max = np.nan

            stats[sid]["overall"][t] = {
                "umbra_lifetime": umbra_lifetime,
                "penumbra_lifetime": penumbra_lifetime,
                "corrected_total_area": safe_call(np.nansum, empty_entry, corr_mask(spots_mask, mu2d=mu2D)),
                "mu_centroid": mu_centroid,
                "mu_min": mu_min,
                "mu_max": mu_max,
                "mu_mean": mu_mean,
            }

    return stats
