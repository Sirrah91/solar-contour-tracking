import numpy as np

from scr.utils.types_alias import Contours, Mask, Masks, Stat
from scr.utils.filesystem import is_empty

from scr.geometry.contours.sampling import sample_map_at_contour, calc_spherical_length
from scr.geometry.contours.fractal import fractal_dimension_mask
from scr.geometry.contours.length import contour_length
from scr.geometry.contours.area import contour_signed_area

from scr.stats.computation.masks import overall_mask, corr_mask
from scr.stats.computation.utils import safe_call


def compute_geometry_stats(
        contours: Contours,
        masks: Masks,
        masks_border: Masks,
        shape: tuple[int, int],
        mu2d: np.ndarray,
        lon2d: np.ndarray,
        lat2d: np.ndarray,
        rsun: float,
) -> Stat:
    """Compute geometric stats (areas, lengths, fractals) for a set of contours and masks."""
    empty_entry = is_empty(contours)
    total_mask = overall_mask(masks, shape=shape)
    total_mask_border = overall_mask(masks_border, shape=shape)

    lons1d = [sample_map_at_contour(contour=contour, data_map=lon2d, interp=True) for contour in contours]
    lats1d = [sample_map_at_contour(contour=contour, data_map=lat2d, interp=True) for contour in contours]

    # Fractal dimensions
    fractal_dims = [safe_call(fractal_dimension_mask, empty_entry, mask) for mask in masks_border]
    fractal_dim = safe_call(fractal_dimension_mask, empty_entry, total_mask_border)

    # Mask border lengths
    mask_lengths = [safe_call(np.nansum, empty_entry, mask) for mask in masks_border]
    mask_length = safe_call(np.nansum, empty_entry, total_mask_border)

    # Contour border lengths
    contour_lengths = [safe_call(contour_length, empty_entry, c) for c in contours]
    _contour_length = safe_call(np.nansum, empty_entry, contour_lengths)

    # Corrected border lengths
    corrected_lengths = [safe_call(calc_spherical_length, empty_entry, lon, lat, rsun)
                         for lon, lat in zip(lons1d, lats1d)]
    corrected_length = safe_call(np.nansum, empty_entry, corrected_lengths)

    # Mask areas
    mask_areas = [safe_call(np.nansum, empty_entry, mask) for mask in masks]
    mask_area = safe_call(np.nansum, empty_entry, total_mask)

    # Contour areas
    contour_areas = [safe_call(contour_signed_area, empty_entry, c) for c in contours]
    contour_area = safe_call(np.nansum, empty_entry, contour_areas)

    # Corrected areas
    corrected_areas = [safe_call(np.nansum, empty_entry, corr_mask(mask, mu2d=mu2d)) for mask in masks]
    corrected_area = safe_call(np.nansum, empty_entry, corr_mask(total_mask, mu2d=mu2d))

    # Counts and holes
    counts, holes = safe_call(count_components, empty_entry, contour_areas, n_outputs=2)

    return {
        # Areas
        "corrected_area": corrected_area,
        "contour_area": contour_area,
        "mask_area": mask_area,
        "corrected_area_list": corrected_areas,
        "contour_area_list": contour_areas,
        "mask_area_list": mask_areas,

        # Lengths
        "corrected_length": corrected_length,
        "contour_length": _contour_length,
        "mask_length": mask_length,
        "corrected_length_list": corrected_lengths,
        "contour_length_list": contour_lengths,
        "mask_length_list": mask_lengths,

        # Fractals
        "fractal_dimension": fractal_dim,
        "fractal_dimension_list": fractal_dims,

        # Counts
        "component_count": counts,
        "hole_count": holes,
    }


def count_components(areas: list[float]) -> tuple[int, int]:
    """
    Count components and holes based only on sign of area.
    Preserves contour order; only counts.
    """
    counts = sum(a >= 0. for a in areas)
    holes = sum(a < 0. for a in areas)
    return counts, holes


def compute_corrected_total_area(
        spot_mask: Mask,
        mu2d: np.ndarray
) -> dict[str, float]:
    return {
        "corrected_total_area": safe_call(np.nansum, not np.any(spot_mask), corr_mask(spot_mask, mu2d=mu2d))
    }
