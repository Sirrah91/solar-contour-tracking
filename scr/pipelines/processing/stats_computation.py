from typing import Literal

from scr.utils.types_alias import StatsByObject

from scr.io.fits.read import load_fits_headers
from scr.io.fits.stack import load_fits_stack
from scr.io.tracks import load_tracks_and_stats

from scr.stats.computation.evolution import compute_sunspot_statistics_evolution


def compute_stats_from_contours(
        contour_file: str,
        quantities: list[Literal["Ic", "B", "Bp", "Bt", "Br", "Bhor"]],
        stat_types: list[Literal["sunspots", "pores"]],
        header_index: int = 0,
        min_step: float = 0.5,
) -> tuple[dict, StatsByObject, dict]:
    """
    Returns: tracks, stats, metadata
    """
    tracks, _, metadata = load_tracks_and_stats(contour_file)
    metadata |= {
        "contour_path": contour_file,
        "quantities": quantities,
        "stat_types": stat_types,
        "header_index": header_index,
        "min_step": min_step,
    }
    headers = load_fits_headers(
        metadata["filename_list"],
        header_index=header_index
    )

    stats = {stype: {} for stype in stat_types}

    for quantity in quantities:
        print(f"Quantity: {quantity}")
        images = load_fits_stack(
            metadata["filename_list"],
            quantity,
            allow_inhomogeneous_shape=True
        )

        for stat_type in stat_types:
            print(f"Feature: {stat_type}")
            stats[stat_type][quantity] = compute_sunspot_statistics_evolution(
                sunspots=tracks[stat_type],
                images=images,
                headers=headers,
                min_step=min_step,
                take_abs=quantity in ["Bp", "Bt"]
            )

    return tracks, stats, metadata
