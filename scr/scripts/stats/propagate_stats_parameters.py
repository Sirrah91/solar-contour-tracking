from os import path
from glob import glob
from typing import Literal

from scr.config.paths import PATH_CONTOURS

from scr.io.tracks import load_tracks_and_stats, save_tracks_and_stats
from scr.io.fits.read import load_fits_headers, load_image

from scr.stats.computation.evolution import compute_sunspot_statistics_evolution
from scr.stats.postprocessing.propagation import propagate_stat_parameter


def main() -> None:
    QUANTITIY: Literal["Ic", "B", "Bp", "Bt", "Br", "Bver", "Bhor"] = "Bhor"
    PROPAGATED_PARAMETER = "fractal_dimensions"

    contour_files = sorted(glob(path.join(PATH_CONTOURS, "*.npz")))

    for contour_file in contour_files:
        tracks, stats, metadata = load_tracks_and_stats(contour_file)

        headers = load_fits_headers(metadata["filename_list"], header_index=0)
        images = [load_image(filename, quantity=QUANTITIY) for filename in metadata["filename_list"]]

        for mode in ["sunspots", "pores"]:
            quantity_stats = compute_sunspot_statistics_evolution(
                sunspots=tracks[mode],
                headers=headers,
                min_step=0.5,
                take_abs=QUANTITIY in ["Bp", "Bt"],
                images=images
            )

            stats.setdefault(mode, {})[QUANTITIY] = quantity_stats
            propagate_stat_parameter(
                stats[mode],
                source_quantity=QUANTITIY,
                target_quantities=["Ic", "B", "Bp", "Bt", "Br", "Bhor"],
                param=PROPAGATED_PARAMETER,
            )

        save_tracks_and_stats(
            filename=contour_file.replace(".npz", "_NEW.npz"),
            tracks=tracks,
            stats=stats,
            metadata=metadata
        )
