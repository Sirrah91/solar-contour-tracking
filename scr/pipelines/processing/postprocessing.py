import numpy as np
import pandas as pd
from os import path
from typing import Literal

from scr.config.paths import PATH_CONTOURS_PHASES, SLOPES_FILE

from scr.utils.types_alias import SunspotsPhasesByObservation
from scr.utils.filesystem import check_dir
from scr.utils.nested import nested_cast_arrays_dtype

from scr.io.npz import save_npz
from scr.io.parquet import save_parquet, load_parquet
from scr.io.tracks import load_tracks_and_stats

from scr.stats.dataframe.flatten import flatten_spot_features_with_frame
from scr.stats.dataframe.filtering import filter_combined_df
from scr.stats.segments.annotation import apply_segments_to_combined_df
from scr.stats.segments.collection import collect_slopes

from scr.postanalysis.phases import split_by_phase


def compute_phase_split(
        contour_files: list[str],
        mode: Literal["sunspots", "pores"] = "sunspots",
        collect_new_slopes: bool = False
) -> tuple[SunspotsPhasesByObservation, pd.DataFrame]:
    # --------------------------------------------------------------
    # 1) Load all tracks & stats
    # --------------------------------------------------------------

    print("Collecting statistics and contours...")

    all_stats: dict = {}
    all_contours: dict = {}
    all_filenames: dict = {}

    for contour_file in contour_files:
        tracks, stats, metadata = load_tracks_and_stats(contour_file)
        all_stats[contour_file] = stats[mode]
        all_contours[contour_file] = tracks[mode]
        all_filenames[contour_file] = metadata["filename_list"]

    # --------------------------------------------------------------
    # 2) Flatten statistics
    # --------------------------------------------------------------

    print("Flatten statistics...")

    combined_df = flatten_spot_features_with_frame(all_stats=all_stats)

    combined_df["image_path"] = [
        all_filenames[id_][frame]
        for id_, frame in zip(combined_df["observation_id"], combined_df["frame"])
    ]
    combined_df["image_path"] = combined_df["image_path"].astype("category")

    if collect_new_slopes or not path.isfile(SLOPES_FILE):
        # --------------------------------------------------------------
        # 3) Prefilter ONLY for slope fitting
        # --------------------------------------------------------------

        print("Slope fitting...")

        df_fit = filter_combined_df(
            df=combined_df,
            filtering_kwargs={
                "overall_mu_min": {"min_value": 0.15, "mode": "frame-wise"}
            }
        )

        df_fit = df_fit[
            [
                "observation_id",
                "sunspot_id",
                "spot_global_index",
                "frame",
                "Br_umbra_corrected_flux_total",
                "Br_penumbra_corrected_flux_total",
            ]
        ].copy()

        # --------------------------------------------------------------
        # 4) Fit slopes
        # --------------------------------------------------------------

        segments_df = collect_slopes(
            df=df_fit,
            control_plots=True,
        )
    else:
        print("Using precomputed slopes...")

        segments_df = load_parquet(SLOPES_FILE)

    # --------------------------------------------------------------
    # 5) Apply segments to combined_df
    # --------------------------------------------------------------

    print("Merging statistics and segments...")

    apply_segments_to_combined_df(
        combined_df=combined_df,
        segments_df=segments_df,
    )

    # --------------------------------------------------------------
    # 6) Phase assignment (forming / stable / decaying)
    # --------------------------------------------------------------

    print("Phase assignment...")

    master_column = "segment_slope"
    slope_threshold = 0.00225

    conditions = [
        np.isfinite(combined_df[master_column]) &
        (combined_df[master_column] > slope_threshold),

        np.isfinite(combined_df[master_column]) &
        (combined_df[master_column] <= slope_threshold) &
        (combined_df[master_column] >= -slope_threshold),

        np.isfinite(combined_df[master_column]) &
        (combined_df[master_column] < -slope_threshold),
    ]

    choices = ["forming", "stable", "decaying"]

    combined_df["phase"] = np.select(
        conditions,
        choices,
        default="unknown",
    )

    combined_df["phase"] = combined_df["phase"].astype("category")

    # --------------------------------------------------------------
    # 7) Split back by phase
    # --------------------------------------------------------------

    print("Splitting by phases...")

    contours_phases = split_by_phase(
        combined_df,
        all_contours,
    )

    contours_phases = nested_cast_arrays_dtype(contours_phases, dtype=np.float32)

    # --------------------------------------------------------------
    # 8) Final output
    # --------------------------------------------------------------

    print("Saving...")
    check_dir(PATH_CONTOURS_PHASES)
    filename = path.join(PATH_CONTOURS_PHASES, f"all_{mode}_phases")
    save_npz(filename=f"{filename}.npz", contours_phases=contours_phases)
    save_parquet(filename=f"{filename}.parquet", df=combined_df)

    return contours_phases, combined_df
