import pandas as pd
from typing import Literal

from scr.utils.types_alias import SunspotsPhasesByObservation
from scr.config.filtering import gimme_filtering_kwargs

from scr.io.datasets import load_contours_and_df_stat

from scr.stats.dataframe.filtering import filter_combined_df


def load_filtered_phase_tracks(
        nosuffix_filename: str,
        mode: Literal["sunspots", "pores", "all_sunspots", "all_pores"],
        drop_unknown: bool = True,
) -> tuple[SunspotsPhasesByObservation, pd.DataFrame]:
    """
    Load contour phase tracks and apply standard filtering.

    Returns
    -------
    contours_phases : dict
        Nested contour structure per observation.
    combined_df : pandas.DataFrame
        Filtered metadata table.
    """
    contours_phases, combined_df = load_contours_and_df_stat(nosuffix_filename)

    filtering_kwargs = gimme_filtering_kwargs(mode=mode)
    combined_df = filter_combined_df(
        combined_df,
        filtering_kwargs=filtering_kwargs,
    )

    if drop_unknown and "phase" in combined_df:
        combined_df = combined_df[combined_df["phase"] != "unknown"]

    return contours_phases, combined_df
