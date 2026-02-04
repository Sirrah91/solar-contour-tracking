import numpy as np

from scr.tracks.tracking import track_contours
from scr.tracks.filtering import remove_clockwise_contours
from scr.tracks.normalization import remove_nested_tracks, relabel_tracks_by_lifetime

from scr.sunspots.association import associate_inner_outer_tracks


def track_and_merge_sunspots(
        images: np.ndarray,
        outer_level: float = 0.9,
        middle_level: float = 0.65,
        inner_level: float = 0.5,
        min_area: float = 5.,
        min_frames: int = 3,
        max_gap: int = 3,
        iou_threshold: float = 0.3,
        registration: bool = True,
        min_containment: float = 0.8
) -> dict:
    """
    Track and associate sunspots from image sequence, combining penumbrae and umbrae.

    Parameters:
        images: 3D array of shape (T, H, W), input image sequence.
            If None, projection correction is skipped.
        outer_level: Contour level for penumbrae (e.g., 0.9).
        middle_level: Contour level foe pores (e.g., 0.65)
        inner_level: Contour level for umbrae (e.g., 0.5).
        min_area: Minimum area threshold for contour inclusion (px).
        max_gap: Maximum number of frames to allow gap in tracking.
        iou_threshold: IoU threshold for tracking match.
        min_frames: Minimum lifetime (frames) for a contour to be kept.
        registration: If True, register previous image to current.
        min_containment: Minimum fraction of the smaller region that must be inside the larger one.

    Returns:
        Dictionary with:
            - "sunspots": nested dict with "outer" and "inner" contours
            - "outer_tracks": original penumbrae tracks
            - "inner_tracks": original umbrae tracks
            - "stats": dict of track statistics (if compute_stats)
    """
    # Track outer penumbrae
    outer_tracks = track_contours(
        images=images,
        level=outer_level,
        min_area=min_area,
        max_gap=max_gap,
        iou_threshold=iou_threshold,
        min_frames=min_frames,
        registration=registration
    )

    # Track inner umbrae
    inner_tracks = track_contours(
        images=images,
        level=inner_level,
        min_area=min_area,
        max_gap=max_gap,
        iou_threshold=iou_threshold,
        min_frames=min_frames,
        registration=registration
    )

    # Track inner pores
    middle_tracks = track_contours(
        images=images,
        level=middle_level,
        min_area=min_area,
        max_gap=max_gap,
        iou_threshold=iou_threshold,
        min_frames=min_frames,
        registration=registration
    )

    # Remove "inner" penumbrae (from lower to higher values); possibly is more general to previous correction
    ##### !!!! DANGEROUS !!!! ONLY APPLICABLE TO IC CONTOURS, WILL DESTROY B CONTOURS #####
    outer_tracks_filtered = relabel_tracks_by_lifetime(remove_clockwise_contours(tracks=outer_tracks))

    # Remove nested penumbrae
    outer_tracks_filtered = remove_nested_tracks(
        tracks=outer_tracks_filtered,
        image_shapes=[image.shape for image in images],
        min_containment=min_containment
    )

    # Associate inner with outer
    sunspots = associate_inner_outer_tracks(
        outer_tracks=outer_tracks_filtered,
        inner_tracks=inner_tracks,
        image_shapes=[image.shape for image in images],
        min_containment=min_containment
    )

    pores = associate_inner_outer_tracks(
        outer_tracks=outer_tracks_filtered,
        inner_tracks=middle_tracks,
        image_shapes=[image.shape for image in images],
        min_containment=min_containment
    )
    return {
        "sunspots": sunspots,
        "pores": pores,
        "outer_tracks": outer_tracks,
        "outer_tracks_filtered": outer_tracks_filtered,
        "middle_tracks": middle_tracks,
        "inner_tracks": inner_tracks
    }
