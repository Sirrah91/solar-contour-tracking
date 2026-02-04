import numpy as np

from scr.utils.types_alias import Tracks, TrackID, FrameID

from scr.geometry.raster.mask import contours_to_mask
from scr.geometry.raster.containment import containment_ratio


def find_nested_tracks(
        tracks: Tracks,
        image_shapes: list[tuple[int, int]],
        min_containment: float = 0.8,
) -> set[tuple[TrackID, FrameID]]:
    """
    Identify penumbra track frames that are nested inside larger penumbrae
    in the same frame.

    Returns:
        Set of (track_id, frame_id) pairs to be removed.
    """
    to_remove: set[tuple[TrackID, FrameID]] = set()

    track_ids = list(tracks.keys())

    # All frames where at least one track is present
    all_frames = set(
        frame
        for history in tracks.values()
        for frame in history.keys()
    )

    for frame in all_frames:
        active_ids = [
            tid for tid in track_ids
            if frame in tracks[tid]
        ]

        if len(active_ids) < 2:
            continue

        # Build masks per track
        masks: dict[TrackID, list[np.ndarray]] = {
            tid: [
                contours_to_mask(contours, image_shapes[frame])
                for contours in tracks[tid][frame]
            ]
            for tid in active_ids
        }

        for i, id1 in enumerate(active_ids):
            for id2 in active_ids[i + 1:]:
                for mask1 in masks[id1]:
                    for mask2 in masks[id2]:
                        area1 = mask1.sum()
                        area2 = mask2.sum()

                        # Always treat smaller region as candidate for removal
                        if area1 < area2:
                            ratio = containment_ratio(mask1, mask2)
                            if ratio >= min_containment:
                                to_remove.add((id1, frame))
                        else:
                            ratio = containment_ratio(mask2, mask1)
                            if ratio >= min_containment:
                                to_remove.add((id2, frame))

    return to_remove
