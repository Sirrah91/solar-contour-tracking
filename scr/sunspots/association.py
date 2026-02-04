from scr.utils.types_alias import Tracks, Sunspots
from scr.utils.collections import nested_defaultdict

from scr.geometry.raster.mask import contours_to_mask
from scr.geometry.raster.containment import containment_ratio


def associate_inner_outer_tracks(
        outer_tracks: Tracks,
        inner_tracks: Tracks,
        image_shapes: list[tuple[int, int]],
        min_containment: float = 0.8
) -> Sunspots:
    """
    Associate inner contours (e.g. umbrae) with outer contours (e.g. penumbrae) across frames.

    Parameters:
        outer_tracks: Track dictionary for outer features.
        inner_tracks: Track dictionary for inner features.
        image_shapes: List of shape of the images, needed for masks.
        min_containment: Minimum fraction of the smaller region that must be inside the larger one.

    Returns:
        A new dictionary:
        {
            outer_id: {
                "outer": {frame_idx: [contours]},
                "inner": {frame_idx: [contours_inside]}
            },
            ...
        }
    """

    print("Associating inner contours with outer contours across frames...")

    merged = {}

    for outer_id, outer_data in outer_tracks.items():
        merged[outer_id] = {"outer": outer_data, "inner": nested_defaultdict(depth=1, factory=list)}

        for t, outer_contours in outer_data.items():
            image_shape = image_shapes[t]
            # Collect all inner contours at this frame
            for inner_id, inner_data in inner_tracks.items():
                if t not in inner_data:
                    continue
                for inner_contour in inner_data[t]:
                    inner_mask = contours_to_mask(inner_contour, image_shape)
                    for outer_contour in outer_contours:
                        outer_mask = contours_to_mask(outer_contour, image_shape)
                        ratio = containment_ratio(inner_mask, outer_mask)
                        if ratio >= min_containment:
                            merged[outer_id]["inner"][t].append(inner_contour)
                            break  # stop after first match

    return merged
