from copy import deepcopy

from scr.utils.types_alias import Tracks, TrackID, FrameID

from scr.tracks.association import find_nested_tracks


def relabel_tracks_by_lifetime(
        tracks: Tracks
) -> Tracks:
    """
    Reassign track IDs sorted by descending lifetime (number of frames).

    Parameters:
        tracks: Dictionary of original tracks {old_id: {frame_idx: contours}}.

    Returns:
        A new dictionary with relabelled track IDs {new_id: ...}, where 0 is longest-lived.
    """
    # Sort by number of time steps (descending)
    sorted_items = sorted(tracks.items(), key=lambda item: len(item[1]), reverse=True)

    # Rebuild dictionary with new contiguous keys
    relabelled_tracks = {new_id: track for new_id, (_, track) in enumerate(sorted_items)}
    return relabelled_tracks


def remove_track_frames(
        tracks: Tracks,
        to_remove: set[tuple[TrackID, FrameID]],
) -> Tracks:
    """
    Remove specific (track_id, frame_id) entries from tracks.
    Empty tracks are removed.
    """
    cleaned = deepcopy(tracks)

    for track_id, frame in to_remove:
        if track_id not in cleaned:
            continue
        if frame in cleaned[track_id]:
            del cleaned[track_id][frame]
        if not cleaned[track_id]:
            del cleaned[track_id]

    return cleaned


def remove_nested_tracks(
    tracks: Tracks,
    image_shapes: list[tuple[int, int]],
    min_containment: float = 0.8,
) -> Tracks:
    """
    Remove penumbrae nested inside larger penumbrae and relabel tracks.
    """
    to_remove = find_nested_tracks(
        tracks,
        image_shapes,
        min_containment,
    )
    return cleanup_nested_tracks(tracks, to_remove)


def cleanup_nested_tracks(
        tracks: Tracks,
        to_remove: set[tuple[TrackID, FrameID]],
) -> Tracks:
    """
    Apply nested-track removals and normalise track IDs.
    """
    tracks = remove_track_frames(tracks, to_remove)
    return relabel_tracks_by_lifetime(tracks)
