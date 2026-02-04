from scr.utils.types_alias import StatsByObject
from scr.utils.filesystem import check_dir

from scr.io.npz import load_npz, save_npz


def load_tracks_and_stats(
        filename: str
) -> tuple[dict, StatsByObject, dict]:
    """
    Load track data, statistics, and metadata from a .npz file.

    Parameters:
        filename: Path to the saved .npz archive.

    Returns:
        Tuple of (tracks, stats, metadata) dictionaries.
    """
    data = load_npz(filename)
    return (
        data["tracks"].item(),
        data["stats"].item(),
        data["metadata"].item(),
    )


def save_tracks_and_stats(
        filename: str,
        tracks: dict,
        stats: StatsByObject,
        metadata: dict | None = None,
) -> None:
    """
    Save track data, statistics, and optional metadata to a compressed .npz file.

    Parameters:
        filename: File path to save the .npz archive.
        tracks: Dictionary of tracked contours.
        stats: Dictionary of statistics per track.
        metadata: Optional additional information (e.g., parameters).
    """
    check_dir(filename, is_file=True)

    save_npz(
        filename,
        tracks=tracks,
        stats=stats,
        metadata=metadata or {},
    )
