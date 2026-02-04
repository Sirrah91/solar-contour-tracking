import numpy as np
from skimage.transform import EuclideanTransform
from tqdm import tqdm

from scr.utils.types_alias import Tracks

from scr.geometry.raster.mask import contours_to_mask
from scr.geometry.contours.area import contour_area
from scr.geometry.contours.filtering import filter_contours_by_area
from scr.geometry.contours.extraction import find_contours

from scr.tracks.filtering import filter_tracks_by_lifetime
from scr.tracks.normalization import relabel_tracks_by_lifetime
from scr.tracks.matching import compute_iou, warp_contour, register_images_pairwise


def track_contours(
        images: np.ndarray,
        level: float,
        min_area: float = 5.,
        max_gap: int = 3,
        iou_threshold: float = 0.3,
        min_frames: int = 3,
        registration: bool = True,
        area_ratio_bounds: tuple[float, float] = (0.5, 2.0)
) -> Tracks:
    """
    Track contours across frames using IoU and image registration.

    Parameters:
        images: Array of shape (T, H, W), one image per time step.
        level: Contour level to extract.
        min_area: Minimum area (px) to consider a contour.
        max_gap: Max frames to look back for matches.
        iou_threshold: Minimum IoU to consider a match.
        min_frames: Minimum lifetime to keep a track.
        registration: If True, register previous image to current.
        area_ratio_bounds: (min_ratio, max_ratio) to reject mismatched areas early.

    Returns:
        Dictionary of tracks: {track_id: {frame_index: [contours]}}
    """

    tracks = {}
    next_id = 0
    registration_cache = {}  # Cache image pair registrations to avoid recomputation
    rmin, rmax = area_ratio_bounds

    for t, image in enumerate(tqdm(images, desc="Tracking")):
        # Step 1: Extract and filter contours
        contours = filter_contours_by_area(find_contours(image, level), threshold_min=min_area)

        # Step 2: Sort by area to improve matching consistency
        contours = sorted(contours, key=contour_area, reverse=True)
        assigned = [False] * len(contours)

        # Step 3: Attempt to match with previous contours
        for tid, hist in tracks.items():
            for dt in range(1, max_gap + 1):
                t_prev = t - dt
                if t_prev < 0 or t_prev not in hist:
                    continue

                pair_key = (t_prev, t)
                # Check cache or compute only once per image pair
                if pair_key not in registration_cache:
                    # Register image[t_prev] to image[t] once
                    registration_cache[pair_key] = register_images_pairwise(
                        img_source=images[t_prev].astype(np.float32),
                        img_target=image
                    ) if registration else EuclideanTransform()
                transform = registration_cache[pair_key]

                # Sort previous contours by area
                prev_contours = sorted(hist[t_prev], key=contour_area, reverse=True)

                for prev_c in prev_contours:
                    warped_prev_c = warp_contour(prev_c, transform)
                    prev_mask = contours_to_mask(warped_prev_c, image.shape)

                    prev_area = contour_area(prev_c)  # Part of the early area ratio check
                    for i, c in enumerate(contours):
                        if assigned[i]:
                            continue

                        # Early area ratio check
                        area_ratio = contour_area(c) / prev_area
                        if not (rmin <= area_ratio <= rmax):
                            continue

                        mask = contours_to_mask(c, image.shape)
                        if compute_iou(prev_mask, mask) >= iou_threshold:
                            hist.setdefault(t, []).append(c)
                            assigned[i] = True
                            break  # contour c assigned
                    if any(assigned):
                        break
                if any(assigned):
                    break

        # Step 4: Create new tracks for unmatched contours
        for i, c in enumerate(contours):
            if not assigned[i]:
                tracks[next_id] = {t: [c]}
                next_id += 1

    # Step 5: Keep only long-enough tracks
    return relabel_tracks_by_lifetime(filter_tracks_by_lifetime(tracks=tracks, min_lifetime=min_frames))
