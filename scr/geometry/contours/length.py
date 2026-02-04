import numpy as np

from scr.utils.types_alias import Contour, Contours


def contour_length(
        contour: Contour,
        correction: np.ndarray | float = 1.
) -> float:
    """
    Compute length of a contour with optional per-point correction.

    Parameters
    ----------
    contour : (N,2) array
        Coordinates of the contour points (x,y or col,row)
    correction : float or (N,) array
        Multiplicative correction. If array, should be per-point. Each segment weight
        is taken as the average of the weights at the segment endpoints.

    Returns
    -------
    float
        Corrected contour length
    """
    if not np.isscalar(correction):
        correction = np.asarray(correction)
        correction = 0.5 * (correction[:-1] + correction[1:])

    return np.nansum(np.sqrt(np.nansum(np.diff(contour, axis=0) ** 2, axis=1)) * correction)


def total_contours_length(
        contours: Contours
) -> float:
    """
    Estimate the total length (perimeter) of a list of 2D contours.

    Parameters:
        contours: List of Nx2 arrays representing ordered (y, x) or (row, col) coordinates of contours.

    Returns:
        Total length of all contours combined.
    """
    return sum(contour_length(contour) for contour in contours)


def compute_contour_arc_lengths(
        lon_deg: np.ndarray,
        lat_deg: np.ndarray,
        rsun: float = 1.0
) -> np.ndarray:
    """
    Compute spherical arc-length weights Δs_i for a contour given by
    heliographic longitude and latitude.

    Parameters
    ----------
    lon_deg : array-like
        Heliographic longitudes (degrees), same length as lat_deg.
    lat_deg : array-like
        Heliographic latitudes (degrees), same length as lon_deg.
    rsun : float
        Solar radius in the same length units desired for Δs.
        (Metres, kilometres, or Mm  your choice.)

    Returns
    -------
    ds : ndarray
        Arc lengths between successive contour points.
        Length N, where ds[i] is the arc between point i and i+1,
        and ds[-1] is between last and first point (closed contour).
    """

    lon = np.radians(np.asarray(lon_deg, dtype=float))
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    n = len(lon)

    if n < 2:
        return np.zeros(n)

    # Differences with wrap-around at the end
    lon2 = np.roll(lon, -1)
    lat2 = np.roll(lat, -1)

    # Make longitudes continuous: minimal angular difference
    dlon = lon2 - lon
    dlon = (dlon + np.pi) % (2. * np.pi) - np.pi

    # Spherical law of cosines for arc-length angle
    # cos(dσ) = sin φ1 sin φ2 + cos φ1 cos φ2 cos Δλ
    cos_dsigma = (
            np.sin(lat) * np.sin(lat2) +
            np.cos(lat) * np.cos(lat2) * np.cos(dlon)
    )

    # Numerical safety
    cos_dsigma = np.clip(cos_dsigma, -1.0, 1.0)

    dsigma = np.arccos(cos_dsigma)  # angular distance in radians

    # Physical distance
    ds = rsun * dsigma

    return ds
