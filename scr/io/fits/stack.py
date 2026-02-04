import numpy as np
from glob import glob
from os import path
from typing import Literal

from scr.utils.filesystem import is_empty

from scr.io.fits.read import load_image, load_fits_headers


def load_fits_stack(
        fits_dir_or_filename_list: str | list[str],
        quantity: Literal["Ic", "B", "Bp", "Bt", "Br", "Bver", "Bhor"] | int,
        regex: str = "*",
        allow_inhomogeneous_shape: bool = False,
        pad_to_homogeneous: bool = False,
        header_index: int = 0
) -> np.ndarray:
    if isinstance(fits_dir_or_filename_list, str) and path.isdir(fits_dir_or_filename_list):
        fits_all = sorted(glob(path.join(fits_dir_or_filename_list, regex)))
    else:
        fits_all = fits_dir_or_filename_list

    if is_empty(fits_all):
        raise ValueError("No FITS files found.")

    if allow_inhomogeneous_shape:
        if pad_to_homogeneous:
            headers = load_fits_headers(fits_all, regex=regex, header_index=header_index)
            shapes = []
            for hdr in headers:
                naxis = hdr.get("NAXIS", 0)
                if naxis < 2:
                    raise ValueError("FITS file does not contain 2D image data.")
                shape = (hdr["NAXIS2"], hdr["NAXIS1"])
                shapes.append(shape)

            max_height = max(h for h, _ in shapes)
            max_width = max(w for _, w in shapes)
            result = np.full((len(fits_all), max_height, max_width), np.nan, dtype=np.float32)

            for i, fname in enumerate(fits_all):
                data = load_image(fname, quantity)
                h, w = data.shape
                result[i, :h, :w] = data
        else:
            data_list = [load_image(fname, quantity) for fname in fits_all]

            # Try to stack if possible
            try:
                result = np.stack(data_list, axis=0)
            except ValueError:
                # Create an object array explicitly and assign manually to avoid broadcasting issues
                result = np.empty(len(data_list), dtype=object)
                for i, arr in enumerate(data_list):
                    result[i] = arr

    else:
        data = load_image(fits_all[0], quantity)
        ref_shape = data.shape
        result = np.zeros((len(fits_all), *ref_shape), dtype=np.float32)

        for i, fname in enumerate(fits_all[1:]):
            data = load_image(fname, quantity)
            if data.shape != ref_shape:
                raise ValueError(f"Inhomogeneous shape detected: {data.shape} != {ref_shape}")
            result[i] = data

    return result
