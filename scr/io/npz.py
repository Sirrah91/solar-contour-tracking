import numpy as np
from numpy.lib.npyio import NpzFile


def load_npz(filename: str) -> NpzFile:
    """Load a NumPy .npz archive without interpreting contents."""
    return np.load(filename, allow_pickle=True)


def save_npz(filename: str, **kwargs) -> None:
    """Save a NumPy .npz archive without interpreting contents."""
    np.savez_compressed(filename, **kwargs, allow_pickle=True)
