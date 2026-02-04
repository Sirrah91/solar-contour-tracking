import numpy as np


def compute_Bhor(
        Bp: np.ndarray,
        Bt: np.ndarray
) -> np.ndarray:
    return np.sqrt((np.square(Bp) + np.square(Bt)))
