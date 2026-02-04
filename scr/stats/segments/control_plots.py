import numpy as np
from pwlf import PiecewiseLinFit
from matplotlib import pyplot as plt

from scr.config.figures import SAVEFIG_KWARGS

from scr.plotting.generic.lines import plot_line


def plot_flux_fit_control(
        *,
        t: np.ndarray,
        total_flux: np.ndarray,
        model: PiecewiseLinFit,
        outfile: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    plot_line(ax, t, total_flux,
              line_kwargs={"color": "blue", "label": "Data"})

    t_model = model.fit_breaks
    total_flux_model = model.predict(t_model)
    plot_line(ax, t_model, total_flux_model,
              line_kwargs={"marker": "o", "color": "red", "label": "PWLF"})

    ax.set_xlabel("Frame ID")
    ax.set_ylabel("Normalised Total Flux")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    fig.savefig(outfile, **SAVEFIG_KWARGS)
    plt.close(fig)
