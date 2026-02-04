import numpy as np
from matplotlib import pyplot as plt

from scr.config.figures import SAVEFIG_KWARGS

from scr.plotting.generic.lines import plot_line


def plot_fractal_dimension_control(
        *,
        logs_eps: np.ndarray,
        logs_N: np.ndarray,
        logs_N_fit: np.ndarray,
        fractal_dim: float,
        outfile: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    plot_line(ax, logs_eps, logs_N,
              line_kwargs={"marker": "o", "label": "Box counts (observed)"})
    plot_line(ax, logs_eps, logs_N_fit,
              line_kwargs={"label": rf"Fit line → $D \approx {fractal_dim:.4f}$"})

    ax.set_xlabel(r"$\log(1 / \mathrm{box\ size})$")
    ax.set_ylabel(r"$\log(N_\mathrm{boxes})$")
    ax.set_title("Box-Counting Fractal Dimension")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    fig.savefig(outfile, **SAVEFIG_KWARGS)
    plt.close(fig)
