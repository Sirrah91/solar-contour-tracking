import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from os import path
from typing import Literal

from scr.config.paths import PATH_CONTOURS_PHASES, PATH_FIGURES
from scr.config.quantities import get_quantity_spec
from scr.config.figures import SAVEFIG_KWARGS, FIG_FORMAT

from scr.utils.filesystem import check_dir
from scr.io.parquet import load_parquet
from scr.stats.aggregations import lifetime_and_mean

from scr.plotting.generic.scatter import plot_scatter
from scr.plotting.style.latex import latex_style


def main():
    """
    Scatter plot of mean magnetic quantity versus sunspot lifetime.
    """
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    QUANTITY: Literal["Ic", "B", "Bp", "Bt", "Br", "Bver", "Bhor"] = "Bhor"
    spec = get_quantity_spec(QUANTITY)
    where = spec.location.split(" ")[0]

    figure_outdir = path.join(PATH_FIGURES, "paper_plots")
    check_dir(figure_outdir)

    # ------------------------------------------------------------------
    # Load data and compute aggregations
    # ------------------------------------------------------------------
    df = load_parquet(
        path.join(PATH_CONTOURS_PHASES, "all_sunspots_phases.parquet")
    )

    # Lifetime (hours) and corresponding mean value per object
    lifetime, mean_values = lifetime_and_mean(
        df,
        value_col=spec.mean_col,
    )

    del df  # free memory

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    with latex_style(fontsize=20):
        fig, ax = plt.subplots(figsize=(8, 6))

        plot_scatter(
            ax,
            lifetime,
            mean_values,
        )

        ax.set_xlabel("Lifetime (h)")
        ax.set_ylabel(spec.ylabel_mean)

        plt.tight_layout()

        fig.savefig(
            path.join(figure_outdir, f"lifetime_{QUANTITY}_{where}.{FIG_FORMAT}"),
            format=FIG_FORMAT,
            **SAVEFIG_KWARGS,
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
