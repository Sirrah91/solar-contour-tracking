import matplotlib.pyplot as plt
from contextlib import contextmanager


def latex_setup(
    fontsize: int = 12,
    use_tex: bool = True,
    amsmath: bool = True,
) -> None:
    """
    Configure matplotlib for LaTeX-style plotting.

    Parameters
    ----------
    fontsize : int
        Base font size.
    use_tex : bool
        Whether to enable LaTeX rendering.
    amsmath : bool
        Whether to include amsmath in the LaTeX preamble.
    """
    plt.rc("text", usetex=use_tex)

    if use_tex and amsmath:
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

    plt.rcParams.update({
        "font.size": fontsize,
    })


@contextmanager
def latex_style(
        fontsize: int = 12,
        use_tex: bool = True,
        amsmath: bool = True,
):
    old_rc = plt.rcParams.copy()
    latex_setup(fontsize, use_tex, amsmath)
    try:
        yield
    finally:
        plt.rcParams.update(old_rc)

