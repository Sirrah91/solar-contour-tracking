from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class QuantitySpec:
    mean_col: str
    std_col: str
    latex: str
    unit: str
    location: str
    threshold: float

    @property
    def latex_mean(self) -> str:
        return rf"$\langle {self.latex[1:-1]} \rangle$"

    @property
    def latex_std(self) -> str:
        return rf"$\sigma_{{{self.latex[1:-1]}}}$"

    @property
    def ylabel_mean(self) -> str:
        if self.unit:
            return f"{self.latex_mean} ({self.unit}, {self.location})"
        else:
            return f"{self.latex_mean} ({self.location})"

    @property
    def ylabel_std(self) -> str:
        if self.unit:
            return f"{self.latex_std} ({self.unit}, {self.location})"
        else:
            return f"{self.latex_std} ({self.location})"


_QUANTITY_REGISTRY = {
    "Ic": QuantitySpec(
        mean_col="Ic_penumbra_corrected_border_flux_mean",
        std_col="Ic_penumbra_corrected_border_flux_std",
        latex=r"$I^\mathrm{c}/I^\mathrm{c}_\mathrm{QS}$",
        unit="",
        location="penumbra outer boundary",
        threshold=0.9,
    ),
    "B": QuantitySpec(
        mean_col="B_penumbra_corrected_border_flux_mean",
        std_col="B_penumbra_corrected_border_flux_std",
        latex=r"$B$",
        unit="G",
        location="penumbra outer boundary",
        threshold=605.1246290384582,
    ),
    "Bp": QuantitySpec(
        mean_col="Bp_penumbra_corrected_border_flux_mean",
        std_col="Bp_penumbra_corrected_border_flux_std",
        latex=r"$B_{\mathrm{p}}$",
        unit="G",
        location="penumbra boundary",
        threshold=0.0,
    ),
    "Bt": QuantitySpec(
        mean_col="Bt_penumbra_corrected_border_flux_mean",
        std_col="Bt_penumbra_corrected_border_flux_std",
        latex=r"$B_{\mathrm{t}}$",
        unit="G",
        location="penumbra boundary",
        threshold=0.0,
    ),
    "Bver": QuantitySpec(
        mean_col="Br_umbra_corrected_border_flux_mean",
        std_col="Br_umbra_corrected_border_flux_std",
        latex=r"$B_{\mathrm{ver}}$",
        unit="G",
        location="umbra boundary",
        threshold=0.0,
    ),
    "Bhor": QuantitySpec(
        mean_col="Bhor_penumbra_corrected_border_flux_mean",
        std_col="Bhor_penumbra_corrected_border_flux_std",
        latex=r"$B_{\mathrm{hor}}$",
        unit="G",
        location="penumbra outer boundary",
        threshold=599.659971407691,
    ),
}
_QUANTITY_REGISTRY["Br"] = _QUANTITY_REGISTRY["Bver"]


def get_quantity_spec(name: Literal["Ic", "B", "Bp", "Bt", "Br", "Bver", "Bhor"]) -> QuantitySpec:
    try:
        return _QUANTITY_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown quantity: {name}")
