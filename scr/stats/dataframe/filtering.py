import numpy as np
import pandas as pd
from typing import Literal

from scr.utils.filesystem import is_empty


def filter_combined_df(
        df: pd.DataFrame,
        filtering_kwargs: dict
) -> pd.DataFrame:
    """
    Filter a combined sunspot statistics DataFrame using flexible, hierarchical criteria.
    """

    group_cols = ["observation_id", "sunspot_id"]

    # Precompute group keys ONCE (important for speed & correctness)
    group_keys = df[group_cols].apply(tuple, axis=1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_column_name(part: str, param: str, stats_key: str | None = None) -> str:
        if part in {"overall", "ratio"}:
            return f"{part}_{param}"

        if "flux" in param or "variation" in param:
            if stats_key is None:
                raise ValueError(f"'stats_key' required for flux parameter '{param}'")
            return f"{stats_key}_{part}_{param}"

        return f"{part}_{param}"

    def _cell_satisfies(
            value,
            min_val: float | None = None,
            max_val: float | None = None,
            exact_val: float | str | None = None,
    ) -> bool:
        """
        Scalar or array-like test.
        - Scalars → treated as length-1 arrays
        - Arrays → ALL elements must satisfy
        - NaN / empty → False
        """
        if value is None:
            return False

        if isinstance(value, float) and np.isnan(value):
            return False

        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value)
            if is_empty(arr):
                return False
        else:
            arr = np.asarray([value])

        if exact_val is not None:
            return bool(np.all(arr == exact_val))

        cond = np.ones(arr.shape, dtype=bool)

        if min_val is not None:
            cond &= arr >= min_val
        if max_val is not None:
            cond &= arr <= max_val

        return bool(np.all(cond))

    def _scalar_satisfies(
            arr: np.ndarray,
            min_val: float | None = None,
            max_val: float | None = None,
            exact_val: float | str | None = None,
    ) -> np.ndarray:
        """Vectorised scalar-only version."""
        mask = np.isfinite(arr)

        if exact_val is not None:
            return mask & (arr == exact_val)

        if min_val is not None:
            mask &= arr >= min_val
        if max_val is not None:
            mask &= arr <= max_val

        return mask

    # ------------------------------------------------------------------
    # Core filtering logic
    # ------------------------------------------------------------------

    def _apply_filter(
            df: pd.DataFrame,
            column: str,
            mode: Literal["frame-wise", "any", "all"],
            min_val: float | None = None,
            max_val: float | None = None,
            exact_val: float | str | None = None,
    ) -> pd.DataFrame:

        series = df[column]

        # ---------------- Scalar fast path
        if pd.api.types.is_numeric_dtype(series):
            row_mask = _scalar_satisfies(
                series.to_numpy(),
                min_val=min_val,
                max_val=max_val,
                exact_val=exact_val,
            )
            row_mask = pd.Series(row_mask, index=series.index)

        # ---------------- Generic (arrays / objects)
        else:
            row_mask = series.apply(
                _cell_satisfies,
                min_val=min_val,
                max_val=max_val,
                exact_val=exact_val,
            )

        if row_mask.isna().any():
            raise ValueError(
                f"Non-boolean mask produced for column '{column}'. "
                f"Check NaNs or invalid cell values."
            )

        # ---------------- Frame-wise
        if mode == "frame-wise":
            return df[row_mask]

        # ---------------- Group-wise reduction
        if mode == "any":
            group_mask = row_mask.groupby(group_keys, observed=True).any()
        elif mode == "all":
            group_mask = row_mask.groupby(group_keys, observed=True).all()
        else:
            raise ValueError(f"Unknown mode '{mode}'")
        """
        if mode == "any":
            group_mask = row_mask.groupby(group_keys, observed=True).agg("any")
        elif mode == "all":
            group_mask = row_mask.groupby(group_keys, observed=True).agg(
                lambda x: x.any() and x.all()
            )
        else:
            raise ValueError(f"Unknown mode '{mode}'")
        """

        if group_mask.isna().any():
            raise ValueError(
                f"NaN encountered in group mask for column '{column}'."
            )

        keep_ids = group_mask[group_mask].index
        idx = pd.MultiIndex.from_frame(df[group_cols])

        return df[idx.isin(keep_ids)]

    # ------------------------------------------------------------------
    # Apply all filters sequentially
    # ------------------------------------------------------------------

    for key, spec in filtering_kwargs.items():

        # ---- Case 1: direct column
        if "min_value" in spec or "exact_value" in spec:
            df = _apply_filter(
                df,
                column=key,
                min_val=spec.get("min_value"),
                max_val=spec.get("max_value"),
                exact_val=spec.get("exact_value"),
                mode=spec["mode"],
            )
            continue

        # ---- Case 2: structured
        part = key
        for param, p_spec in spec.items():
            col = _build_column_name(
                part=part,
                param=param,
                stats_key=p_spec.get("stats_key"),
            )

            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")

            df = _apply_filter(
                df,
                column=col,
                min_val=p_spec.get("min_value"),
                max_val=p_spec.get("max_value"),
                exact_val=p_spec.get("exact_value"),
                mode=p_spec["mode"],
            )

    return df
