import pandas as pd
from typing import Literal
from os import path

from scr.utils.filesystem import check_dir

from scr.io.parquet import load_parquet, save_parquet


def merge_pore_sunspot_dataframes(
        df_pores: pd.DataFrame,
        df_sunspots: pd.DataFrame,
        major: Literal["sunspots", "pores"],
        save_path: str | None = None,
        keys: tuple[str, ...] = ("observation_id", "sunspot_id", "frame")
) -> pd.DataFrame:
    """
    Merge pore (0.65) and umbra (0.5) contour statistics into a single dataframe.

    - Penumbral (0.9) quantities come from the `major` dataframe.
    - Penumbral area statistics are computed as penumbra minus inner contour of major object:
        - major='pores'  → penumbra - pore (0.65)
        - major='sunspots' → penumbra - umbra (0.5)
    - Auxiliary dataframe contributes non-area columns only.
    """

    # --- Row alignment ---
    if (not df_pores[list(keys)].equals(df_sunspots[list(keys)])
            or not df_pores["image_path"].equals(df_sunspots["image_path"])):
        raise ValueError("Row alignment mismatch between df_pores and df_sunspots")

    # --- Helper: token-safe renaming ---
    def _rename_umbra_to_pore(col: str) -> str:
        parts = col.split("_")
        parts = ["pore" if p == "umbra" else p for p in parts]
        return "_".join(parts)

    # --- Always start from a copy of df_pores with renamed columns ---
    df_pores_renamed = df_pores.copy(deep=True).rename(columns=_rename_umbra_to_pore)

    if major == "pores":
        df = df_pores_renamed
        # select umbra columns from sunspots
        umbra_cols = [
            c for c in df_sunspots.columns
            if ("umbra" in c.split("_")) and (c not in keys)
        ]
        overlap = set(df.columns) & set(umbra_cols)
        if overlap:
            raise ValueError(f"Column collision during merge: {overlap}")
        df_aux = df_sunspots[umbra_cols].copy()
        df = pd.concat((df, df_aux), axis=1)

    elif major == "sunspots":
        df = df_sunspots.copy(deep=True)
        # select pore columns from df_pores_renamed
        pore_cols = [
            c for c in df_pores_renamed.columns
            if ("pore" in c.split("_")) and (c not in keys)
        ]
        overlap = set(df.columns) & set(pore_cols)
        if overlap:
            raise ValueError(f"Column collision during merge: {overlap}")
        df_aux = df_pores_renamed[pore_cols].copy()
        df = pd.concat((df, df_aux), axis=1)

    else:
        raise ValueError(f"Invalid value for `major`: {major}")

    # --- Optional save ---
    if save_path is not None:
        if not save_path.endswith(".parquet"):
            save_path = f"{save_path}.parquet"
        check_dir(save_path, is_file=True)
        save_parquet(save_path, df=df)

    return df


def run_merge_pore_sunspot_dataframes(
        path_pore_df: str,
        path_sunspot_df: str,
        outdir: str
) -> None:
    df_pores = load_parquet(path_pore_df)
    df_sunspots = load_parquet(path_sunspot_df)

    filename = path.split(path_pore_df)[-1].replace(".parquet", "_merged.parquet")
    merge_pore_sunspot_dataframes(
        df_pores,
        df_sunspots,
        major="pores",
        save_path=path.join(outdir, filename),
    )

    filename = path.split(path_sunspot_df)[-1].replace(".parquet", "_merged.parquet")
    merge_pore_sunspot_dataframes(
        df_pores,
        df_sunspots,
        major="sunspots",
        save_path=path.join(outdir, filename),
    )
