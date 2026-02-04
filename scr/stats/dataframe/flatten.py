import numpy as np
import pandas as pd

from scr.utils.types_alias import ObservationID, StatsByQuantity


def flatten_spot_features_with_frame(all_stats: dict[ObservationID, StatsByQuantity]) -> pd.DataFrame:
    """
    Flatten nested sunspot statistics into a Pandas DataFrame.

    Rules
    -----
    * Penumbra / umbra:
        - Keep ALL parameters containing "flux" (unique per physical quantity).
        - Keep ALL non-flux parameters ONCE per frame (unique per contour).

    * Ratio / overall:
        - Keep ALL parameters ONCE per frame (independent of quantity).

    Output
    ------
    DataFrame where each row represents one (observation_id, sunspot_id, frame),
    dtype-optimised:
        - observation_id: category
        - sunspot_id, frame: int32
        - all numeric values: float32
    """
    records = []
    valid_parts = {"penumbra", "umbra", "ratio", "overall"}

    for obs_id, quantities in all_stats.items():

        # All spot ids under any physical quantity
        spot_ids = set().union(*(q.keys() for q in quantities.values()))

        for spot_id in spot_ids:

            # Collect all frames for this spot
            all_frames = set()
            for spots in quantities.values():
                if spot_id not in spots:
                    continue
                for part in valid_parts:
                    if part in spots[spot_id]:
                        all_frames.update(spots[spot_id][part].keys())

            # Process each frame
            for frame in all_frames:

                record = {
                    "observation_id": obs_id,  # will convert to category later
                    "sunspot_id": np.int32(spot_id),
                    "frame": np.int32(frame),
                }

                written_overall = False
                written_ratio = False
                written_geom_umbra = False
                written_geom_pen = False

                for phys_q, spots in quantities.items():
                    if spot_id not in spots:
                        continue

                    spot_data = spots[spot_id]

                    for part in valid_parts:
                        if part not in spot_data:
                            continue

                        part_data = spot_data[part]
                        if frame not in part_data:
                            continue

                        params = part_data[frame]

                        # ---------------- ratio: once per frame ----------------
                        if part == "ratio":
                            if not written_ratio:
                                for param, val in params.items():
                                    record[f"ratio_{param}"] = (
                                        np.float32(val) if val is not None else np.nan
                                    )
                                written_ratio = True
                            continue

                        # ---------------- overall: once per frame --------------
                        if part == "overall":
                            if not written_overall:
                                for param, val in params.items():
                                    record[f"overall_{param}"] = (
                                        np.float32(val) if val is not None else np.nan
                                    )
                                written_overall = True
                            continue

                        # ---------------- umbra / penumbra ---------------------
                        if part in {"umbra", "penumbra"}:

                            # (A) flux parameters → quantity-dependent
                            for param, val in params.items():
                                if "flux" in param:
                                    key = f"{phys_q}_{part}_{param}"
                                    record[key] = (
                                        np.float32(val) if val is not None else np.nan
                                    )

                            # (B) geometric parameters → once per part per frame
                            if part == "umbra" and not written_geom_umbra:
                                for param, val in params.items():
                                    if "flux" not in param:
                                        key = f"{part}_{param}"
                                        record[key] = (
                                            np.float32(val)
                                            if val is not None
                                            else np.nan
                                        )
                                written_geom_umbra = True

                            if part == "penumbra" and not written_geom_pen:
                                for param, val in params.items():
                                    if "flux" not in param:
                                        key = f"{part}_{param}"
                                        record[key] = (
                                            np.float32(val)
                                            if val is not None
                                            else np.nan
                                        )
                                written_geom_pen = True

                records.append(record)

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Optimise ID columns
    df["observation_id"] = df["observation_id"].astype("category")
    df["sunspot_id"] = df["sunspot_id"].astype("int32")
    df["frame"] = df["frame"].astype("int32")

    # ---- Sort here for stable output ----
    df.sort_values(["observation_id", "sunspot_id", "frame"], inplace=True)

    # ---- Add per-sunspot local index ----
    df["spot_global_index"] = (
        df[["observation_id", "sunspot_id"]]
        .astype(str)
        .agg("::".join, axis=1)
        .astype("category")
        .cat.codes
        .astype("int32")
    )

    df.reset_index(drop=True, inplace=True)

    return df
