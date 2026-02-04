import pandas as pd
from copy import deepcopy

from scr.utils.types_alias import ObservationID, Sunspots, SunspotsPhasesByObservation
from scr.utils.collections import nested_defaultdict


def split_by_phase(
        combined_df: pd.DataFrame,
        all_contours: dict[ObservationID, Sunspots],
) -> SunspotsPhasesByObservation:
    phase_contours = nested_defaultdict(depth=2)

    for _, row in combined_df.iterrows():
        ph = row.phase
        if ph.lower() not in ["forming", "stable", "decaying"]:
            continue
        fid, sid, frame = row.observation_id, row.sunspot_id, row.frame

        # Copy contours
        spot = all_contours[fid].get(sid, {})
        for region in ("inner", "outer"):
            if frame in spot.get(region, {}):
                phase_contours[fid] \
                    .setdefault(sid, {}) \
                    .setdefault(ph, {}) \
                    .setdefault(region, {})[frame] = deepcopy(spot[region][frame])

    return phase_contours
