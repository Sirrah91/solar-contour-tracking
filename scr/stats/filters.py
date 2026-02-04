from scr.utils.types_alias import SunspotID, Sunspots, Stats, StatsByQuantity


def remove_empty_stats(
        stats: Stats
) -> Stats:
    """Remove stat entries with no valid region data."""
    return {
        sid: sdata for sid, sdata in stats.items()
        if any(sdata.get(region) and sdata[region] for region in ("penumbra", "umbra", "ratio", "overall"))
    }


def slice_statistics(
        stats: Stats,
        sunspot_indices: list[SunspotID]
) -> Stats:
    return {sid: value for sid, value in stats.items() if sid in sunspot_indices}


def slice_statistics_to_sunspots(
        stats: StatsByQuantity,
        sunspots: Sunspots
) -> StatsByQuantity:
    """
    Slice statistics to match sunspots by ID and frame indices.

    Parameters:
        stats: Full statistics dictionary.
        sunspots: Filtered sunspots dictionary.

    Returns:
        Trimmed statistics dictionary aligned with provided sunspots.
    """
    result = {}
    for quantity in stats:
        result_part: Stats = {}
        stats_part = stats[quantity]
        which = "outer"

        for sid in sunspots:
            if sid not in stats_part:
                continue
            result_part[sid] = {}

            valid_frames = sorted(set(sunspots[sid][which].keys()))

            for region in ["penumbra", "umbra", "ratio", "overall"]:
                if region not in stats_part[sid]:
                    continue
                result_part[sid][region] = {
                    t: stat for t, stat in stats_part[sid][region].items()
                    if t in valid_frames
                }
        result[quantity] = result_part

    return result
