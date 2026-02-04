from scr.utils.types_alias import StatsByQuantity


def propagate_stat_parameter(
        stats: StatsByQuantity,
        source_quantity: str,
        target_quantities: list[str],
        param: str,
) -> None:
    for obs_id, areas in stats.get(source_quantity, {}).items():
        for area, frames in areas.items():
            for frame_id, params in frames.items():
                if param not in params:
                    continue  # skip if `param` not computed for this frame

                value = params[param]
                for q in target_quantities:
                    stats.setdefault(q, {}) \
                         .setdefault(obs_id, {}) \
                         .setdefault(area, {}) \
                         .setdefault(frame_id, {})[param] = value

                    # stats.setdefault(q, {}).setdefault(obs_id, {}).setdefault(area, {}).setdefault(frame_id, {})
                    # stats[q][obs_id][area][frame_id][param] = value
                    value = params[param]
