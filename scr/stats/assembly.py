from os import path

from scr.utils.types_alias import StatsByObject
from scr.utils.filesystem import check_dir
from scr.utils.nested import nested_cast

from scr.io.tracks import load_tracks_and_stats, save_tracks_and_stats
from scr.io.discovery.stats import discover_stat_files


def _load_reference_schema(
        reference_file: str
) -> tuple[dict, dict]:
    tracks, _, metadata = load_tracks_and_stats(reference_file)
    return tracks, metadata


def _assemble_stats(
        contour_file: str,
        quantities: list[str],
        stat_types: list[str]
) -> StatsByObject:
    stats = {stype: {} for stype in stat_types}

    for quantity in quantities:
        for stype in stat_types:
            fname = contour_file.replace(".npz", f"_{quantity}_{stype}.npz")

            if not path.isfile(fname):
                print(f"Missing file:\n\t{fname}")
                continue

            _, stat, _ = load_tracks_and_stats(fname)
            stats[stype][quantity] = stat[stype][quantity]

    return stats


def _enforce_schema(
        tracks: dict,
        stats: StatsByObject,
        metadata: dict,
        schema_file: str | None
) -> tuple[dict, StatsByObject, dict]:
    if schema_file and path.isfile(schema_file):
        ref_tracks, ref_stats, ref_meta = load_tracks_and_stats(schema_file)
        tracks = nested_cast(ref_tracks, tracks)
        stats = nested_cast(ref_stats, stats)
        metadata = nested_cast(ref_meta, metadata)

    return tracks, stats, metadata


def combine_single_stat_files(
        contour_file: str,
        contour_file_types: str | None = None,
        outdir: str | None = None,
        force_completeness: bool = True
) -> None:
    files = discover_stat_files(contour_file, force_completeness)

    indir, name = path.split(contour_file)
    outdir = outdir or indir
    check_dir(outdir)

    tracks, metadata = _load_reference_schema(files[0])

    stats = _assemble_stats(
        contour_file=contour_file,
        quantities=["Ic", "B", "Bp", "Bt", "Br", "Bhor"],
        stat_types=["sunspots", "pores"],
    )

    tracks, stats, metadata = _enforce_schema(
        tracks, stats, metadata, contour_file_types
    )

    save_tracks_and_stats(
        filename=path.join(outdir, name.replace(".npz", "_NEW.npz")),
        tracks=tracks,
        stats=stats,
        metadata=metadata,
    )
