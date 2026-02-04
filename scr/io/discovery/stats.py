from glob import glob


def discover_stat_files(
        contour_file: str,
        force_completeness: bool = True
) -> list[str]:
    files = glob(f"{contour_file.replace('.npz', '')}*")

    if force_completeness and len(files) != 12:
        raise IOError(f"Not enough files for {contour_file}")

    return sorted(files)
