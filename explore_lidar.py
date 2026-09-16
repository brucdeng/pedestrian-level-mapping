"""Explore NTNU Mobile LiDAR LAS tiles.

Examples:
    python explore_lidar.py --data-dir TRD_MLS --inspect
    python explore_lidar.py --data-dir TRD_MLS --tile 3 --sample-points 200000 --plot
    python explore_lidar.py --data-dir TRD_MLS --tile 3 --sample-points 200000 --process

The script deliberately samples large files in chunks. It does not merge all
tiles into memory unless a caller explicitly chooses to do so.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import laspy
import numpy as np


def find_tiles(data_dir: Path) -> list[Path]:
    """Return LAS/LAZ tiles in stable path order."""
    return sorted({*data_dir.rglob("*.las"), *data_dir.rglob("*.laz")})


def dimensions(header: laspy.LasHeader) -> set[str]:
    return set(header.point_format.dimension_names)


def inspect_tiles(paths: Iterable[Path]) -> None:
    """Print metadata without reading point records into memory."""
    for path in paths:
        with laspy.open(path) as reader:
            header = reader.header
            dims = dimensions(header)
            print(f"\n{path}")
            print(f"  points: {header.point_count:,}")
            print(f"  version: {header.version} | format: {header.point_format}")
            print(f"  scale: {header.scales} | offset: {header.offsets}")
            print(f"  bounds: min={header.mins} max={header.maxs}")
            print(f"  CRS: {header.parse_crs() or 'not stored'}")
            print(f"  dimensions: {', '.join(sorted(dims))}")


def read_sample(path: Path, limit: int) -> dict[str, np.ndarray]:
    """Read at most ``limit`` evenly spaced records using chunked iteration."""
    with laspy.open(path) as reader:
        total = reader.header.point_count
        stride = max(1, int(np.ceil(total / limit)))
        chunks: dict[str, list[np.ndarray]] = {"xyz": [], "intensity": []}
        dims = dimensions(reader.header)
        for points in reader.chunk_iterator(1_000_000):
            indices = np.arange(0, len(points), stride)
            chunks["xyz"].append(np.column_stack((points.x, points.y, points.z))[indices])
            if "intensity" in dims:
                chunks["intensity"].append(np.asarray(points.intensity)[indices])
            if {"red", "green", "blue"}.issubset(dims):
                chunks.setdefault("rgb", []).append(
                    np.column_stack((points.red, points.green, points.blue))[indices]
                )
            if "classification" in dims:
                chunks.setdefault("classification", []).append(np.asarray(points.classification)[indices])
            if "return_number" in dims:
                chunks.setdefault("return_number", []).append(np.asarray(points.return_number)[indices])

    sample = {key: np.concatenate(value) for key, value in chunks.items() if value}
    if len(sample["xyz"]) > limit:
        sample = {key: value[:limit] for key, value in sample.items()}
    return sample


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    """Convert 8-bit or 16-bit LAS colors to Open3D's [0, 1] range."""
    rgb = rgb.astype(np.float64)
    divisor = 65535.0 if rgb.max(initial=0) > 255 else 255.0
    return np.clip(rgb / divisor, 0.0, 1.0)


def scalar_colors(values: np.ndarray) -> np.ndarray:
    """Map scalar values to a blue-green-yellow-red color ramp."""
    minimum = float(values.min())
    maximum = float(values.max())
    normalized = np.zeros_like(values, dtype=np.float64)
    if maximum > minimum:
        normalized = (values - minimum) / (maximum - minimum)
    stops = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    positions = normalized * (len(stops) - 1)
    lower = np.floor(positions).astype(int).clip(0, len(stops) - 2)
    fraction = positions - lower
    return stops[lower] * (1.0 - fraction[:, None]) + stops[lower + 1] * fraction[:, None]


def make_elevation_legend(points: np.ndarray):
    """Create a colored strip displayed beside the cloud as an in-scene legend."""
    import open3d as o3d

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    x_span = max(maximum[0] - minimum[0], 1.0)
    levels = np.linspace(minimum[2], maximum[2], 64)
    strip = np.column_stack(
        (
            np.full(len(levels), maximum[0] + max(x_span * 0.05, 0.5)),
            np.full(len(levels), minimum[1]),
            levels,
        )
    )
    strip = np.repeat(strip, 8, axis=0)
    strip[:, 0] += np.tile(np.linspace(-x_span * 0.01, x_span * 0.01, 8), len(levels))
    strip_colors = scalar_colors(levels)
    strip_colors = np.repeat(strip_colors, 8, axis=0)
    legend = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(strip))
    legend.colors = o3d.utility.Vector3dVector(strip_colors)
    return legend, float(minimum[2]), float(maximum[2])


def print_distributions(sample: dict[str, np.ndarray]) -> None:
    xyz = sample["xyz"]
    print("\nSample summary")
    print(f"  records: {len(xyz):,}")
    print(f"  bounds: min={xyz.min(axis=0)} max={xyz.max(axis=0)}")
    if "rgb" in sample:
        print(f"  RGB range: {sample['rgb'].min(axis=0)} .. {sample['rgb'].max(axis=0)}")
    for key in ("intensity", "classification", "return_number"):
        if key in sample:
            values, counts = np.unique(sample[key], return_counts=True)
            print(f"  {key}: {dict(zip(values.tolist(), counts.tolist()))}")


def make_point_cloud(sample: dict[str, np.ndarray]):
    """Build a local-coordinate Open3D cloud from a bounded sample."""
    import open3d as o3d

    xyz = sample["xyz"].astype(np.float64)
    origin = xyz.mean(axis=0)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz - origin))
    if "rgb" in sample:
        cloud.colors = o3d.utility.Vector3dVector(normalize_rgb(sample["rgb"]))
    return cloud, origin


def process_and_visualize(sample: dict[str, np.ndarray], voxel_size: float) -> None:
    try:
        import open3d as o3d
    except ImportError:
        process_numpy_fallback(sample, voxel_size)
        return

    cloud, origin = make_point_cloud(sample)
    downsampled = cloud.voxel_down_sample(voxel_size)
    statistical, _ = downsampled.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    radius, _ = downsampled.remove_radius_outlier(nb_points=8, radius=max(0.15, voxel_size * 3))
    statistical.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(0.2, voxel_size * 4), max_nn=30)
    )
    statistical_points = np.asarray(statistical.points)
    statistical.colors = o3d.utility.Vector3dVector(scalar_colors(statistical_points[:, 2]))
    legend, minimum_elevation, maximum_elevation = make_elevation_legend(statistical_points)
    print(f"\nProcessing (local origin: {origin})")
    print(f"  sampled -> voxel downsampled: {len(cloud.points):,} -> {len(downsampled.points):,}")
    print(f"  statistical outlier result: {len(statistical.points):,}")
    print(f"  radius outlier result: {len(radius.points):,}")
    print(f"  color legend: blue={minimum_elevation:.3f} m, red={maximum_elevation:.3f} m (local elevation)")
    o3d.visualization.draw_geometries(
        [statistical, legend],
        window_name="NTNU LiDAR: cleaned | elevation colors | blue low - red high",
    )


def process_numpy_fallback(sample: dict[str, np.ndarray], voxel_size: float) -> None:
    """Run bounded voxel processing when Open3D is unavailable."""
    xyz = sample["xyz"].astype(np.float64)
    origin = xyz.mean(axis=0)
    local_xyz = xyz - origin
    voxel_keys = np.floor(local_xyz / voxel_size).astype(np.int64)
    unique_keys, inverse, counts = np.unique(voxel_keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(unique_keys), 3), dtype=np.float64)
    np.add.at(sums, inverse, local_xyz)
    downsampled = sums / counts[:, None]

    occupied = {tuple(key): index for index, key in enumerate(unique_keys)}
    keep = np.zeros(len(unique_keys), dtype=bool)
    for index, key in enumerate(unique_keys):
        neighbors = sum(
            tuple(key + offset) in occupied
            for offset in np.ndindex(3, 3, 3)
            for offset in (np.asarray(offset) - 1,)
            if np.any(offset)
        )
        keep[index] = neighbors >= 2
    if not keep.any():
        keep[:] = True
        filter_note = "sample too sparse for density filtering; retained all voxels"
    else:
        filter_note = "removed voxels with fewer than two occupied neighbors"

    print("\nOpen3D is unavailable for this Python 3.14 environment.")
    print("Ran the NumPy fallback: voxel aggregation and isolated-voxel filtering.")
    print(f"  local origin: {origin}")
    print(f"  sampled -> voxel downsampled: {len(xyz):,} -> {len(downsampled):,}")
    print(f"  density-filtered result: {keep.sum():,} ({filter_note})")
    print("  normals and interactive 3D viewing require Open3D on Python 3.12 or 3.13.")


def plot_diagnostics(sample: dict[str, np.ndarray]) -> None:
    try:
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(13, 5))
        elevation = sample["xyz"][:, 2]
        axes[0].hist(elevation, bins=100, color="steelblue", label="Z / elevation")
        axes[0].set(xlabel="Elevation", ylabel="Points", title="Elevation distribution")
        axes[0].legend(loc="upper right")
        if "intensity" in sample:
            intensity = sample["intensity"]
            axes[1].hist(intensity, bins=100, color="darkorange", label="LAS intensity")
            axes[1].set(xlabel="Intensity", ylabel="Points", title="Intensity distribution")
            axes[1].legend(loc="upper right")
        else:
            axes[1].axis("off")
        figure.suptitle(
            f"NTNU LiDAR diagnostics | elevation {elevation.min():.3f}-{elevation.max():.3f} m"
        )
        figure.tight_layout()
        plt.show()
        return
    except ImportError as exc:
        if "_backend_agg" not in str(exc) and "Application Control policy" not in str(exc):
            raise

    print("\nMatplotlib's native plotting backend is blocked by Windows policy.")
    print("Showing text histograms instead; install/run Python in an environment that allows Matplotlib DLLs for figures.\n")
    print_histogram("Elevation", sample["xyz"][:, 2])
    if "intensity" in sample:
        print_histogram("Intensity", sample["intensity"])


def print_histogram(label: str, values: np.ndarray, bins: int = 30) -> None:
    """Print a compact histogram without compiled plotting dependencies."""
    counts, edges = np.histogram(values, bins=bins)
    maximum = counts.max(initial=0)
    print(label)
    for count, start, end in zip(counts, edges[:-1], edges[1:]):
        width = int(round(40 * count / maximum)) if maximum else 0
        print(f"  {start:12.3f} .. {end:12.3f} | {'#' * width} {count:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("TRD_MLS"))
    parser.add_argument("--tile", type=str, help="Folder name, for example 3")
    parser.add_argument("--sample-points", type=int, default=200_000)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--inspect", action="store_true", help="Print metadata for every tile")
    parser.add_argument("--plot", action="store_true", help="Show elevation and intensity histograms")
    parser.add_argument("--process", action="store_true", help="Downsample, clean, estimate normals, and view")
    args = parser.parse_args()

    paths = find_tiles(args.data_dir)
    if not paths:
        raise SystemExit(f"No .las or .laz files found under {args.data_dir.resolve()}")
    if args.inspect or not (args.plot or args.process):
        inspect_tiles(paths)

    selected = next((path for path in paths if args.tile and path.parent.name == args.tile), paths[0])
    print(f"\nSelected tile: {selected}")
    sample = read_sample(selected, args.sample_points)
    print_distributions(sample)
    if args.plot:
        plot_diagnostics(sample)
    if args.process:
        process_and_visualize(sample, args.voxel_size)


if __name__ == "__main__":
    main()