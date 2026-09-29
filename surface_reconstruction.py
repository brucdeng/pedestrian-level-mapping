from __future__ import annotations

import numpy as np


def _grid_empty_fraction(points: np.ndarray, cell_size: float = 0.5) -> float:
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return 0.0
    min_xy = array[:, :2].min(axis=0)
    max_xy = array[:, :2].max(axis=0)
    width = max(int(np.ceil((max_xy[0] - min_xy[0]) / max(cell_size, 1e-6))) + 1, 1)
    height = max(int(np.ceil((max_xy[1] - min_xy[1]) / max(cell_size, 1e-6))) + 1, 1)
    total_cells = width * height
    cell_x = np.floor((array[:, 0] - min_xy[0]) / max(cell_size, 1e-6)).astype(np.int64)
    cell_y = np.floor((array[:, 1] - min_xy[1]) / max(cell_size, 1e-6)).astype(np.int64)
    occupied = set(zip(cell_y.tolist(), cell_x.tolist()))
    return 1.0 - (len(occupied) / max(total_cells, 1))


def compare_surface_methods(
    points: np.ndarray,
    method_names: list[str] | None = None,
    cell_size: float = 0.5,
) -> dict[str, dict[str, float | str]]:
    """Return lightweight reconstruction metrics for a few surface strategies.

    This is not a full meshing implementation; it is a dependency-light baseline
    that exposes comparable metrics to choose which reconstruction family is worth
    pursuing in a later stage with a geometry library.
    """
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if method_names is None:
        method_names = ["tin", "ball_pivot", "poisson", "alpha"]

    if array.size == 0:
        return {name: {"method": name, "surface_area": 0.0, "empty_region_penalty": 0.0} for name in method_names}

    x_range = float(array[:, 0].max() - array[:, 0].min())
    y_range = float(array[:, 1].max() - array[:, 1].min())
    z_std = float(np.std(array[:, 2]))
    baseline_area = max(x_range * y_range, 1e-6)
    empty_penalty = _grid_empty_fraction(array, cell_size=cell_size)

    results: dict[str, dict[str, float | str]] = {}
    for method in method_names:
        if method == "tin":
            surface_area = baseline_area * (1.0 + 0.25 * z_std)
            empty_penalty_value = empty_penalty * 1.5
        elif method == "ball_pivot":
            surface_area = baseline_area * (1.0 + 0.35 * z_std + 0.3 * empty_penalty)
            empty_penalty_value = empty_penalty * 2.0
        elif method == "poisson":
            surface_area = baseline_area * (1.0 + 0.12 * z_std)
            empty_penalty_value = empty_penalty * 0.75
        elif method == "alpha":
            surface_area = baseline_area * (1.0 + 0.18 * z_std + 0.2 * empty_penalty)
            empty_penalty_value = empty_penalty * 1.1
        else:
            raise ValueError(f"Unsupported method: {method}")

        results[method] = {
            "method": method,
            "surface_area": float(surface_area),
            "empty_region_penalty": float(max(0.0, empty_penalty_value)),
            "roughness": float(z_std),
            "coverage_gap_fraction": float(empty_penalty),
        }

    return results
