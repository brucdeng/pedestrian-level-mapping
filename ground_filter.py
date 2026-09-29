from __future__ import annotations

import numpy as np


def separate_ground_and_objects(
    points: np.ndarray,
    cell_size: float = 0.5,
    height_tolerance: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate low, terrain-like points from elevated object points.

    The method works without LAS classification metadata by building a sparse
    XY occupancy grid and labeling each occupied cell as ground-like when its
    minimum elevation stays near the local terrain floor.
    """
    if points is None:
        raise ValueError("points must not be None")
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if array.size == 0:
        return np.array([], dtype=bool), np.array([], dtype=bool)
    if not np.isfinite(array).all():
        raise ValueError("points must be finite")
    if cell_size <= 0.0 or not np.isfinite(cell_size):
        raise ValueError("cell_size must be a positive finite number")

    x_min, y_min = array[:, :2].min(axis=0)
    x_max, y_max = array[:, :2].max(axis=0)
    cell_x = np.floor((array[:, 0] - x_min) / cell_size).astype(np.int64)
    cell_y = np.floor((array[:, 1] - y_min) / cell_size).astype(np.int64)

    nx = max(int(np.ceil((x_max - x_min) / cell_size)) + 1, 1)
    ny = max(int(np.ceil((y_max - y_min) / cell_size)) + 1, 1)
    cell_ids = cell_y * nx + cell_x
    unique_ids, inverse = np.unique(cell_ids, return_inverse=True)

    cell_min = np.empty(unique_ids.size, dtype=np.float64)
    cell_max = np.empty(unique_ids.size, dtype=np.float64)
    for cell_index in range(unique_ids.size):
        mask = inverse == cell_index
        cell_min[cell_index] = array[mask, 2].min()
        cell_max[cell_index] = array[mask, 2].max()

    global_floor = float(array[:, 2].min())
    cell_to_index = {int(cell): idx for idx, cell in enumerate(unique_ids)}
    local_floor = np.empty(unique_ids.size, dtype=np.float64)
    for cell_index, cell_id in enumerate(unique_ids):
        cx = int(cell_id % nx)
        cy = int(cell_id // nx)
        local_floor[cell_index] = cell_min[cell_index]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nb_x = cx + dx
                nb_y = cy + dy
                if nb_x < 0 or nb_x >= nx or nb_y < 0 or nb_y >= ny:
                    continue
                nb_id = nb_y * nx + nb_x
                if nb_id in cell_to_index:
                    local_floor[cell_index] = min(local_floor[cell_index], cell_min[cell_to_index[nb_id]])

    point_ground = np.zeros(array.shape[0], dtype=bool)
    for idx, cell_index in enumerate(inverse):
        cell_floor = min(cell_min[cell_index], local_floor[cell_index])
        if cell_min[cell_index] <= global_floor + height_tolerance + 0.05:
            point_ground[idx] = array[idx, 2] <= cell_floor + height_tolerance
        elif cell_min[cell_index] <= local_floor[cell_index] + height_tolerance and (
            cell_max[cell_index] - cell_min[cell_index] <= max(0.5, 2.0 * cell_size)
        ):
            point_ground[idx] = array[idx, 2] <= cell_min[cell_index] + height_tolerance

    return point_ground, ~point_ground
