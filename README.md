# NTNU Mobile LiDAR exploration

This workspace contains the `TRD_MLS` LAS tiles and a bounded, chunked Python workflow for metadata inspection, diagnostics, RGB/intensity exploration, downsampling, denoising, and normal estimation.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For `.laz` input, install the optional backend with `python -m pip install "laspy[lazrs]"`.
The `--process` Open3D step is optional; install `open3d` separately when a wheel is available for your Python version.
`pyproj` is also optional because the supplied files do not store a CRS. Install it later when you know the source CRS and need coordinate transformation.

## Run

Inspect all seven tile headers and sample tile `3`:

```powershell
python explore_lidar.py --data-dir TRD_MLS --inspect --tile 3
```

Command parameters:

- `--data-dir TRD_MLS`: folder searched recursively for `.las` and `.laz` files.
- `--inspect`: print each tile's point count, LAS version, point format, scale, bounds, CRS, and dimensions.
- `--tile 3`: select the file inside the folder named `3` for sampling. Without this option, the first tile is selected.

Plot elevation/intensity diagnostics:

```powershell
python explore_lidar.py --data-dir TRD_MLS --tile 3 --sample-points 200000 --plot
```

- `--sample-points 200000`: maximum number of evenly spaced records loaded from the selected tile. Lower values use less memory and run faster; larger values give more representative distributions.
- `--plot`: show elevation and intensity histograms. The elevation histogram uses `Z`; the intensity histogram uses the LAS `intensity` field. Each plot now includes a legend and the elevation range. These are distributions, not a spatial map.

If Windows reports `Application Control policy has blocked this file` for Matplotlib's `_backend_agg`, the script automatically prints text histograms instead. For graphical figures, use a supported Python environment such as a Python 3.12/3.13 virtual environment or conda environment, then reinstall `numpy` and `matplotlib`; the error is caused by the Windows DLL policy blocking Matplotlib's compiled backend, not by the LAS files.

Run the Open3D workflow with a 5 cm voxel:

```powershell
python explore_lidar.py --data-dir TRD_MLS --tile 3 --sample-points 200000 --voxel-size 0.05 --process
```

- `--voxel-size 0.05`: voxel edge length in metres. A value of `0.05` means 5 cm; larger values produce fewer, lighter points and more smoothing.
- `--process`: run Open3D processing and open an interactive point-cloud window. The workflow recenters coordinates, voxel-downsamples, applies statistical and radius outlier filters, estimates normals, and displays the statistically cleaned cloud colored by local elevation.

### Interpreting the Open3D result

The Open3D window displays the selected tile sample, not all points in all seven files. Its processing stages are:

1. Coordinates are shifted by the sample mean so the cloud is easier for local algorithms to handle. The printed `local origin` is the world-coordinate value that was subtracted.
2. Voxel downsampling replaces points in each 5 cm voxel with a representative averaged point. Fine details smaller than the voxel size may disappear.
3. Statistical outlier removal keeps points whose neighbourhood distances are not unusually large, using 20 neighbours and a `std_ratio` of `2.0`.
4. Radius outlier removal is also calculated using at least 8 neighbours within a radius of 15 cm or three voxel widths, whichever is larger. The displayed cloud is currently the statistical result; the radius result is reported for comparison.
5. Normals are estimated from neighbours within 20 cm or four voxel widths, with at most 30 neighbours. Normals are useful for surface orientation, plane detection, meshing, and registration.

The shape shows the 3D geometry captured by the mobile scanner, such as a road corridor, terrain, or nearby objects. It is not a photograph. The supplied LAS files do not contain `red`, `green`, or `blue` dimensions, so the data has no true RGB colour. The viewer now includes a colored strip beside the cloud and prints its range: blue represents the lowest local elevation and red the highest local elevation. This is an elevation legend, not camera colour. The intensity plot has its own labeled legend and should not be confused with the Open3D elevation colors.

If Open3D is unavailable, `--process` uses the NumPy fallback for voxel aggregation and isolated-voxel filtering. That fallback does not estimate normals or open an interactive 3D window.

The script uses local coordinates for Open3D while retaining the world-coordinate origin in its output. It does not merge all tiles into RAM. After understanding the attributes and CRS, the next dedicated step is a PDAL pipeline for ground/non-ground classification, for example with SMRF and an outlier filter.

## Reconstruction exports

Export one sparse 2.5D map per tile. Each compressed NPZ contains occupied-cell indices, point count, minimum/maximum/mean elevation, within-cell elevation standard deviation, and mean intensity. Coordinates are reconstructed as `x_origin + ix * cell_size` and `y_origin + iy * cell_size`.

```powershell
python explore_lidar.py --data-dir TRD_MLS --height-map --voxel-size 0.10 --output-dir reconstruction_output
```

Export sparse voxel surfels with centroid position, occupancy, mean intensity, and local PCA normals:

```powershell
python explore_lidar.py --data-dir TRD_MLS --surfels --voxel-size 0.10 --output-dir reconstruction_output
```

Use `--tile 3` to export only one tile. Surfel normals are estimated from occupied neighboring voxels, marked by `normal_valid`, and oriented into the positive-Z hemisphere. They are not globally viewpoint-oriented normals; use the original trajectory or a later orientation pass if a consistent surface orientation is required. Finer voxel sizes increase detail and memory use substantially.

### Viewing reconstruction outputs

The outputs are compressed NumPy `.npz` archives. They are not images or mesh files, so inspect their arrays with Python or visualize them with Matplotlib.

List the arrays and their shapes:

```powershell
python -c "import numpy as np; d=np.load('reconstruction_output/3_0-0_las_height_1m.npz'); print(d.files); [print(k, d[k].shape, d[k].dtype) for k in d.files]"
```

View a 2.5D height map, where color represents mean elevation:

```powershell
python -c "import numpy as np, matplotlib.pyplot as plt; d=np.load('reconstruction_output/3_0-0_las_height_1m.npz'); plt.scatter(d['ix'], d['iy'], c=d['z_mean'], s=2); plt.gca().set_aspect('equal'); plt.colorbar(label='Mean elevation'); plt.show()"
```

View the 3D sparse surfels, colored by mean intensity:

```powershell
python -c "import numpy as np, matplotlib.pyplot as plt; d=np.load('reconstruction_output/3_0-0_las_surfels_2m.npz'); ax=plt.figure().add_subplot(projection='3d'); ax.scatter(d['x'], d['y'], d['z'], c=d['intensity_mean'], s=2); ax.set(xlabel='X', ylabel='Y', zlabel='Z'); plt.show()"
```

For the 2.5D file, `ix` and `iy` are sparse grid indices. Recover world coordinates with `x_origin + ix * cell_size` and `y_origin + iy * cell_size`; `z_mean` is the average elevation in each occupied cell, while `z_min`, `z_max`, and `z_std` describe vertical spread. `count` is the number of source points per cell and `intensity_mean` is the average LiDAR intensity.

For the surfel file, `x`, `y`, and `z` are voxel centroids, `count` is voxel occupancy, and `intensity_mean` is the average intensity. `normal_x`, `normal_y`, and `normal_z` are local PCA normals; use only rows where `normal_valid` is true. The normals are oriented toward positive Z, so they are useful for local surface inspection but are not guaranteed to represent the physically correct side of walls or undersides.

Expect gaps where the scanner did not observe surfaces, reduced detail when using larger cell or voxel sizes, and intensity-based coloring rather than photographic color. These exports are sparse point/surfel representations, not watertight meshes. The height map is especially useful for terrain, roads, and broad surface structure; the surfel output preserves more general 3D geometry for later registration, meshing, or spatial analysis.

## Phase 4: ground and object separation

Because the LAS classification fields are not trustworthy in this dataset, the project now includes a lightweight fallback ground filter that operates on the XYZ cloud directly. It groups points into a sparse XY grid, estimates a local terrain floor, and separates likely ground points from elevated object points without depending on vendor classification labels.

This is intentionally conservative:

- it keeps near-ground cells in the ground set,
- it removes elevated mixed cells from the ground set,
- it stores the separation as a boolean mask, which can be applied to the point cloud or used to process ground and object geometry separately.

The implementation is in `ground_filter.py`, and the validation tests live in `tests/test_ground_filter.py`.

## Phase 5: local surface reconstruction baselines

For representative blocks, the project includes a lightweight comparison layer that estimates reconstruction behavior for a few families without requiring a heavy geometry stack:

- TIN-style planar approximation
- ball-pivot style dense-surface behavior
- Poisson-style smooth local surface behavior
- alpha-shape style boundary preservation

The comparison is measured in terms of surface area and empty-region penalty so the user can judge how much each method may invent geometry in unobserved or sparse regions. The implementation is in `surface_reconstruction.py`, with validation in `tests/test_surface_reconstruction.py`.

## Phase 6: depth-map research guardrail

This phase is intentionally deferred until scanner pose and calibration are confirmed. The project now includes a guard that blocks advanced depth-fusion and implicit-surface methods unless metadata shows pose and calibration are available. This prevents a premature TSDF or neural-surface pipeline from being built on an unverified acquisition setup.

The guard is in `depth_research.py`, and the validation lives in `tests/test_depth_guard.py`.

## Review process used in the project

Between major stages, the project uses independent validation tests that exercise the phase outputs before moving on. These tests are intentionally small, synthetic, and deterministic so they can confirm that the script logic behaves as expected without requiring full dataset processing.

Run the full validation suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

This currently validates the ground separation, reconstruction comparison, and depth guardrail logic.

## Observed dataset metadata

The current seven files are LAS 1.2, point format 1, with approximately 16.7 million points total. They contain XYZ, intensity, GPS time, return/classification and scan metadata, but no `red`, `green`, or `blue` dimensions and no stored CRS. The sampled classification and return-number values are zero, so treat those fields as unpopulated until verified against the source documentation. The coordinates are projected-looking world coordinates around easting `571,000` and northing `7,031,000`; do not transform them without confirming the source CRS.