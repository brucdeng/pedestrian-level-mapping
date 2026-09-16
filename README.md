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

## Observed dataset metadata

The current seven files are LAS 1.2, point format 1, with approximately 16.7 million points total. They contain XYZ, intensity, GPS time, return/classification and scan metadata, but no `red`, `green`, or `blue` dimensions and no stored CRS. The sampled classification and return-number values are zero, so treat those fields as unpopulated until verified against the source documentation. The coordinates are projected-looking world coordinates around easting `571,000` and northing `7,031,000`; do not transform them without confirming the source CRS.