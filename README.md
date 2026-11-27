# pqtools

`pqtools` compares the surface chlorophyll-a concentration of the
Mediterranean Sea from three different sources distributed by the
[Copernicus Marine Service](https://marine.copernicus.eu/):

| Source       | Dataset                                              | Resolution | Description                                   |
|--------------|------------------------------------------------------|------------|-----------------------------------------------|
| `reanalysis` | `cmems_mod_med_bgc-plankton_my_4.2km_P1D-m`          | ~4.2 km    | Biogeochemical model, multi-year reanalysis   |
| `analysis`   | `cmems_mod_med_bgc-pft_anfc_4.2km_P1D-m`             | ~4.2 km    | Biogeochemical model, analysis and forecast   |
| `satellite`  | `cmems_obs-oc_med_bgc-plankton_my_l3-multi-1km_P1D`  | ~1 km      | Multi-sensor ocean colour observations (L3)   |

All the data are daily and are read directly from the Copernicus Marine
servers: there is no need to download any file in advance.

At the moment, `pqtools` can compute the **daily timeseries** of the
chlorophyll averaged over a basin, for each of the three sources, and the
**comparison maps** between a model source and a reference one.


## Installation

### Requirements

- Python 3.12 or later
- [Poetry](https://python-poetry.org/) 2.x
- Git, which is needed to download
  [bit.sea](https://github.com/inogs/bit.sea), the library that provides the
  definitions of the basins and of the grids
- A (free) account of the Copernicus Marine Service

### Install the package

`pqtools` depends on `bitsea`, which is installed automatically from its
[Git repository](https://github.com/inogs/bit.sea.git), at the commit recorded
in `poetry.lock`. To move to the latest commit, run `poetry update bitsea`.

From the root of the repository:

```bash
poetry install
```

This creates a virtual environment (in `.venv`) with all the dependencies and
installs the `pqtools` command in it.

> **Note:** `pyproject.toml` pins `python-dateutil` to the 2.x series. This is
> not used directly by `pqtools`, but without it Poetry gets stuck for a very
> long time while resolving the dependencies of `copernicusmarine` (all the
> versions of `botocore` require `python-dateutil < 3`).

### Copernicus Marine credentials

Reading the data requires valid credentials of the Copernicus Marine Service.
The simplest way is to save them once with:

```bash
poetry run copernicusmarine login
```

Alternatively, you can set the environment variables
`COPERNICUSMARINE_SERVICE_USERNAME` and `COPERNICUSMARINE_SERVICE_PASSWORD`.


## Usage

After the installation, the command line interface is available as
`pqtools` (inside the virtual environment) or as `poetry run pqtools`. It
can also be run with `python -m pqtools`.

```bash
poetry run pqtools --help
```

The interface has two actions:

- `compute_timeseries`: computes the daily timeseries of the chlorophyll on a
  basin and saves them in a netCDF file;
- `compute_comparison_maps`: computes the daily maps of the difference between
  a model source and a reference source, with their mean and root mean square,
  and saves them in a netCDF file.

### The basin

Both the actions require the option `--basin`, which can be:

- `bitsea:UUID`, where `UUID` is the uuid of a basin of bitsea: the name of a
  basin of the default module `V2` (e.g. `bitsea:adr1`) or `..MODULE.NAME` for
  the other modules (e.g. `bitsea:..V0.adn`);
- the path of a YAML file that describes a basin, whose shape can be a polygon:

  ```yaml
  name: my_polygon
  extended_name: My favourite basin  # optional
  polygon:
    - {longitude: 12.0, latitude: 44.0}
    - {longitude: 13.5, latitude: 44.0}
    - {longitude: 13.0, latitude: 45.5}
  ```

  or a rectangle with the sides parallel to the axes of the grid:

  ```yaml
  name: my_rectangle
  rectangle:
    minimum_longitude: 12.0
    maximum_longitude: 13.5
    minimum_latitude: 44.0
    maximum_latitude: 45.5
  ```

  The name cannot contain dots or commas. A polygon requires at least 3
  vertices and is closed automatically; the border of a rectangle is part of the
  basin. If the basin is a rectangle, prefer `rectangle`, which is simpler to
  compute.

Only basins made of polygons or rectangles (or unions of them) that overlap the Mediterranean Sea
are supported.

### `compute_timeseries`

```bash
poetry run pqtools compute_timeseries \
    --basin bitsea:adr1 \
    --start-date 2025-03-01 \
    --end-date 2025-03-31 \
    --satellite-average-window 3 \
    --satellite-max-qi 3.5 \
    -o timeseries_march_2025.nc
```

| Option                              | Required | Default | Description |
|-------------------------------------|----------|---------|-------------|
| `--basin BASIN`                     | yes      |         | Basin on which the statistics are computed (see [The basin](#the-basin)) |
| `--start-date`                      | yes      |         | First day of the period (`YYYY-MM-DD`, inclusive) |
| `--end-date`                        | yes      |         | Last day of the period (`YYYY-MM-DD`, inclusive); must not be earlier than `--start-date` |
| `-o`, `--output-file`               | yes      |         | Output netCDF file; its directory must already exist |
| `--satellite-average-window DAYS`   | no       | 3       | Half-width (in days) of the moving average applied to the satellite data; `0` disables it |
| `--satellite-max-qi QI`             | no       | 0       | Threshold on the quality index of the satellite data; `0` disables the filter |

Dates without a time zone are interpreted as UTC, which is the time
reference of the datasets.

#### Output

The output file is a netCDF file with a `time` dimension (one value per day
of the requested period) and the following variables:

| Variable              | Units  | Description |
|-----------------------|--------|-------------|
| `reanalysis_mean`     | mg m-3 | Spatial mean of the reanalysis chlorophyll |
| `reanalysis_std`      | mg m-3 | Spatial standard deviation of the reanalysis chlorophyll |
| `analysis_mean`       | mg m-3 | Spatial mean of the analysis chlorophyll |
| `analysis_std`        | mg m-3 | Spatial standard deviation of the analysis chlorophyll |
| `satellite_mean`      | mg m-3 | Spatial mean of the satellite chlorophyll |
| `satellite_std`       | mg m-3 | Spatial standard deviation of the satellite chlorophyll |
| `satellite_coverage`  | %      | Percentage of the area of the basin covered by valid satellite data |

The global attributes of the file record the basin, the options used for the
satellite data and, in `source_datasets`, the Copernicus Marine datasets that
have been read, with their versions. If a source has no data on some days of
the period (e.g. the analysis, which covers only the last few years), its
variables are NaN on those days.

### `compute_comparison_maps`

```bash
poetry run pqtools compute_comparison_maps \
    --basin bitsea:adr1 \
    --evaluated-dataset analysis \
    --start-date 2025-03-01 \
    --end-date 2025-03-31 \
    -o comparison_march_2025.nc
```

The options are the same as `compute_timeseries`, plus:

| Option                              | Required | Default     | Description |
|-------------------------------------|----------|-------------|-------------|
| `--evaluated-dataset`               | yes      |             | Model source to evaluate (`analysis` or `reanalysis`) |
| `--reference-dataset`               | no       | `satellite` | Source used as reference; must be different from `--evaluated-dataset` |
| `--satellite-min-coverage FRACTION` | no       | 0.75        | Minimum fraction (0 – 1) of the sea area of a model cell that must be covered by valid satellite data; `0` keeps all the cells with at least one valid value |

The options on the satellite data have no effect if the reference is not the
satellite.

#### Output

The output file is a netCDF file, on the model grid, with the following
variables:

| Variable             | Dimensions                | Units  | Description |
|----------------------|---------------------------|--------|-------------|
| `difference`         | time, latitude, longitude | mg m-3 | Evaluated - reference, on the cells valid in both the sources (NaN elsewhere) |
| `mean_difference`    | time                      | mg m-3 | Spatial mean of the differences (bias) |
| `rms_difference`     | time                      | mg m-3 | Spatial root mean square of the differences (the bias is not removed) |
| `reference_coverage` | time, latitude, longitude | 1      | Fraction (0 – 1) of the sea area of each cell covered by valid satellite data; only if the reference is the satellite |

The global attributes of the file record the basin, the two datasets, the
Copernicus Marine datasets that have been read with their versions
(`source_datasets`) and, if the reference is the satellite, the options used
for its data. The days without
any valid cell (e.g. the ones available in only one of the two sources) are NaN.


## How the timeseries are computed

For each source, `compute_timeseries` performs the following steps.

1. **Download.** The surface chlorophyll (for the models, the first depth
   level, about 1 m) is read on the bounding box of the basin, clipped to the
   Mediterranean Sea. The units are checked and always returned as `mg m-3`.

2. **Quality control of the satellite data.**
   - The values outside the valid range declared by the product
     (0.01 – 100 mg m-3) are always removed.
   - If `--satellite-max-qi` is greater than 0, the values whose quality
     index `QI_CHL` is, in absolute value, strictly greater than the
     threshold are removed, together with the values without a quality
     index. `QI_CHL` is the anomaly of the daily value with respect to the
     climatology, in units of climatological standard deviations: it is not
     an error flag, and high values can also be real events (e.g. blooms).

3. **Moving average of the satellite data.** If
   `--satellite-average-window` is greater than 0, the satellite data are
   smoothed in time with a centered window of `2 * DAYS + 1` days, to fill
   the cells left empty by clouds. For each cell, the result is the
   **geometric mean** of the valid observations in the window (i.e. the
   exponential of the mean of their logarithms), since the chlorophyll is
   approximately log-normally distributed; if the cell has no valid
   observations in the window, it stays NaN. The data are downloaded on a
   period extended by `DAYS` days on both sides, so that every day of the
   requested period has its full window.

4. **Masks.** The values outside the mask of their source are set to NaN:
   - **model mask:** the land/sea masks of the analysis and of the
     reanalysis are downloaded from their static datasets and combined (a
     point is sea only if it is sea in both), so that the two model sources
     are compared on exactly the same points;
   - **satellite mask:** the L3 product has no static mask, and its NaNs mark
     both land and clouds. The mask is therefore built from the gap-free L4
     version of the same product
     (`cmems_obs-oc_med_bgc-plankton_my_l4-gapfree-multi-1km_P1D`), which is
     defined on the same grid and whose NaNs mark land only;
   - both masks are then restricted to the points inside the basin.

5. **Statistics.** For each day, the spatial mean and standard deviation are
   computed on the valid values only, **weighting each cell with its area**.
   The standard deviation is the one of the weighted population (no
   correction for the degrees of freedom). For the satellite, the coverage is
   the ratio between the area of the cells with valid data and the area of
   the satellite mask.

For the timeseries, the model and the satellite data are **not regridded**:
each source is averaged on its own grid and only the resulting timeseries are
compared.


## How the comparison maps are computed

The data are downloaded, filtered, averaged in time and masked as for the
timeseries (steps 1-4), with one exception: the satellite data are cut on a
rectangle slightly larger than the bounding box of the basin, because the model
cells on the border of the basin also cover satellite points outside of it.
Then:

1. **Regridding of the satellite data** (only if the reference is the
   satellite). The satellite data are interpolated on the model grid with a
   conservative regridding: the value of each model cell is the arithmetic mean
   of the valid satellite values that overlap it, weighted with the area of the
   overlap. The coverage of a cell is the fraction of its sea area (according
   to the satellite mask) with valid data: the cells with a coverage lower
   than `--satellite-min-coverage` (by default 0.75, i.e. 75%) are set to NaN.
   The coverage itself is saved in `reference_coverage` also for these cells,
   while it is NaN on the model cells that are entirely land for the satellite.

2. **Difference.** For each day, the difference (evaluated - reference) is
   computed on the cells valid in both the sources.

3. **Statistics.** For each day, the mean and the root mean square of the
   differences are computed on the valid cells, **weighting each cell with its
   area**.


## Things to keep in mind

- **Data availability.** The analysis covers only a rolling window of a bit
  more than the last two years (the oldest days are removed as new ones are
  added), while the reanalysis and the satellite cover several decades. Check
  the current period with `poetry run copernicusmarine describe --dataset-id
  cmems_mod_med_bgc-pft_anfc_4.2km_P1D-m`. If the
  requested period is *partially* outside the availability of a source, the
  missing days are NaN; if it is *entirely* outside, the command fails with
  an error raised by `copernicusmarine`.
- **Different coastlines.** The model grid (~4.2 km) and the satellite grid
  (~1 km) do not have the same coastline, so the area of the basin is
  slightly different for the two (e.g. about 6% larger for the satellite in
  the northern Adriatic Sea). This matters most in basins with complex
  coasts.
- **Coastal lagoons.** The satellite mask includes as sea about 0.1% of
  points where the L3 product never has valid data (mainly coastal lagoons,
  such as the Nile delta lagoons and the Venice lagoon, and narrow channels
  between islands). They do not affect the mean, but they slightly lower the
  coverage, which therefore never reaches exactly 100%.
- **Precision.** All the computations are done in single precision
  (`float32`), which is more than enough for these data and halves the
  memory usage.
- **Compression.** The maps are saved with a lossless zlib compression.
- **Dataset versions.** The Copernicus Marine Service periodically retires
  old versions of its datasets (`copernicusmarine` prints a warning when a
  version is going to be retired). If a dataset stops working, check the
  identifiers in `src/pqtools/components/data_retrievers.py` and
  `src/pqtools/components/masks.py`. The versions used for an output file are
  recorded in its `source_datasets` attribute.
- **Performance.** The data are read lazily with dask and only the parts
  needed for the requested period and basin are transferred. Long periods on
  large basins (e.g. the whole Mediterranean Sea at 1 km) still require
  transferring a large amount of data.


## Project structure

```
src/pqtools/
├── __main__.py                    # Entry point of the `pqtools` command
└── components/
    ├── cli.py                     # Command line interface (argparse)
    ├── basins.py                  # Bounding boxes of the basins
    ├── data_retrievers.py         # Download and harmonisation of the data
    ├── basin_loader.py            # Loading of the basins (bitsea or YAML)
    ├── dataset_versions.py        # Versions of the Copernicus Marine datasets
    ├── basin_data.py              # Masked data of the sources on a basin
    ├── satellite_processing.py    # Moving geometric mean of the satellite data
    ├── masks.py                   # Land/sea masks, basin masks and cell areas
    ├── regridding.py              # Interpolation on the model grid
    ├── output.py                 # Writing of the (compressed) netCDF files
    ├── compute_timeseries.py      # Computation of the timeseries
    └── compute_comparison_maps.py # Computation of the comparison maps
```

The functions in these modules can also be used directly from Python, e.g.:

```python
from datetime import datetime

from bitsea.basins import Basin

from pqtools.components.basin_data import SatelliteDatasetOptions
from pqtools.components.compute_timeseries import compute_timeseries

timeseries = compute_timeseries(
    start_date=datetime(2025, 3, 1),
    end_date=datetime(2025, 3, 31),
    basin=Basin.load_from_uuid("adr1"),
    satellite_dataset_options=SatelliteDatasetOptions(half_window_size=3),
)
```
