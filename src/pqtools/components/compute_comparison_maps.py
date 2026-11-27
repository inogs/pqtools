from datetime import datetime

import numpy as np
import xarray as xr
from bitsea.basins import Basin

from pqtools.components.basin_data import MODEL_SOURCES
from pqtools.components.basin_data import SatelliteDatasetOptions
from pqtools.components.basin_data import download_basin_data
from pqtools.components.basin_data import source_dataset_ids
from pqtools.components.basins import rectangle_around_basin
from pqtools.components.data_retrievers import DataSource
from pqtools.components.dataset_versions import describe_dataset_versions
from pqtools.components.masks import compute_cell_area
from pqtools.components.regridding import DEFAULT_MIN_SATELLITE_COVERAGE
from pqtools.components.regridding import SPATIAL_DIMS
from pqtools.components.regridding import regrid_on_mask_grid

# Resolution of the model grid, in degrees
MODEL_GRID_STEP = 1 / 24

# Margin, in degrees, added on each side of the bounding box of the basin to
# build the area where the satellite data are downloaded. The satellite data
# are interpolated on the model grid, and the model cells on the border of
# the basin extend half a cell beyond their center: a full model cell gives
# them complete coverage, with some tolerance
SATELLITE_MARGIN = MODEL_GRID_STEP

DIFFERENCE_VARIABLE = "difference"
MEAN_DIFFERENCE_VARIABLE = "mean_difference"
RMS_DIFFERENCE_VARIABLE = "rms_difference"
REFERENCE_COVERAGE_VARIABLE = "reference_coverage"


def compute_comparison_maps(
    evaluated_dataset: DataSource,
    reference_dataset: DataSource,
    start_date: datetime,
    end_date: datetime,
    basin: Basin,
    satellite_dataset_options: SatelliteDatasetOptions | None = None,
    satellite_min_coverage: float = DEFAULT_MIN_SATELLITE_COVERAGE,
) -> xr.Dataset:
    """Computes the maps comparing a model dataset with a reference dataset.

    For each day, the difference ``evaluated - reference`` is computed on
    all the cells of the model grid that are valid (non-NaN) in both the
    datasets. If the reference is the satellite, its data are first
    interpolated on the model grid (see `regrid_on_mask_grid`). The mean and
    the root mean square of the differences are then computed on the valid
    cells of each day, weighting each cell with its area.

    Args:
        evaluated_dataset: The dataset to evaluate; it must be a model
            product (``DataSource.ANALYSIS`` or ``DataSource.REANALYSIS``).
        reference_dataset: The dataset used as reference; any source is
            valid (including ``DataSource.SATELLITE``), as long as it is
            different from `evaluated_dataset`.
        start_date: First day of the period (inclusive).
        end_date: Last day of the period (inclusive).
        basin: The basin on which the maps are computed.
        satellite_dataset_options: Options for the processing of the
            satellite data, used only if `reference_dataset` is
            ``DataSource.SATELLITE``. If None, the default options are used
            (no filter on the quality index and a moving average over 7
            days).
        satellite_min_coverage: The minimum fraction, between 0 and 1, of
            the sea area of a model cell that must be covered by valid
            satellite data; the other cells are set to NaN. It is used only
            if `reference_dataset` is ``DataSource.SATELLITE``.

    Returns:
        A `xarray.Dataset` with the variables (in mg m-3, except the
        coverage):

        * ``difference``, with dimensions ``(time, latitude, longitude)``:
          the difference ``evaluated - reference``, NaN on the cells that
          are not valid in both the datasets;
        * ``mean_difference``, with dimension ``time``: the weighted spatial
          mean of the differences (i.e. the bias);
        * ``rms_difference``, with dimension ``time``: the square root of
          the weighted spatial mean of the squared differences (the bias is
          not removed);
        * ``reference_coverage``, with dimensions
          ``(time, latitude, longitude)``, only if the reference is the
          satellite: the fraction (between 0 and 1) of the sea area of each
          cell covered by valid satellite data. It is defined on all the
          cells of the basin, also the ones where the difference is NaN
          because the coverage is too low.

        The days without any valid cell (e.g. the ones available in only
        one of the datasets) are NaN. The values are computed (not lazy).

    Raises:
        ValueError: If `evaluated_dataset` is not a model product, if
            `reference_dataset` is the same as `evaluated_dataset`, if
            `start_date` is later than `end_date` or if
            `satellite_min_coverage` is not between 0 and 1.
    """
    if evaluated_dataset not in MODEL_SOURCES:
        raise ValueError(
            f"evaluated_dataset must be one of "
            f"{sorted(s.value for s in MODEL_SOURCES)}; received "
            f"{evaluated_dataset.value!r}"
        )
    if reference_dataset is evaluated_dataset:
        raise ValueError(
            f"reference_dataset must be different from evaluated_dataset; "
            f"both are {evaluated_dataset.value!r}"
        )
    if start_date > end_date:
        raise ValueError(
            f"start_date ({start_date}) must not be later than end_date "
            f"({end_date})"
        )

    if not 0.0 <= satellite_min_coverage <= 1.0:
        raise ValueError(
            f"satellite_min_coverage must be between 0 and 1; received "
            f"{satellite_min_coverage}"
        )

    if satellite_dataset_options is None:
        satellite_dataset_options = SatelliteDatasetOptions()

    basin_data = download_basin_data(
        data_sources=(evaluated_dataset, reference_dataset),
        start_date=start_date,
        end_date=end_date,
        basin=basin,
        satellite_dataset_options=satellite_dataset_options,
        satellite_basin=rectangle_around_basin(basin, SATELLITE_MARGIN),
        # Where the rectangle is clipped by the borders of the domain (e.g.
        # at the Strait of Gibraltar), "outside" includes also the first
        # satellite point beyond the border, which covers part of the
        # outer half of the model cells on the border
        satellite_coordinates_selection_method="outside",
    )
    evaluated = basin_data[evaluated_dataset].data
    if reference_dataset is DataSource.SATELLITE:
        # The satellite data are interpolated on the model grid and cut on
        # the basin with the model mask, so that the two datasets are
        # compared on the same cells
        satellite = basin_data[DataSource.SATELLITE]
        regridded = regrid_on_mask_grid(
            data=satellite.data,
            data_mask=satellite.mask,
            target_mask=basin_data[evaluated_dataset].mask,
            min_coverage=satellite_min_coverage,
        )
        reference = regridded.data
        reference_coverage = regridded.coverage
    else:
        reference = basin_data[reference_dataset].data
        reference_coverage = None

    # The two datasets are on the same spatial grid; the days available in
    # only one of them are kept, with NaN in the other one
    evaluated, reference = xr.align(evaluated, reference, join="outer")

    difference = evaluated - reference

    # xarray does not accept NaN in the weights. The NaN values of the
    # difference are excluded, together with their weights, by xarray
    weights = compute_cell_area(basin_data[evaluated_dataset].mask).fillna(0.0)
    mean_difference = difference.weighted(weights).mean(dim=SPATIAL_DIMS)
    rms_difference = np.sqrt(
        (difference**2).weighted(weights).mean(dim=SPATIAL_DIMS)
    )

    units = evaluated.attrs["units"]
    names = f"{evaluated_dataset.value} - {reference_dataset.value}"
    # The attributes are replaced (and not updated) because the ones inherited
    # from the chlorophyll (e.g. its standard name) do not describe a
    # difference
    difference.attrs = {
        "long_name": f"Difference of the chlorophyll ({names})",
        "units": units,
    }
    mean_difference.attrs = {
        "long_name": f"Spatial mean of the difference of the chlorophyll "
        f"({names}), weighted on the area of the cells",
        "units": units,
    }
    rms_difference.attrs = {
        "long_name": f"Spatial root mean square of the difference of the "
        f"chlorophyll ({names}), weighted on the area of the cells",
        "units": units,
    }

    attrs = {
        "basin": basin.get_uuid(),
        "evaluated_dataset": evaluated_dataset.value,
        "reference_dataset": reference_dataset.value,
        "source_datasets": describe_dataset_versions(
            source_dataset_ids((evaluated_dataset, reference_dataset))
        ),
    }
    if reference_dataset is DataSource.SATELLITE:
        attrs["satellite_moving_average_half_window"] = (
            satellite_dataset_options.half_window_size
        )
        attrs["satellite_max_qi"] = satellite_dataset_options.max_qi
        attrs["satellite_min_coverage"] = satellite_min_coverage

    comparison_maps = xr.Dataset(
        {
            DIFFERENCE_VARIABLE: difference,
            MEAN_DIFFERENCE_VARIABLE: mean_difference,
            RMS_DIFFERENCE_VARIABLE: rms_difference,
        },
        attrs=attrs,
    )

    if reference_coverage is not None:
        comparison_maps[REFERENCE_COVERAGE_VARIABLE] = reference_coverage

    # All the variables are computed together, so that the data shared
    # among them are downloaded only once
    return comparison_maps.compute()
