from datetime import datetime

import xarray as xr
from bitsea.basins import Basin

from pqtools.components.basin_data import SatelliteDatasetOptions
from pqtools.components.basin_data import download_basin_data
from pqtools.components.data_retrievers import DataSource
from pqtools.components.masks import compute_cell_area

SATELLITE_COVERAGE_VARIABLE = "satellite_coverage"


def _weighted_mean_and_std(
    data: xr.DataArray,
    cell_area: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Computes the spatial mean and standard deviation weighted on the area.

    For each day, only the valid (non-NaN) values of `data` are considered,
    each one weighted with the area of its cell:

    * mean = sum(area * value) / sum(area)
    * std = sqrt(sum(area * (value - mean)^2) / sum(area))

    The standard deviation is the one of the (weighted) population, i.e.
    without any correction for the number of degrees of freedom. On the days
    without valid values, both are NaN.

    Args:
        data: The data, with dimensions ``(time, latitude, longitude)``.
        cell_area: The area of the cells, NaN outside the mask (as returned
            by `compute_cell_area`), on the same grid as `data`.

    Returns:
        The mean and the standard deviation, with dimension ``time``.
    """
    # xarray does not accept NaN in the weights: the cells outside the mask
    # get a weight of 0. The NaN values of the data are already excluded,
    # together with their weights, by xarray.
    # The computation is done in float32 (as the data and the areas): on the
    # whole Mediterranean Sea (~2 million satellite cells), the relative
    # difference with respect to float64 is ~1e-6 for the mean and ~5e-4
    # for the std, i.e. of the same order of the accuracy of the areas
    # computed by bitsea and negligible with respect to the uncertainty of
    # the data, while float32 halves the memory
    weights = cell_area.fillna(0.0)
    weighted = data.weighted(weights)
    spatial_dims = ("latitude", "longitude")
    return weighted.mean(dim=spatial_dims), weighted.std(dim=spatial_dims)


def _coverage(data: xr.DataArray, cell_area: xr.DataArray) -> xr.DataArray:
    """Computes the percentage of the area of the mask with valid data.

    Args:
        data: The data, with dimensions ``(time, latitude, longitude)``.
        cell_area: The area of the cells, NaN outside the mask (as returned
            by `compute_cell_area`), on the same grid as `data`.

    Returns:
        For each day, the ratio (in %) between the total area of the cells
        with valid (non-NaN) data and the total area of the cells of the
        mask, with dimension ``time``.
    """
    spatial_dims = ("latitude", "longitude")
    total_area = cell_area.sum(dim=spatial_dims)
    valid_area = cell_area.where(data.notnull()).sum(dim=spatial_dims)
    return 100.0 * valid_area / total_area


def compute_timeseries(
    start_date: datetime,
    end_date: datetime,
    basin: Basin,
    satellite_dataset_options: SatelliteDatasetOptions | None = None,
) -> xr.Dataset:
    """Computes the daily timeseries of the chlorophyll on a basin.

    For each source (reanalysis, analysis and satellite), the surface
    chlorophyll is downloaded on the bounding box of `basin`, the values
    outside the mask of the source (land, or outside the basin) are set to
    NaN and, for each day, the spatial mean and standard deviation are
    computed, weighting each cell with its area. Only the valid values of
    each day are used: e.g. for the satellite, the cells covered by clouds
    are ignored.

    Before computing the statistics, the satellite data are filtered on
    their quality index (if `satellite_dataset_options.max_qi` is greater
    than 0) and averaged in time with a moving geometric mean (if
    `satellite_dataset_options.half_window_size` is greater than 0).

    Args:
        start_date: First day of the timeseries (inclusive).
        end_date: Last day of the timeseries (inclusive).
        basin: The basin on which the statistics are computed.
        satellite_dataset_options: Options for the processing of the
            satellite data. If None, the default options are used (no
            filter on the quality index and a moving average over 7 days).

    Returns:
        A `xarray.Dataset` with dimension ``time`` and the variables:

        * ``<source>_mean`` and ``<source>_std``, for each source in
          `DataSource` (``<source>`` is its value, e.g. ``reanalysis_mean``
          or ``satellite_std``): the weighted spatial mean and standard
          deviation of the chlorophyll, in mg m-3;
        * ``satellite_coverage``: the percentage of the area of the basin
          (according to the satellite mask) covered by valid satellite
          data.

        The days in which a source has no data are NaN for that source. The
        values are computed (not lazy).
    """
    if satellite_dataset_options is None:
        satellite_dataset_options = SatelliteDatasetOptions()

    basin_data = download_basin_data(
        data_sources=DataSource,
        start_date=start_date,
        end_date=end_date,
        basin=basin,
        satellite_dataset_options=satellite_dataset_options,
    )

    datasets = {
        data_source: source_data.data
        for data_source, source_data in basin_data.items()
    }

    # The area is computed only once for each mask (the model sources
    # share the same one)
    areas_by_mask = {}
    cell_areas = {}
    for data_source, source_data in basin_data.items():
        mask_id = id(source_data.mask)
        if mask_id not in areas_by_mask:
            areas_by_mask[mask_id] = compute_cell_area(source_data.mask)
        cell_areas[data_source] = areas_by_mask[mask_id]

    variables = {}
    for data_source, data in datasets.items():
        mean, std = _weighted_mean_and_std(data, cell_areas[data_source])
        variables[f"{data_source.value}_mean"] = mean.assign_attrs(
            long_name=f"Spatial mean of the {data_source.value} chlorophyll, "
            f"weighted on the area of the cells",
            units=data.attrs["units"],
        )
        variables[f"{data_source.value}_std"] = std.assign_attrs(
            long_name=f"Spatial standard deviation of the "
            f"{data_source.value} chlorophyll, weighted on the "
            f"area of the cells",
            units=data.attrs["units"],
        )

    variables[SATELLITE_COVERAGE_VARIABLE] = _coverage(
        datasets[DataSource.SATELLITE], cell_areas[DataSource.SATELLITE]
    ).assign_attrs(
        long_name="Percentage of the area of the basin covered by valid "
        "satellite data",
        units="%",
    )

    # The variables are aligned on their time coordinate: if a source has no
    # data on some days, its variables are NaN on those days
    timeseries = xr.Dataset(variables).assign_attrs(
        basin=basin.get_uuid(),
        satellite_moving_average_half_window=(
            satellite_dataset_options.half_window_size
        ),
        satellite_max_qi=satellite_dataset_options.max_qi,
    )

    # All the variables are computed together, so that the data shared
    # among them are downloaded only once
    return timeseries.compute()
