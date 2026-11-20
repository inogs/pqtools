from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta

import numpy as np
import xarray as xr
from bitsea.basins import Basin

from pqtools.components.basins import basin_to_bounding_box
from pqtools.components.data_retrievers import DataSource
from pqtools.components.data_retrievers import download_data
from pqtools.components.masks import apply_mask
from pqtools.components.masks import compute_cell_area
from pqtools.components.masks import cut_mask_on_basin
from pqtools.components.masks import get_model_mask
from pqtools.components.masks import get_satellite_mask
from pqtools.components.satellite_processing import log_moving_average


@dataclass(frozen=True, slots=True)
class TimeseriesOptions:
    half_satellite_window_size: int = 0
    satellite_max_qi: float = 0.0


def _as_naive_utc(date: datetime) -> np.datetime64:
    """Converts `date` to a numpy datetime64 in UTC, as the dataset times.

    Naive datetimes are interpreted as UTC.
    """
    if date.tzinfo is not None:
        date = date.astimezone(UTC).replace(tzinfo=None)
    return np.datetime64(date)


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
    options: TimeseriesOptions | None = None,
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
    their quality index (if `options.satellite_max_qi` is greater than 0)
    and averaged in time with a moving geometric mean (if
    `options.half_satellite_window_size` is greater than 0).

    Args:
        start_date: First day of the timeseries (inclusive).
        end_date: Last day of the timeseries (inclusive).
        basin: The basin on which the statistics are computed.
        options: Options for the processing of the satellite data. If None,
            the default options are used (no filter on the quality index
            and no moving average).

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
    if options is None:
        options = TimeseriesOptions()

    bounding_box = basin_to_bounding_box(basin)

    datasets = {}
    for data_source in DataSource:
        if data_source is DataSource.SATELLITE:
            # The satellite data are downloaded on a period extended by the
            # half-width of the moving average on both sides, so that every
            # day in [start_date, end_date] has its full window
            window_extension = timedelta(
                days=options.half_satellite_window_size
            )
            satellite_data = download_data(
                start_date=start_date - window_extension,
                end_date=end_date + window_extension,
                region=bounding_box,
                data_source=data_source,
                satellite_max_qi=options.satellite_max_qi,
            )
            datasets[data_source] = log_moving_average(
                satellite_data,
                half_window=options.half_satellite_window_size,
            ).sel(
                time=slice(_as_naive_utc(start_date), _as_naive_utc(end_date))
            )
        else:
            datasets[data_source] = download_data(
                start_date=start_date,
                end_date=end_date,
                region=bounding_box,
                data_source=data_source,
            )

    regional_model_mask = get_model_mask(bounding_box=bounding_box)
    regional_satellite_mask = get_satellite_mask(bounding_box=bounding_box)

    model_mask = cut_mask_on_basin(mask=regional_model_mask, basin=basin)
    satellite_mask = cut_mask_on_basin(
        mask=regional_satellite_mask, basin=basin
    )

    masks = {
        DataSource.ANALYSIS: model_mask,
        DataSource.REANALYSIS: model_mask,
        DataSource.SATELLITE: satellite_mask,
    }

    # The values outside the mask of their source (land or outside the
    # basin) are set to NaN
    datasets = {
        data_source: apply_mask(data, masks[data_source])
        for data_source, data in datasets.items()
    }

    model_cell_area = compute_cell_area(model_mask)
    cell_areas = {
        DataSource.ANALYSIS: model_cell_area,
        DataSource.REANALYSIS: model_cell_area,
        DataSource.SATELLITE: compute_cell_area(satellite_mask),
    }

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
        satellite_moving_average_half_window=options.half_satellite_window_size,
        satellite_max_qi=options.satellite_max_qi,
    )

    # All the variables are computed together, so that the data shared
    # among them are downloaded only once
    return timeseries.compute()
