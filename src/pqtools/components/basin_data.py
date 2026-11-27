from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta

import numpy as np
import xarray as xr
from bitsea.basins import Basin

from pqtools.components.basins import CoordinatesSelectionMethod
from pqtools.components.basins import basin_to_bounding_box
from pqtools.components.data_retrievers import DATASETS
from pqtools.components.data_retrievers import DataSource
from pqtools.components.data_retrievers import download_data
from pqtools.components.masks import ANALYSIS_DATASET
from pqtools.components.masks import REANALYSIS_DATASET
from pqtools.components.masks import SATELLITE_L4_DATASET
from pqtools.components.masks import apply_mask
from pqtools.components.masks import cut_mask_on_basin
from pqtools.components.masks import get_model_mask
from pqtools.components.masks import get_satellite_mask
from pqtools.components.satellite_processing import log_moving_average

MODEL_SOURCES = frozenset({DataSource.ANALYSIS, DataSource.REANALYSIS})


@dataclass(frozen=True, slots=True)
class SatelliteDatasetOptions:
    """Options for the processing of the satellite data.

    Attributes:
        half_window_size: Half-width, in days, of the centered moving
            geometric mean applied to the satellite data (the window covers
            ``2 * half_window_size + 1`` days, i.e. 7 days with the default
            value 3); 0 disables it.
        max_qi: Threshold on the quality index of the satellite data: the
            values with ``|QI_CHL| > max_qi`` are removed (see
            `download_data`); 0 disables the filter.
    """

    half_window_size: int = 3
    max_qi: float = 0.0


@dataclass(frozen=True, slots=True)
class BasinData:
    """The data of a source restricted to a basin.

    Attributes:
        data: The surface chlorophyll, with dimensions
            ``(time, latitude, longitude)``, NaN outside `mask`.
        mask: The mask of the source restricted to the basin (True on the
            sea points inside the basin), on the same grid as `data`.
    """

    data: xr.DataArray
    mask: xr.DataArray


def as_naive_utc(date: datetime) -> np.datetime64:
    """Converts `date` to a numpy datetime64 in UTC, as the dataset times.

    Naive datetimes are interpreted as UTC.
    """
    if date.tzinfo is not None:
        date = date.astimezone(UTC).replace(tzinfo=None)
    return np.datetime64(date)


def _download_satellite_data(
    start_date: datetime,
    end_date: datetime,
    basin: Basin,
    options: SatelliteDatasetOptions,
    coordinates_selection_method: CoordinatesSelectionMethod,
) -> xr.DataArray:
    # The satellite data are downloaded on a period extended by the
    # half-width of the moving average on both sides, so that every day in
    # [start_date, end_date] has its full window
    window_extension = timedelta(days=options.half_window_size)
    satellite_data = download_data(
        start_date=start_date - window_extension,
        end_date=end_date + window_extension,
        region=basin_to_bounding_box(basin),
        data_source=DataSource.SATELLITE,
        satellite_max_qi=options.max_qi,
        coordinates_selection_method=coordinates_selection_method,
    )
    return log_moving_average(
        satellite_data, half_window=options.half_window_size
    ).sel(time=slice(as_naive_utc(start_date), as_naive_utc(end_date)))


def source_dataset_ids(data_sources: Iterable[DataSource]) -> list[str]:
    """Returns the ids of the datasets read by `download_basin_data`.

    For each source, these are the dataset of its data and the datasets used
    to build its mask: the static datasets of both the models for the model
    sources, the L4 product for the satellite.

    Args:
        data_sources: The sources passed to `download_basin_data`.

    Returns:
        The ids of the datasets, without duplicates, in the order of
        `data_sources`.
    """
    dataset_ids = []
    for data_source in data_sources:
        dataset_ids.append(DATASETS[data_source])
        if data_source in MODEL_SOURCES:
            dataset_ids.extend((ANALYSIS_DATASET, REANALYSIS_DATASET))
        else:
            dataset_ids.append(SATELLITE_L4_DATASET)
    return list(dict.fromkeys(dataset_ids))


def download_basin_data(
    data_sources: Iterable[DataSource],
    start_date: datetime,
    end_date: datetime,
    basin: Basin,
    satellite_dataset_options: SatelliteDatasetOptions | None = None,
    satellite_basin: Basin | None = None,
    satellite_coordinates_selection_method: CoordinatesSelectionMethod = (
        "inside"
    ),
) -> dict[DataSource, BasinData]:
    """Downloads the surface chlorophyll of some sources on a basin.

    For each source, the data are downloaded on the bounding box of `basin`
    and the values outside the mask of the source (land, or outside the
    basin) are set to NaN. The model sources (analysis and reanalysis)
    share the same mask, where a point is sea only if it is sea in both;
    the satellite source uses its own mask, on its own (finer) grid. Only
    the masks needed by `data_sources` are downloaded.

    The satellite data can be cut on a different basin (`satellite_basin`)
    than the model data. This is useful when the satellite data are later
    interpolated on the model grid: the model cells on the border of
    `basin` also cover satellite points that are outside of it, and a
    larger satellite basin (e.g. a rectangle around `basin`, see
    `rectangle_around_basin`) keeps them.

    The satellite data are also filtered on their quality index and
    averaged in time with a moving geometric mean, according to
    `satellite_dataset_options`; these options are ignored for the other
    sources.

    Args:
        data_sources: The sources to download; duplicates are ignored.
        start_date: First day of the period (inclusive).
        end_date: Last day of the period (inclusive).
        basin: The basin on which the data are cut.
        satellite_dataset_options: Options for the processing of the
            satellite data. If None, the default options are used (no
            filter on the quality index and a moving average over 7 days).
        satellite_basin: The basin on which the satellite data are
            downloaded and cut. If None, `basin` is used.
        satellite_coordinates_selection_method: How the points of the
            bounding box of `satellite_basin` are selected for the satellite
            data and mask (see `CoordinatesSelectionMethod`). The model data
            always use "inside".

    Returns:
        A dictionary that associates each source in `data_sources` with
        its masked data and its mask. The model sources share the same
        mask object. The data are lazy (backed by dask), while the masks
        have already been computed.
    """
    if satellite_dataset_options is None:
        satellite_dataset_options = SatelliteDatasetOptions()

    if satellite_basin is None:
        satellite_basin = basin

    data_sources = list(dict.fromkeys(data_sources))
    bounding_box = basin_to_bounding_box(basin)

    masks = {}
    if any(source in MODEL_SOURCES for source in data_sources):
        model_mask = cut_mask_on_basin(
            mask=get_model_mask(bounding_box=bounding_box), basin=basin
        )
        for source in MODEL_SOURCES:
            masks[source] = model_mask
    if DataSource.SATELLITE in data_sources:
        masks[DataSource.SATELLITE] = cut_mask_on_basin(
            mask=get_satellite_mask(
                bounding_box=basin_to_bounding_box(satellite_basin),
                coordinates_selection_method=(
                    satellite_coordinates_selection_method
                ),
            ),
            basin=satellite_basin,
        )

    basin_data = {}
    for data_source in data_sources:
        if data_source is DataSource.SATELLITE:
            data = _download_satellite_data(
                start_date=start_date,
                end_date=end_date,
                basin=satellite_basin,
                options=satellite_dataset_options,
                coordinates_selection_method=(
                    satellite_coordinates_selection_method
                ),
            )
        else:
            data = download_data(
                start_date=start_date,
                end_date=end_date,
                region=bounding_box,
                data_source=data_source,
            )
        basin_data[data_source] = BasinData(
            data=apply_mask(data, masks[data_source]),
            mask=masks[data_source],
        )

    return basin_data
