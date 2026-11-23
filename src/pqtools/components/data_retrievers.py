from datetime import UTC
from datetime import datetime
from enum import Enum
from types import MappingProxyType

import copernicusmarine
import numpy as np
import xarray as xr

from pqtools.components.basins import BoundingBox
from pqtools.components.basins import CoordinatesSelectionMethod


class DataSource(Enum):
    REANALYSIS = "reanalysis"
    ANALYSIS = "analysis"
    SATELLITE = "satellite"

    def __str__(self) -> str:
        # Used by argparse in the help and in the error messages
        return self.value


DATASETS: MappingProxyType[DataSource, "str"] = MappingProxyType(
    {
        DataSource.REANALYSIS: "cmems_mod_med_bgc-plankton_my_4.2km_P1D-m",
        DataSource.ANALYSIS: "cmems_mod_med_bgc-pft_anfc_4.2km_P1D-m",
        DataSource.SATELLITE: "cmems_obs-oc_med_bgc-plankton_my_l3-multi-1km_P1D",
    }
)

# Name of the chlorophyll variable inside each dataset
CHLOROPHYLL_VARIABLES: MappingProxyType[DataSource, "str"] = MappingProxyType(
    {
        DataSource.REANALYSIS: "chl",
        DataSource.ANALYSIS: "chl",
        DataSource.SATELLITE: "CHL",
    }
)

# Name of the variable returned by `download_data`, whatever the source
OUTPUT_VARIABLE = "chl"

# Range of valid chlorophyll values (in mg/m^3) declared by the satellite
# product (`valid_min` and `valid_max` attributes of the CHL variable).
# xarray does not apply these attributes automatically, and the dataset
# does contain values outside this range
SATELLITE_VALID_MIN = 0.01
SATELLITE_VALID_MAX = 100.0

# Quality index of the satellite product: the anomaly of the daily value
# with respect to the climatology, in units of climatological standard
# deviations
SATELLITE_QI_VARIABLE = "QI_CHL"

# Units of the data returned by `download_data`, whatever the source
OUTPUT_UNITS = "mg m-3"

# Spellings of mg/m^3 that can be found in the `units` attribute of the
# datasets; all of them are rewritten as OUTPUT_UNITS
ACCEPTED_UNITS = frozenset(
    {
        "mg m-3",
        "mg m^-3",
        "mg/m3",
        "mg/m^3",
        "mg.m-3",
        "milligram m-3",
        "milligram m^-3",
    }
)


class UnexpectedUnitsError(ValueError):
    """Raised when a dataset provides chlorophyll in units other than mg/m^3."""


def _normalize_units(
    data: xr.DataArray, data_source: "DataSource"
) -> xr.DataArray:
    """Checks that `data` is in mg/m^3 and writes its units as OUTPUT_UNITS.

    Raises:
        UnexpectedUnitsError: If the `units` attribute is missing or is not
            one of the spellings of mg/m^3 in ACCEPTED_UNITS.
    """
    units = data.attrs.get("units")
    if units is None or units.strip() not in ACCEPTED_UNITS:
        raise UnexpectedUnitsError(
            f"The chlorophyll of the {data_source.value} dataset "
            f'"{DATASETS[data_source]}" has units {units!r}, while mg/m^3 '
            f"was expected (accepted spellings: {sorted(ACCEPTED_UNITS)})"
        )
    return data.assign_attrs(units=OUTPUT_UNITS)


def _as_utc(date: datetime) -> datetime:
    """Interprets naive datetimes as UTC (the reference of the datasets).

    Without this, copernicusmarine would interpret a naive datetime in the
    local timezone of the machine, shifting the requested interval (e.g. on
    a machine in CET, midnight of the last day would become 23:00 UTC of
    the previous day, and the last day would be excluded).
    """
    if date.tzinfo is None:
        return date.replace(tzinfo=UTC)
    return date


def download_data(
    start_date: datetime,
    end_date: datetime,
    region: BoundingBox,
    data_source: DataSource,
    satellite_max_qi: float = 0.0,
    coordinates_selection_method: CoordinatesSelectionMethod = "inside",
) -> xr.DataArray:
    """Retrieves the surface chlorophyll-a concentration from Copernicus Marine.

    Opens the daily chlorophyll dataset associated with `data_source` on the
    Copernicus Marine Service and subsets it in time and space. The three
    sources are harmonised so that they can be compared directly:

    * the variable is always named ``"chl"`` (the satellite dataset calls it
      ``"CHL"``);
    * for the model datasets (analysis and reanalysis), only the surface
      layer (the first depth level, about 1 m) is kept and the ``depth``
      coordinate is dropped, so that every source returns a 2D field for
      each day.

    For the satellite source, the data are also quality-checked:

    * the values outside the valid range declared by the product
      (`SATELLITE_VALID_MIN` - `SATELLITE_VALID_MAX`, i.e. 0.01 - 100
      mg/m^3) are always removed;
    * if `satellite_max_qi` is greater than 0, the values whose quality
      index (``QI_CHL``) is, in absolute value, strictly greater than
      `satellite_max_qi` are removed too, together with the values that
      have no quality index.

    Removed values are set to NaN, as the points where there are no data.

    The region is first intersected with the Mediterranean Sea domain (see
    `BoundingBox.cut_on_mediterranean_sea`), in the same way as it is done
    for the masks in `pqtools.components.masks.get_mask`.

    The data are opened lazily: the returned array is always backed by dask
    (also for small requests, which copernicusmarine would open without
    dask) and the values are actually transferred from the remote server
    only when they are needed (e.g. when calling ``.load()``, ``.values`` or when
    computing a statistic). Call ``.load()`` on the result to bring
    everything into memory at once.

    Note:
        The model datasets have a horizontal resolution of about 4.2 km,
        while the satellite one is at about 1 km: the returned arrays are
        therefore NOT defined on the same grid and must be regridded before
        being compared point by point.

        Accessing the Copernicus Marine Service requires valid credentials
        (see ``copernicusmarine login``).

    Args:
        start_date: First day of the requested period (inclusive). A naive
            datetime (without tzinfo) is interpreted as UTC, which is the
            time reference of the datasets.
        end_date: Last day of the requested period (inclusive). Must not be
            earlier than `start_date`. Naive datetimes are interpreted as
            UTC, as for `start_date`.
        region: Geographical area to retrieve. It is clipped to the
            Mediterranean Sea before sending the request.
        data_source: Which product to read the chlorophyll from: the model
            reanalysis (multi-year), the model analysis/forecast or the
            satellite L3 observations.
        satellite_max_qi: Threshold on the quality index of the satellite
            data. ``QI_CHL`` is the anomaly of the daily value with respect
            to the climatology, in units of climatological standard
            deviations, so it can be negative: the values with
            ``|QI_CHL| > satellite_max_qi`` are removed. The (few) values
            without a quality index are removed too, since their quality
            cannot be checked. If 0 (the default), no filter is applied on
            the quality index and the values without it are kept.
            It must not be negative. It is ignored if `data_source` is not
            `DataSource.SATELLITE`.
        coordinates_selection_method: How the points of `region` are
            selected (see `CoordinatesSelectionMethod`). The masks used with
            these data must be downloaded with the same method.

    Returns:
        A lazy `xarray.DataArray` named ``"chl"`` with dimensions
        ``(time, latitude, longitude)``, containing the daily mean
        chlorophyll-a concentration in mg/m^3. Its ``units`` attribute is
        always ``"mg m-3"``, independently of how the source dataset spells
        it (e.g. the satellite one uses ``"milligram m^-3"``). Points with
        no data (land, or clouds for the satellite product) are NaN. The
        other attributes of the dataset variable (standard_name,
        long_name, ...) are preserved.

    Raises:
        ValueError: If `start_date` is later than `end_date`, or if
            `satellite_max_qi` is negative or not finite.
        UnexpectedUnitsError: If the dataset does not declare its
            chlorophyll in mg/m^3 (or does not declare any units at all).
            This check is done on the metadata, before any data is
            transferred.
        copernicusmarine.DatasetNotFound: If the dataset associated with
            `data_source` does not exist in the Copernicus Marine catalogue.
        copernicusmarine.InvalidUsernameOrPassword: If the Copernicus Marine
            credentials are not valid.
    """
    if start_date > end_date:
        raise ValueError(
            f"start_date ({start_date}) must not be later than "
            f"end_date ({end_date})"
        )
    if not np.isfinite(satellite_max_qi) or satellite_max_qi < 0:
        raise ValueError(
            f"satellite_max_qi must be a finite number greater or equal "
            f"than 0; received {satellite_max_qi}"
        )

    variable = CHLOROPHYLL_VARIABLES[data_source]
    apply_qi_filter = (
        data_source is DataSource.SATELLITE and satellite_max_qi > 0
    )

    variables = [variable]
    if apply_qi_filter:
        variables.append(SATELLITE_QI_VARIABLE)

    dataset = copernicusmarine.open_dataset(
        dataset_id=DATASETS[data_source],
        variables=variables,
        start_datetime=_as_utc(start_date),
        end_datetime=_as_utc(end_date),
        **region.cut_on_mediterranean_sea().to_dict(),
        coordinates_selection_method=coordinates_selection_method,
    )

    # Only the surface layer of the model datasets is kept. This must be
    # done before chunking: otherwise, for data opened without dask, the
    # single chunk would contain (and download) all the depth levels
    if "depth" in dataset.dims:
        dataset = dataset.isel(depth=0, drop=True)

    # For small requests copernicusmarine opens the data without dask, and
    # the operations below would load them in memory; `.chunk()` converts
    # them to dask (with a single chunk), while it leaves unchanged the
    # chunks of the data that are already backed by dask
    dataset = dataset.chunk()

    chl = dataset[variable]

    chl = _normalize_units(chl, data_source)

    if data_source is DataSource.SATELLITE:
        chl = chl.where(
            (chl >= SATELLITE_VALID_MIN) & (chl <= SATELLITE_VALID_MAX)
        )
        if apply_qi_filter:
            # A NaN QI gives False in the comparison: the values without a
            # quality index are removed too
            chl = chl.where(
                abs(dataset[SATELLITE_QI_VARIABLE]) <= satellite_max_qi
            )

    return chl.rename(OUTPUT_VARIABLE)
