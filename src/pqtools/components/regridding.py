from dataclasses import dataclass

import xarray as xr
import xarray_regrid  # noqa: F401 (registers the `regrid` accessor of xarray)

from pqtools.components.masks import _check_alignment

# Minimum fraction of the sea area of a model cell (according to the
# satellite mask) that must be covered by valid satellite data for the cell
# to receive a value; the other cells are set to NaN
MIN_SATELLITE_COVERAGE = 0.75

SPATIAL_DIMS = ("latitude", "longitude")


@dataclass(frozen=True, slots=True)
class RegriddedData:
    """Data interpolated on a coarser grid.

    Attributes:
        data: The interpolated chlorophyll, with dimensions
            ``(time, latitude, longitude)`` on the target grid.
        coverage: For each cell of the target grid and each day, the
            fraction (between 0 and 1) of the sea area of the cell that is
            covered by valid input data, with the same dimensions as
            `data`.
    """

    data: xr.DataArray
    coverage: xr.DataArray


def _regrid(data: xr.DataArray, target_grid: xr.Dataset) -> xr.DataArray:
    """Conservative regridding of `data`, weighting on the overlapping area.

    The value of each target cell is the average of the input cells that
    overlap it, each one weighted with the area of the overlap. `data` must
    not contain NaN values: they would propagate to the target cells.
    """
    return data.regrid.conservative(
        target_grid,
        latitude_coord="latitude",
        skipna=False,
    )


def regrid_on_mask_grid(
    data: xr.DataArray,
    data_mask: xr.DataArray,
    target_mask: xr.DataArray,
    min_coverage: float = MIN_SATELLITE_COVERAGE,
) -> RegriddedData:
    """Interpolates the chlorophyll from a fine grid to the grid of a mask.

    The value of each target cell is the arithmetic mean of the valid values
    of the input cells that overlap it, each one weighted with the area of
    the overlap (conservative regridding).

    The coverage of a target cell is the fraction of its sea area (i.e. the
    part of the cell where `data_mask` is True) covered by valid data. The
    land is not counted as missing data, so coastal cells are not
    penalised. The cells whose coverage is lower than `min_coverage` are set
    to NaN, as well as the cells outside `target_mask`.

    Args:
        data: The chlorophyll on the fine grid, with dimensions
            ``(time, latitude, longitude)``; NaN where there are no valid
            data (land, clouds, ...).
        data_mask: The land/sea mask of `data` (True on sea), on the same
            grid as `data`.
        target_mask: The mask of the target grid (True on the cells that
            must receive a value, e.g. the sea cells inside a basin). The
            fine grid should cover all its True cells: the parts of the
            cells that are not covered are ignored.
        min_coverage: The minimum coverage, between 0 and 1, required to
            assign a value to a cell.

    Returns:
        The interpolated data and their coverage, on the grid of
        `target_mask`. They are NaN outside `target_mask`; the data are also
        NaN where the coverage is lower than `min_coverage`. The
        computation is lazy if `data` is backed by dask.

    Raises:
        ValueError: If `min_coverage` is not between 0 and 1.
        MaskMisalignmentError: If `data` and `data_mask` are not on the same
            grid.
    """
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError(
            f"min_coverage must be between 0 and 1; received {min_coverage}"
        )
    _check_alignment(data, data_mask)

    target_grid = xr.Dataset(
        coords={dim: target_mask[dim] for dim in SPATIAL_DIMS}
    )

    # Fractions of the area of each target cell covered by valid data and by
    # sea; the inputs have no NaN, so every target cell overlapped by the
    # fine grid gets a value (0 where there are no valid data or no sea)
    valid_fraction = _regrid(data.notnull().astype(data.dtype), target_grid)
    sea_fraction = _regrid(data_mask.astype(data.dtype), target_grid)
    coverage = (valid_fraction / sea_fraction.where(sea_fraction > 0)).clip(
        max=1.0
    )

    # The mean of the valid values is computed here, instead of asking
    # xarray_regrid to skip the NaNs (skipna=True). In that case, indeed,
    # xarray_regrid divides by the valid fraction before masking the cells
    # where it is 0: the result is correct, but every target cell without
    # valid data (land or clouds) causes a 0 / 0, for which numpy raises an
    # "invalid value encountered in divide" RuntimeWarning. Here, instead,
    # these cells are set to NaN before the division. Moreover, the valid
    # fraction, which is also needed for the coverage, is computed only once
    mean = _regrid(data.fillna(0.0), target_grid) / valid_fraction.where(
        valid_fraction > 0
    )

    regridded = mean.where(coverage >= min_coverage)

    regridded = regridded.where(target_mask).assign_attrs(data.attrs)
    regridded.name = data.name
    coverage = coverage.where(target_mask)
    coverage.name = "coverage"
    coverage.attrs = {
        "long_name": "Fraction of the sea area of the cell covered by valid "
        "data",
        "units": "1",
    }
    return RegriddedData(data=regridded, coverage=coverage)
