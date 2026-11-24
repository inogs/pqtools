from typing import cast

import numpy as np
import pandas as pd
import xarray as xr

# Minimum number of valid (non-NaN) observations that a cell must have inside
# the time window for the moving average to be computed; with fewer valid
# observations, the result for that cell is NaN
MOVING_AVERAGE_MIN_OBSERVATIONS = 1

ONE_DAY = np.timedelta64(1, "D")


def _to_daily_time_axis(data: xr.DataArray) -> xr.DataArray:
    """Reindexes `data` on a regular daily time axis.

    The days missing from the time axis are added and filled with NaN, so
    that a window of N elements along the time axis always spans N days.

    Raises:
        ValueError: If the time coordinate is not sorted, or if its values
            are not spaced by an integer number of days.
    """
    times = data["time"].values
    if times.size == 0:
        return data

    steps = np.diff(times)
    if np.any(steps <= np.timedelta64(0, "s")):
        raise ValueError("The time coordinate must be strictly increasing")
    if np.any(steps % ONE_DAY != np.timedelta64(0, "s")):
        raise ValueError(
            "The time coordinate must be spaced by an integer number of days"
        )

    daily_times = pd.date_range(times[0], times[-1], freq="D")
    return data.reindex(time=daily_times)


def _window_mean(
    values: np.ndarray, half_window: int, axis: int
) -> np.ndarray:
    """Computes a centered moving mean along `axis`, ignoring NaNs.

    The windows are truncated at the borders of the array. The elements whose
    window contains fewer than `MOVING_AVERAGE_MIN_OBSERVATIONS` valid values
    are NaN.
    """
    values = np.moveaxis(values, axis, 0)
    valid = ~np.isnan(values)
    filled = np.where(valid, values, 0)
    total = np.zeros_like(filled)
    count = np.zeros(values.shape, dtype=np.int16)
    size = values.shape[0]
    for shift in range(-half_window, half_window + 1):
        target = slice(max(0, -shift), min(size, size - shift))
        source = slice(max(0, shift), min(size, size + shift))
        if target.start < target.stop:
            total[target] += filled[source]
            count[target] += valid[source]
    mean = np.where(
        count >= MOVING_AVERAGE_MIN_OBSERVATIONS,
        total / np.maximum(count, 1),
        np.nan,
    ).astype(values.dtype)
    return np.moveaxis(mean, 0, axis)


def log_moving_average(data: xr.DataArray, half_window: int) -> xr.DataArray:
    """Computes a centered moving geometric mean in time, ignoring NaNs.

    For each cell and each day, the result is computed over the window made
    of the `half_window` days before, the day itself and the `half_window`
    days after (``2 * half_window + 1`` days in total):

    * if the cell has fewer than `MOVING_AVERAGE_MIN_OBSERVATIONS` valid
      (non-NaN) observations in the window (e.g. it is always NaN), the
      result is NaN;
    * otherwise, the result is the exponential of the mean of the logarithms
      of the valid observations, i.e. their geometric mean.

    The geometric mean is used instead of the arithmetic one because the
    chlorophyll concentration is approximately log-normally distributed: an
    arithmetic mean would be dominated by the few highest values in the
    window.

    Before computing the average, the time axis is made regular: the days
    missing from it are added and filled with NaN, so that the window always
    spans `2 * half_window + 1` calendar days. The returned array is defined
    on this daily time axis.

    At the beginning and at the end of the time axis the window is
    truncated (it only contains the available days). To have the full window
    on every day of a period ``[start, end]``, pass data covering
    ``[start - half_window, end + half_window]`` and select ``[start, end]``
    on the result.

    The computation is lazy if `data` is backed by dask.

    Args:
        data: The data to average, with a ``time`` dimension with daily
            values (e.g. the satellite data returned by
            `pqtools.components.data_retrievers.download_data`). All its
            valid values must be strictly positive; this is not checked, to
            avoid triggering the computation of lazy data.
        half_window: Number of days before and after each day included in
            the window. If 0, `data` is returned unchanged (without
            computing the exponential of the logarithm, to avoid rounding
            errors) and the time axis is not modified.

    Returns:
        An `xarray.DataArray` with the same name, dimensions and attributes
        as `data`, containing the moving geometric mean on a regular daily
        time axis spanning the same period as `data`.

    Raises:
        ValueError: If `half_window` is negative, if `data` has no ``time``
            dimension or if its time coordinate is not sorted or not spaced
            by an integer number of days.
    """
    if half_window < 0:
        raise ValueError(
            f"half_window must be greater or equal than 0; "
            f"received {half_window}"
        )
    if "time" not in data.dims:
        raise ValueError(
            f'data must have a "time" dimension; its dimensions are '
            f"{data.dims}"
        )

    if half_window == 0:
        return data

    daily_data = _to_daily_time_axis(data)

    log_data = cast(xr.DataArray, cast(object, np.log(daily_data)))
    time_axis = cast(int, log_data.get_axis_num("time"))

    # `DataArray.rolling` is not used because, on dask arrays, it splits the
    # chunks in space to compensate for the size of the window. On large
    # domains this produces a huge number of tasks, which makes the
    # computation slower and increases the memory usage
    if log_data.chunks is None:
        mean_values = _window_mean(log_data.values, half_window, time_axis)
    else:
        mean_values = log_data.data.map_overlap(
            _window_mean,
            half_window=half_window,
            axis=time_axis,
            depth={time_axis: half_window},
            boundary=np.nan,
            dtype=log_data.dtype,
            meta=np.empty((0,) * log_data.ndim, dtype=log_data.dtype),
        )
    log_mean = log_data.copy(data=mean_values)

    averaged = np.exp(log_mean)
    averaged.name = data.name
    averaged.attrs = {
        **data.attrs,
        "moving_average": (
            f"centered moving geometric mean over {2 * half_window + 1} "
            f"days, computed on the valid values (at least "
            f"{MOVING_AVERAGE_MIN_OBSERVATIONS} per window)"
        ),
    }
    return averaged
