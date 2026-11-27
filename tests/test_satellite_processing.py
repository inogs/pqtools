import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pqtools.components.satellite_processing import _merge_small_chunks
from pqtools.components.satellite_processing import log_moving_average


def _reference(data: xr.DataArray, half_window: int) -> np.ndarray:
    """Straightforward implementation, used as a reference."""
    times = pd.date_range(
        data["time"].values[0], data["time"].values[-1], freq="D"
    )
    values = data.reindex(time=times).values
    expected = np.full(values.shape, np.nan)
    for t in range(values.shape[0]):
        window = values[max(0, t - half_window) : t + half_window + 1]
        valid = ~np.isnan(window)
        count = valid.sum(axis=0)
        total = np.where(valid, np.log(np.where(valid, window, 1)), 0).sum(0)
        ok = count > 0
        expected[t][ok] = np.exp(total[ok] / count[ok])
    return expected


@pytest.fixture
def data() -> xr.DataArray:
    rng = np.random.default_rng(0)
    values = np.exp(rng.normal(size=(40, 5, 6))).astype("float32")
    values[rng.random(values.shape) < 0.7] = np.nan
    values[:, 0, 0] = np.nan
    array = xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={"time": pd.date_range("2025-01-01", periods=40)},
        name="chl",
    )
    # Some days are missing
    return array.isel(time=[t for t in range(40) if t not in (10, 11, 25)])


@pytest.mark.parametrize("half_window", [1, 3])
# With 40 days, a chunk of 13 days leaves a last chunk of 1 day, smaller than
# the overlap (as it happens with the real data)
@pytest.mark.parametrize("time_chunk", [None, 2, 7, 10, 13, 36])
def test_log_moving_average(data, half_window, time_chunk):
    if time_chunk is not None:
        data_in = data.chunk(time=time_chunk, latitude=2)
    else:
        data_in = data
    result = log_moving_average(data_in, half_window)
    assert (result.chunks is None) == (time_chunk is None)
    assert result.dtype == data.dtype
    np.testing.assert_allclose(
        result.values, _reference(data, half_window), rtol=1e-5
    )


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        ((10, 10, 10, 1), (10, 10, 11)),
        ((2, 2, 2, 2, 2), (4, 6)),
        ((1, 10, 10), (11, 10)),
        ((10, 2), (12,)),
        ((3, 3, 3), (3, 3, 3)),
        ((2,), (2,)),
    ],
)
def test_merge_small_chunks(chunks, expected):
    assert _merge_small_chunks(chunks, 3) == expected
