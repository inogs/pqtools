import numpy as np
import pytest
import xarray as xr

from pqtools.components.regridding import regrid_on_mask_grid

DIMS = ("latitude", "longitude")
RATIO = 4
N = 6
STEP = 1 / 24


def build_inputs(n=N):
    """Fine data on a grid whose cells are exactly nested in the target."""
    target = 44.0 + STEP * np.arange(n)
    fine = 44.0 - STEP / 2 + STEP / RATIO * (np.arange(n * RATIO) + 0.5)
    data_mask = xr.DataArray(
        np.ones((fine.size, fine.size), dtype=bool),
        dims=DIMS,
        coords={"latitude": fine, "longitude": fine},
    )
    target_mask = xr.DataArray(
        np.ones((n, n), dtype=bool),
        dims=DIMS,
        coords={"latitude": target, "longitude": target},
    )
    data = xr.DataArray(
        np.ones((1, fine.size, fine.size), dtype="float32"),
        dims=("time", *DIMS),
        coords={"time": [0], **data_mask.coords},
    )
    return data, data_mask, target_mask


def test_min_coverage():
    data, data_mask, target_mask = build_inputs()
    # Cell (0, 0) has no valid data; half of the longitudes of the cell
    # (1, 1) are valid; the other cells are completely covered
    data[0, :RATIO, :RATIO] = np.nan
    data[0, RATIO : 2 * RATIO, RATIO : RATIO + RATIO // 2] = np.nan

    for min_coverage, half_is_valid in (
        (0.0, True),
        (0.4, True),
        (0.5, True),
        (0.6, False),
        (1.0, False),
    ):
        result = regrid_on_mask_grid(
            data, data_mask, target_mask, min_coverage
        )
        expected = np.ones((N, N), dtype=bool)
        expected[0, 0] = False
        expected[1, 1] = half_is_valid
        np.testing.assert_array_equal(result.data[0].notnull(), expected)
        assert result.coverage[0, 0, 0] == 0
        assert result.coverage[0, 1, 1] == pytest.approx(0.5, abs=1e-5)


def test_full_coverage_with_rounding_errors():
    data, data_mask, target_mask = build_inputs(n=60)
    # A random land mask: every cell is completely covered on its sea
    rng = np.random.default_rng(0)
    data_mask = data_mask & (rng.random(data_mask.shape) > 0.3)
    # The rounding errors appear when the data and the mask have different
    # chunks, because their values are added in a different order
    data = data.where(data_mask).chunk(latitude=50, longitude=70)
    result = regrid_on_mask_grid(
        data, data_mask, target_mask, min_coverage=1.0
    )
    has_sea = result.coverage.notnull()
    assert has_sea.any()
    assert result.data.notnull().equals(has_sea)


@pytest.mark.parametrize("min_coverage", [-0.1, 1.1, np.nan])
def test_invalid_min_coverage(min_coverage):
    with pytest.raises(ValueError, match="min_coverage"):
        regrid_on_mask_grid(*build_inputs(), min_coverage)
