import netCDF4
import numpy as np
import xarray as xr

from pqtools.components.output import COMPRESSION_LEVEL
from pqtools.components.output import write_netcdf


def test_write_netcdf(tmp_path):
    rng = np.random.default_rng(0)
    maps = rng.random((4, 30, 40), dtype="float32")
    maps[:, :15] = np.nan
    dataset = xr.Dataset(
        {
            "difference": (("time", "latitude", "longitude"), maps),
            "mean_difference": (("time",), np.arange(4, dtype="float32")),
        },
        coords={"time": np.arange(4)},
        attrs={"basin": "adr"},
    )
    output_file = tmp_path / "output.nc"

    write_netcdf(dataset, output_file)

    with xr.open_dataset(output_file) as result:
        xr.testing.assert_identical(result.load(), dataset)
    with netCDF4.Dataset(output_file) as nc:
        filters = nc["difference"].filters()
        assert filters["zlib"] and filters["complevel"] == COMPRESSION_LEVEL
        assert not nc["mean_difference"].filters()["zlib"]
