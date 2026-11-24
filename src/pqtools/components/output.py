from pathlib import Path

import xarray as xr

# Level of the (lossless) zlib compression of the output files, from 1 to 9.
# On the maps of the whole Mediterranean Sea, level 4 reduces the size of the
# files by ~83%; higher levels are much slower
COMPRESSION_LEVEL = 4


def write_netcdf(dataset: xr.Dataset, output_file: Path) -> None:
    """Writes a dataset in a compressed netCDF-4 file.

    Only the variables with more than one dimension (the maps) are
    compressed: the timeseries are so small that there is nothing to gain.
    The compression is lossless.

    Args:
        dataset: The dataset to write.
        output_file: The path of the output file; it is overwritten if it
            already exists.
    """
    encoding = {
        name: {
            "zlib": True,
            "complevel": COMPRESSION_LEVEL,
            "shuffle": True,
        }
        for name, variable in dataset.data_vars.items()
        if variable.ndim > 1
    }
    dataset.to_netcdf(output_file, format="NETCDF4", encoding=encoding)
