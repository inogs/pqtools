from datetime import datetime, timezone

import copernicusmarine
import numpy as np
import xarray as xr
from bitsea.basins import Basin
from bitsea.commons.grid import RegularGrid

from pqtools.components.basins import BoundingBox

ANALYSIS_DATASET = "cmems_mod_med_bgc_anfc_4.2km_static"
REANALYSIS_DATASET = "cmems_mod_med_bgc_my_4.2km_static"

# Gap-free (cloud-interpolated) version of the satellite product: it is
# defined on the same 1 km grid as the L3 data and its NaNs mark land only
SATELLITE_L4_DATASET = "cmems_obs-oc_med_bgc-plankton_my_l4-gapfree-multi-1km_P1D"
SATELLITE_L4_VARIABLE = "CHL"

# Day of the L4 dataset used to build the satellite mask. The land/sea
# pattern of the L4 product does not change in time, so any day works; a
# fixed one keeps the mask reproducible.
SATELLITE_MASK_DATE = datetime(2024, 1, 15, tzinfo=timezone.utc)


class MaskMisalignmentError(ValueError):
    """Raised when two masks are not defined on the same lat/lon grid."""


def _check_alignment(mask1: xr.DataArray, mask2: xr.DataArray) -> None:
    for coord in ("latitude", "longitude"):
        values1 = mask1[coord].values
        values2 = mask2[coord].values
        if not np.array_equal(values1, values2):
            raise MaskMisalignmentError(
                f'The masks are not aligned along "{coord}": '
                f"mask 1 has {values1.size} points in "
                f"[{values1.min()}, {values1.max()}], "
                f"mask 2 has {values2.size} points in "
                f"[{values2.min()}, {values2.max()}]"
            )


def get_model_mask(bounding_box: BoundingBox) -> xr.DataArray:
    ds_masks = []
    for ds_name in (ANALYSIS_DATASET, REANALYSIS_DATASET):
        mask_data = copernicusmarine.open_dataset(
            dataset_id = ds_name,
            variables = ["mask",],
            **bounding_box.cut_on_mediterranean_sea().to_dict()
        )
        ds_masks.append(mask_data["mask"].isel(depth=0, drop=True))

    analysis_mask, reanalysis_mask = ds_masks
    _check_alignment(analysis_mask, reanalysis_mask)

    # True only where both masks are 1 (sea), False otherwise.
    # copernicusmarine opens the static datasets without dask, so the
    # comparison above already loads the masks as numpy arrays; the result
    # is converted to dask and persisted, so that (as for the satellite
    # mask) it is a dask array whose chunks are already computed: with
    # dask.distributed they stay on the workers instead of being serialized
    # inside the graph at every compute
    mask = ((analysis_mask == 1) & (reanalysis_mask == 1)).chunk().persist()
    mask.name = "mask"
    mask.attrs = {
        "long_name": "Sea in both analysis and reanalysis: True = sea ; False = land",
    }
    return mask


def get_satellite_mask(bounding_box: BoundingBox) -> xr.DataArray:
    """Computes the land/sea mask of the satellite chlorophyll product.

    The satellite L3 product has no static mask, and its NaNs cannot be used
    to build one, because they mark both land and clouds. The mask is
    therefore built from the gap-free L4 version of the same product
    (`SATELLITE_L4_DATASET`), where the gaps due to clouds are filled by
    interpolation: there, a point is NaN if and only if it is land. The L4
    product is defined on exactly the same grid as the L3 one, so the mask
    can be applied directly to the data returned by
    `pqtools.components.data_retrievers.download_data`.

    Only one day (`SATELLITE_MASK_DATE`) of the L4 dataset is read, since
    its land/sea pattern is constant in time.

    Note:
        The L4 mask has been checked against the daily L3 data of the whole
        Mediterranean Sea over 2020-2024: 99.89% of its sea points have
        valid L3 data on at least one day. The remaining 0.11% (about 2200
        points) are coastal points where the L3 product never has valid
        data: mainly coastal lagoons (e.g. the Nile delta lagoons, the
        Bardawil lagoon and the Venice lagoon) and narrow channels between
        islands (e.g. the Dalmatian coast, the Aegean islands and the
        Strait of Bonifacio). Conversely, 45 isolated points (mostly small
        islands) have valid L3 data at least once but are land in the mask.

    Args:
        bounding_box: The area where the mask is computed. It is clipped to
            the Mediterranean Sea, as for the model mask.

    Returns:
        A boolean `xarray.DataArray` named ``"mask"`` with dimensions
        ``(latitude, longitude)``, True on sea points and False on land.
        The array is backed by dask and already persisted: its values are
        downloaded when this function is called and kept in memory (or on
        the workers, when using dask.distributed), so using the mask in
        later computations does not trigger new downloads.
    """
    l4_data = copernicusmarine.open_dataset(
        dataset_id=SATELLITE_L4_DATASET,
        variables=[SATELLITE_L4_VARIABLE],
        start_datetime=SATELLITE_MASK_DATE,
        end_datetime=SATELLITE_MASK_DATE,
        **bounding_box.cut_on_mediterranean_sea().to_dict()
    )
    chl = l4_data[SATELLITE_L4_VARIABLE].isel(time=0, drop=True)

    # For small requests (like a single day) copernicusmarine opens the data
    # without dask; as in `get_model_mask`, the mask is converted to dask
    # and persisted, so that it is always a dask array with its chunks
    # already computed, whatever the size of the request
    mask = chl.notnull().chunk().persist()
    mask.name = "mask"
    mask.attrs = {
        "long_name": "Satellite land-sea mask (from the L4 gap-free product): "
                     "True = sea ; False = land",
    }
    return mask


def cut_mask_on_basin(mask: xr.DataArray, basin: Basin) -> xr.DataArray:
    """Restricts a land/sea mask to the points that are inside a basin.

    The masks returned by `get_model_mask` and `get_satellite_mask` cover a
    rectangular region (usually the bounding box of the basin), while a basin
    is in general a more complex shape. This function intersects the mask
    with the basin: the longitudes and latitudes of the mask are combined
    into two 2D grids, which are passed to `basin.is_inside`, and a point of
    the new mask is True only if it is True in the original mask (i.e. it is
    a sea point) and it is inside the basin.

    The new mask is defined on the same grid as the original one: the points
    outside the basin are not removed but set to False, so that the mask can
    still be applied to the data downloaded on the same region.

    Args:
        mask: A boolean mask with the dimensions ``latitude`` and
            ``longitude`` (in any order), True on the points to be kept, as
            the ones returned by `get_model_mask` and `get_satellite_mask`.
        basin: The basin used to restrict the mask. It can be any basin of
            bitsea (also a composed one), since only its ``is_inside``
            method is used.

    Returns:
        A boolean `xarray.DataArray` with the same name, dimensions and
        coordinates as `mask`, which is True where `mask` is True and the
        point is inside `basin`, False elsewhere. Its attributes are the ones
        of `mask`, plus a ``basin`` attribute with the uuid of `basin`. As
        for the input masks, the result is a dask array that has already
        been computed (persisted), so it can be used many times without
        computing the intersection again.

    Raises:
        ValueError: If `mask` does not have the ``latitude`` and
            ``longitude`` dimensions.
    """
    missing_dims = {"latitude", "longitude"} - set(mask.dims)
    if missing_dims:
        raise ValueError(
            f"The mask must have the dimensions latitude and longitude; "
            f"missing: {sorted(missing_dims)}"
        )

    # 2D grids of shape (latitude, longitude)
    longitude_2d, latitude_2d = np.meshgrid(
        mask["longitude"].values, mask["latitude"].values
    )
    is_inside = xr.DataArray(
        np.asarray(basin.is_inside(lon=longitude_2d, lat=latitude_2d), dtype=bool),
        dims=("latitude", "longitude"),
        coords={
            "latitude": mask["latitude"],
            "longitude": mask["longitude"],
        },
    )

    # The masks are combined by coordinate name, so the order of the
    # dimensions of `mask` does not matter; the result keeps the order of
    # the dimensions of `mask`
    basin_mask = (mask & is_inside).transpose(*mask.dims).persist()
    basin_mask.name = mask.name
    basin_mask.attrs = {**mask.attrs, "basin": basin.get_uuid()}
    return basin_mask


def apply_mask(data: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """Sets to NaN the values of `data` that are outside `mask`.

    The mask is broadcast along the other dimensions of `data` (e.g. the
    time), so the same mask is applied to every day.

    Before applying the mask, this function checks that `data` and `mask`
    are defined on exactly the same latitude and longitude coordinates.
    This is needed because xarray aligns the two arrays on their
    coordinates: if the grids were different (even by a rounding error),
    the points not in common would be silently dropped from the result.

    Args:
        data: The data to mask, with (at least) the dimensions ``latitude``
            and ``longitude``.
        mask: A boolean mask with the dimensions ``latitude`` and
            ``longitude``, True on the points to be kept (e.g. one of the
            masks returned by `cut_mask_on_basin`).

    Returns:
        An `xarray.DataArray` with the same name, dimensions, coordinates
        and attributes as `data`, where the values outside `mask` are NaN.
        The computation is lazy if `data` or `mask` are backed by dask.

    Raises:
        MaskMisalignmentError: If `data` and `mask` are not defined on the
            same latitude and longitude coordinates.
    """
    _check_alignment(data, mask)
    return data.where(mask)


def compute_cell_area(mask: xr.DataArray) -> xr.DataArray:
    """Computes the area of the cells of a mask, in square meters.

    The latitudes and longitudes of the mask are interpreted as the centers
    of the cells of a regular grid (`bitsea.commons.grid.RegularGrid`). The
    area of each cell is computed by bitsea as ``e1t * e2t``, i.e. the
    product of the lengths of the segments of the parallel and of the
    meridian passing through the center of the cell that lie within the
    cell. Therefore, the area decreases with the latitude, as the length of
    the parallels.

    The area is returned only for the cells inside the mask: the cells where
    the mask is False are set to NaN. In this way, the result can be used
    directly as the weights of a spatial average on the points of the mask.

    Args:
        mask: A boolean mask with the dimensions ``latitude`` and
            ``longitude`` (in any order), True on the points to be kept (e.g.
            one of the masks returned by `cut_mask_on_basin`). Its latitudes
            and longitudes must be strictly increasing, as required by
            `RegularGrid`.

    Returns:
        An `xarray.DataArray` named ``"cell_area"``, with the same dimensions
        and coordinates as `mask`, containing the area (in m^2) of each cell
        where `mask` is True and NaN elsewhere. As for the masks, it is a
        dask array that has already been computed (persisted).

    Raises:
        ValueError: If the latitudes or the longitudes of `mask` are not
            strictly increasing (raised by `RegularGrid`).
    """
    grid = RegularGrid(lat=mask["latitude"].values, lon=mask["longitude"].values)
    cell_area = xr.DataArray(
        grid.area,
        dims=("latitude", "longitude"),
        coords={
            "latitude": mask["latitude"],
            "longitude": mask["longitude"],
        },
    )

    # The cells outside the mask are set to NaN; the result keeps the order
    # of the dimensions of the mask
    cell_area = cell_area.where(mask).transpose(*mask.dims).persist()
    cell_area.name = "cell_area"
    cell_area.attrs = {
        "long_name": "Area of the cells inside the mask (NaN outside)",
        "units": "m2",
    }
    return cell_area
