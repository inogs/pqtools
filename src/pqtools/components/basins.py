import logging
from collections import deque
from dataclasses import dataclass

import numpy as np
from bitsea.basins import Basin

LOGGER = logging.getLogger(__name__)


MED_MINIMUM_LATITUDE = 30.1875
MED_MAXIMUM_LATITUDE = 45.97916793823242
MED_MINIMUM_LONGITUDE = -5.5
MED_MAXIMUM_LONGITUDE = 36.29166793823242


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """
    Represents a geographical bounding box.

    This class embodies a rectangular geographical region defined by latitude
    and longitude boundaries.
    The objects of this class are used to define the slices used by the
    providers to select the data from the remote datasets.

    Attributes:
        minimum_latitude: The southernmost latitude of the bounding box.
        maximum_latitude: The northernmost latitude of the bounding box.
        minimum_longitude: The westernmost longitude of the bounding box.
        maximum_longitude: The easternmost longitude of the bounding box.
    """

    minimum_latitude: float
    maximum_latitude: float
    minimum_longitude: float
    maximum_longitude: float

    def cut_on_mediterranean_sea(self):
        return BoundingBox(
            minimum_latitude=max(self.minimum_latitude, MED_MINIMUM_LATITUDE),
            maximum_latitude=min(self.maximum_latitude, MED_MAXIMUM_LATITUDE),
            minimum_longitude=max(
                self.minimum_longitude, MED_MINIMUM_LONGITUDE
            ),
            maximum_longitude=min(
                self.maximum_longitude, MED_MAXIMUM_LONGITUDE
            ),
        )

    def to_dict(self):
        return {
            "minimum_latitude": self.minimum_latitude,
            "maximum_latitude": self.maximum_latitude,
            "minimum_longitude": self.minimum_longitude,
            "maximum_longitude": self.maximum_longitude,
        }

    @classmethod
    def mediterranean_sea(cls):
        return cls(
            minimum_longitude=MED_MINIMUM_LONGITUDE,
            maximum_longitude=MED_MAXIMUM_LONGITUDE,
            minimum_latitude=MED_MINIMUM_LATITUDE,
            maximum_latitude=MED_MAXIMUM_LATITUDE,
        )


def basin_to_bounding_box(basin: Basin) -> BoundingBox:
    if basin.get_uuid() == "med":
        return BoundingBox(
            minimum_longitude=MED_MINIMUM_LONGITUDE,
            maximum_longitude=MED_MAXIMUM_LONGITUDE,
            minimum_latitude=MED_MINIMUM_LATITUDE,
            maximum_latitude=MED_MAXIMUM_LATITUDE,
        )

    to_be_checked = deque([basin])
    basin_uuid = None
    simple_basins = {}

    # If the basin is composed, add its components to the list of simple basins
    # that we have to check
    while len(to_be_checked) > 0:
        current_basin = to_be_checked.pop()
        basin_uuid = current_basin.get_uuid()
        LOGGER.debug("Checking if basin %s is a composed basin", basin_uuid)
        if basin_uuid in simple_basins:
            LOGGER.debug(
                "Basin %s has already been visited! Skipped", basin_uuid
            )
            continue
        if hasattr(current_basin, "basin_list"):
            LOGGER.debug(
                "Basin %s is composed! Adding its components to the list of "
                "basins that we have to check",
                basin_uuid,
            )
            for basin_component in reversed(current_basin.basin_list):
                LOGGER.debug(
                    "Adding basin %s to the list of basins that we have to check",
                    basin_component.get_uuid(),
                )
                to_be_checked.append(basin_component)
        else:
            LOGGER.debug(
                "Basin %s is not composed! Adding it to the list of simple basins",
                basin_uuid,
            )
            simple_basins[basin_uuid] = current_basin

    if len(simple_basins) == 0:
        raise ValueError(
            f"The basin {basin_uuid} is a composed basin but it does not "
            "contain any other basin"
        )

    min_lat = None
    max_lat = None
    min_lon = None
    max_lon = None
    for current_basin in simple_basins.values():
        if not hasattr(current_basin, "borders"):
            raise NotImplementedError(
                f"This script only supports SimplePolygonalBasins, but basin "
                f"{current_basin.name} (that is part of your composed basin "
                f"{basin.name}) is a {type(current_basin)}"
            )
        longitudes = [p[0] for p in current_basin.borders]
        latitudes = [p[1] for p in current_basin.borders]

        current_min_lon = np.min(longitudes)
        current_max_lon = np.max(longitudes)
        current_min_lat = np.min(latitudes)
        current_max_lat = np.max(latitudes)

        if min_lon is None or min_lon > current_min_lon:
            min_lon = current_min_lon
        if max_lon is None or max_lon < current_max_lon:
            max_lon = current_max_lon
        if min_lat is None or min_lat > current_min_lat:
            min_lat = current_min_lat
        if max_lat is None or max_lat < current_max_lat:
            max_lat = current_max_lat

    if (
        max_lat is None
        or min_lat is None
        or max_lon is None
        or min_lon is None
    ):
        raise ValueError("No points found on the boundary of the basin")

    max_lat = min(max_lat, MED_MAXIMUM_LATITUDE)
    min_lat = max(min_lat, MED_MINIMUM_LATITUDE)

    return BoundingBox(
        minimum_longitude=min_lon,
        maximum_longitude=max_lon,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
    )
