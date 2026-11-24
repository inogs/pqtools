import math
from pathlib import Path
from typing import Any

import yaml
from bitsea.basins import Basin
from bitsea.basins.basin import SimplePolygonalBasin
from bitsea.basins.region import Polygon
from bitsea.basins.region import Rectangle

from pqtools.components.basins import basin_to_bounding_box

# Prefix of the basin specifications that refer to a basin of bitsea
BITSEA_PREFIX = "bitsea:"

_REQUIRED_FIELDS = frozenset({"name"})
_SHAPE_FIELDS = frozenset({"polygon", "rectangle"})
_OPTIONAL_FIELDS = frozenset({"extended_name"}) | _SHAPE_FIELDS
_VERTEX_FIELDS = frozenset({"longitude", "latitude"})
_RECTANGLE_FIELDS = frozenset(
    {
        "minimum_longitude",
        "maximum_longitude",
        "minimum_latitude",
        "maximum_latitude",
    }
)

MAX_LONGITUDE = 180.0
MAX_LATITUDE = 90.0

_MIN_VERTICES = 3


class InvalidBasinError(ValueError):
    """Raised when a basin cannot be loaded from its specification."""


def load_basin(specification: str) -> Basin:
    """Loads a basin from a string that describes it.

    Args:
        specification: Either ``bitsea:<uuid>``, where ``<uuid>`` is the uuid
            of a basin of bitsea (e.g. ``bitsea:adr1`` or
            ``bitsea:..V0.adn``), or the path of a YAML file that describes
            a basin (see `load_basin_from_yaml`).

    Returns:
        The basin.

    Raises:
        InvalidBasinError: If the basin does not exist, if its description
            is not valid, if its shape is not supported or if it does not
            overlap the domain of the data.
    """
    if specification.startswith(BITSEA_PREFIX):
        basin = load_bitsea_basin(specification.removeprefix(BITSEA_PREFIX))
    else:
        basin = load_basin_from_yaml(Path(specification))

    # Checked here so that the user gets an error before any download starts
    try:
        bounding_box = basin_to_bounding_box(basin)
    except (NotImplementedError, ValueError) as error:
        raise InvalidBasinError(
            f'the basin "{specification}" is not supported: {error}'
        ) from error
    domain_box = bounding_box.cut_on_mediterranean_sea()
    if (
        domain_box.minimum_latitude >= domain_box.maximum_latitude
        or domain_box.minimum_longitude >= domain_box.maximum_longitude
    ):
        raise InvalidBasinError(
            f'the basin "{specification}" does not overlap the domain of '
            f"the data (the Mediterranean Sea)"
        )
    return basin


def load_bitsea_basin(uuid: str) -> Basin:
    """Loads a basin of bitsea from its uuid.

    Args:
        uuid: The uuid of the basin, as returned by `Basin.get_uuid`: the
            name of a basin of the default module of bitsea (e.g. ``adr1``)
            or ``..<module>.<name>`` (e.g. ``..V0.adn``).

    Raises:
        InvalidBasinError: If there is no basin with this uuid.
    """
    # `Basin.load_from_uuid` imports the module written in the uuid: only the
    # modules of bitsea are accepted, to avoid that a string written by the
    # user could import (i.e. execute) an arbitrary module
    if "." in uuid and not uuid.startswith(".."):
        raise InvalidBasinError(
            f'"{uuid}" is not a valid uuid of a bitsea basin; use the name of '
            f"the basin (e.g. adr1) or ..<module>.<name> (e.g. ..V0.adn)"
        )
    try:
        return Basin.load_from_uuid(uuid)
    except (ImportError, KeyError) as error:
        raise InvalidBasinError(
            f'there is no basin with uuid "{uuid}" in bitsea'
        ) from error


def load_basin_from_yaml(path: Path) -> SimplePolygonalBasin:
    """Loads a basin from a YAML file.

    The file must contain a mapping with the following fields:

    * ``name``: the name of the basin (without dots and commas);
    * ``extended_name`` (optional): a longer description of the basin;
    * exactly one of the following shapes (all the coordinates are in
      degrees):

      * ``polygon``: the list of the vertices (at least 3), each one with
        the fields ``longitude`` and ``latitude``. The polygon is closed
        automatically;
      * ``rectangle``: a rectangle with the sides parallel to the axes of
        the grid, with the fields ``minimum_longitude``,
        ``maximum_longitude``, ``minimum_latitude`` and
        ``maximum_latitude``. The points on the border are inside. It is
        loaded as a `Rectangle` of bitsea, whose points are found
        more easily than the ones of a generic polygon.

    For example::

        name: my_polygon
        polygon:
          - {longitude: 12.0, latitude: 44.0}
          - {longitude: 13.5, latitude: 44.0}
          - {longitude: 13.0, latitude: 45.5}

    or::

        name: my_rectangle
        rectangle:
          minimum_longitude: 12.0
          maximum_longitude: 13.5
          minimum_latitude: 44.0
          maximum_latitude: 45.5

    Raises:
        InvalidBasinError: If the file cannot be read or if its content is
            not valid.
    """
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as error:
        raise InvalidBasinError(
            f'the basin file "{path}" cannot be read ({error}); to use a '
            f'basin of bitsea, add the prefix "{BITSEA_PREFIX}"'
        ) from error
    except yaml.YAMLError as error:
        raise InvalidBasinError(
            f'the basin file "{path}" is not a valid YAML file: {error}'
        ) from error

    try:
        return _basin_from_content(content)
    except ValueError as error:
        raise InvalidBasinError(
            f'the basin file "{path}" is not valid: {error}'
        ) from error


def _basin_from_content(content: Any) -> SimplePolygonalBasin:
    _check_fields(
        content, _REQUIRED_FIELDS, _OPTIONAL_FIELDS, description="the file"
    )

    for field in ("name", "extended_name"):
        value = content.get(field)
        if field in content and (not isinstance(value, str) or not value):
            raise ValueError(
                f"{field} must be a non-empty string; received {value!r}"
            )

    shapes = _SHAPE_FIELDS & content.keys()
    if len(shapes) != 1:
        raise ValueError(
            f"exactly one of the fields {sorted(_SHAPE_FIELDS)} is required; "
            f"received {sorted(shapes)}"
        )
    if "rectangle" in shapes:
        region = _read_rectangle(content["rectangle"])
    else:
        region = _read_polygon(content["polygon"])

    return SimplePolygonalBasin(
        name=content["name"],
        region=region,
        extended_name=content.get("extended_name"),
    )


def _read_polygon(vertices: Any) -> Polygon:
    if not isinstance(vertices, list):
        raise InvalidBasinError("polygon must be a list of vertices")

    points = []
    for i, vertex in enumerate(vertices, start=1):
        description = f"vertex {i}"
        _check_fields(vertex, _VERTEX_FIELDS, description=description)
        points.append(
            (
                _coordinate(vertex, "longitude", MAX_LONGITUDE, description),
                _coordinate(vertex, "latitude", MAX_LATITUDE, description),
            )
        )
    if len(set(points)) < _MIN_VERTICES:
        raise ValueError(
            f"at least {_MIN_VERTICES} different vertices are required; "
            f"received {len(set(points))}"
        )
    return Polygon(
        lon_list=[p[0] for p in points], lat_list=[p[1] for p in points]
    )


def _read_rectangle(limits: Any) -> Rectangle:
    description = "rectangle"
    _check_fields(limits, _RECTANGLE_FIELDS, description=description)

    ranges = {}
    for axis, limit in (
        ("longitude", MAX_LONGITUDE),
        ("latitude", MAX_LATITUDE),
    ):
        minimum = _coordinate(limits, f"minimum_{axis}", limit, description)
        maximum = _coordinate(limits, f"maximum_{axis}", limit, description)
        if minimum >= maximum:
            raise ValueError(
                f"minimum_{axis} ({minimum}) must be smaller than "
                f"maximum_{axis} ({maximum})"
            )
        ranges[axis] = (minimum, maximum)

    return Rectangle(
        lonmin=ranges["longitude"][0],
        lonmax=ranges["longitude"][1],
        latmin=ranges["latitude"][0],
        latmax=ranges["latitude"][1],
    )


def _check_fields(
    content: Any,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    *,
    description: str,
) -> None:
    if not isinstance(content, dict):
        raise InvalidBasinError(
            f"{description} must be a mapping with the fields "
            f"{sorted(required)}"
        )
    missing = required - content.keys()
    if missing:
        raise ValueError(f"{description} misses the fields {sorted(missing)}")
    unknown = content.keys() - required - optional
    if unknown:
        raise ValueError(
            f"{description} has unknown fields: {sorted(map(str, unknown))}"
        )


def _coordinate(
    content: dict[str, Any], field: str, limit: float, description: str
) -> float:
    value = content[field]
    # In Python, a bool is also an int. The comparison is False for NaN
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.fabs(value) <= limit
    ):
        raise ValueError(
            f"{field} of {description} must be a number between "
            f"{-limit} and {limit}; received {value!r}"
        )
    return float(value)
