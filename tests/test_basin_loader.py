import numpy as np
import pytest
from bitsea.basins import Basin
from bitsea.basins.region import Rectangle

from pqtools.components.basin_loader import InvalidBasinError
from pqtools.components.basin_loader import load_basin
from pqtools.components.basins import basin_to_bounding_box

VERTICES = """
  - {longitude: 12, latitude: 44.0}
  - {longitude: 13.5, latitude: 44.0}
  - {longitude: 13.0, latitude: 45.5}
"""

RECTANGLE = """
  minimum_longitude: 12
  maximum_longitude: 13.5
  minimum_latitude: 44.0
  maximum_latitude: 45.5
"""

INVALID_CONTENTS = {
    "empty": "",
    "not yaml": "a: [",
    "not mapping": "[1, 2]",
    "no name": f"polygon:{VERTICES}",
    "null name": f"name:\npolygon:{VERTICES}",
    "dot in name": f"name: a.b\npolygon:{VERTICES}",
    "unknown field": f"name: a\nfoo: 1\npolygon:{VERTICES}",
    "no shape": "name: a",
    "two shapes": f"name: a\npolygon:{VERTICES}rectangle:{RECTANGLE}",
    "rectangle not mapping": "name: a\nrectangle: [1, 2, 3, 4]",
    "missing side": "name: a\nrectangle: {minimum_longitude: 12}",
    "swapped sides": "name: a\nrectangle:" + RECTANGLE.replace("13.5", "11"),
    "no vertices": "name: a\npolygon: []",
    "vertices not list": "name: a\npolygon: 3",
    "vertex not mapping": "name: a\npolygon: [[1, 2]]",
    "nan": f"name: a\npolygon:{VERTICES}  - {{longitude: .nan, latitude: 1}}",
    "string": f"name: a\npolygon:{VERTICES}  - {{longitude: x, latitude: 1}}",
    "bool": f"name: a\npolygon:{VERTICES}  - {{longitude: true, latitude: 1}}",
    "latitude": f"name: a\npolygon:{VERTICES}"
    "  - {longitude: 1, latitude: 91}",
    "outside domain": "name: a\npolygon:\n"
    "  - {longitude: 50, latitude: 10}\n"
    "  - {longitude: 51, latitude: 10}\n"
    "  - {longitude: 50, latitude: 11}\n",
}


def test_load_bitsea_basin():
    assert load_basin("bitsea:adr1") is Basin.load_from_uuid("adr1")


@pytest.mark.parametrize("uuid", ["..V0.adn", "med"])
def test_bitsea_uuid(uuid):
    assert load_basin(f"bitsea:{uuid}").get_uuid() == uuid


@pytest.mark.parametrize(
    "uuid", ["", "nope", "..nope.adr1", "..V2.nope", "os.path", ".."]
)
def test_invalid_bitsea_uuid(uuid):
    with pytest.raises(InvalidBasinError):
        load_basin(f"bitsea:{uuid}")


def test_load_yaml_basin(tmp_path):
    path = tmp_path / "basin.yaml"
    path.write_text(f"name: abc\nextended_name: A basin\npolygon:{VERTICES}")
    basin = load_basin(str(path))
    assert basin.name == "abc"
    assert basin.extended_name == "A basin"
    assert basin.is_inside(lon=13.0, lat=44.5)
    assert not basin.is_inside(lon=12.1, lat=45.4)


def test_load_yaml_rectangle(tmp_path):
    path = tmp_path / "basin.yaml"
    path.write_text(f"name: abc\nrectangle:{RECTANGLE}")
    basin = load_basin(str(path))
    assert isinstance(basin.region, Rectangle)

    # The points on the border are inside
    lon = np.array([11.99, 12.0, 13.0, 13.5, 13.51])
    inside = [False, True, True, True, False]
    for lat in (44.0, 45.0, 45.5):
        assert basin.is_inside(lon=lon, lat=lat).tolist() == inside
    for lat in (43.99, 45.51):
        assert not basin.is_inside(lon=lon, lat=lat).any()

    box = basin_to_bounding_box(basin)
    assert (box.minimum_longitude, box.maximum_longitude) == (12.0, 13.5)
    assert (box.minimum_latitude, box.maximum_latitude) == (44.0, 45.5)


@pytest.mark.parametrize(
    "content", INVALID_CONTENTS.values(), ids=INVALID_CONTENTS.keys()
)
def test_invalid_yaml_basin(tmp_path, content):
    path = tmp_path / "basin.yaml"
    path.write_text(content)
    with pytest.raises(InvalidBasinError):
        load_basin(str(path))


def test_missing_file_suggests_prefix():
    with pytest.raises(InvalidBasinError, match="bitsea:"):
        load_basin("adr1")
