import pytest

from pqtools.components.cli import parse_args
from pqtools.components.regridding import DEFAULT_MIN_SATELLITE_COVERAGE

COMMON = [
    "--basin=bitsea:adr1",
    "--start-date=2025-03-01",
    "--end-date=2025-03-02",
    "-o=out.nc",
]
MAPS = ["compute_comparison_maps", "--evaluated-dataset=analysis", *COMMON]


def test_default_min_coverage():
    args = parse_args(MAPS)
    assert args.satellite_min_coverage == DEFAULT_MIN_SATELLITE_COVERAGE


@pytest.mark.parametrize("value", ["0", "0.5", "1"])
def test_min_coverage(value):
    args = parse_args([*MAPS, f"--satellite-min-coverage={value}"])
    assert args.satellite_min_coverage == float(value)


@pytest.mark.parametrize("value", ["-0.1", "1.1", "75", "nan", "x"])
def test_invalid_min_coverage(value):
    with pytest.raises(SystemExit):
        parse_args([*MAPS, f"--satellite-min-coverage={value}"])


def test_timeseries_has_no_min_coverage():
    with pytest.raises(SystemExit):
        parse_args(
            ["compute_timeseries", *COMMON, "--satellite-min-coverage=0.5"]
        )
