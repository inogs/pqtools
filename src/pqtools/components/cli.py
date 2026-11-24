import argparse
import math
from datetime import datetime
from pathlib import Path

from bitsea.basins import Basin

from pqtools.components.basin_data import SatelliteDatasetOptions
from pqtools.components.basin_loader import InvalidBasinError
from pqtools.components.basin_loader import load_basin
from pqtools.components.data_retrievers import DataSource
from pqtools.components.regridding import DEFAULT_MIN_SATELLITE_COVERAGE

COMPUTE_TIMESERIES = "compute_timeseries"
COMPUTE_COMPARISON_MAPS = "compute_comparison_maps"

_DEFAULT_SATELLITE_OPTIONS = SatelliteDatasetOptions()


def date_argument(value: str) -> datetime:
    """Parses a command line argument as a date.

    Accepts the ISO 8601 formats supported by `datetime.fromisoformat`,
    e.g. ``2025-03-01`` or ``2025-03-01T12:00:00``.

    Raises:
        argparse.ArgumentTypeError: If `value` is not a valid date.
    """
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'"{value}" is not a valid date; use the format YYYY-MM-DD '
            f"(e.g. 2025-03-01)"
        )


def output_file_argument(value: str) -> Path:
    """Parses a command line argument as the path of an output file.

    The file itself does not need to exist, but the directory that should
    contain it must.

    Raises:
        argparse.ArgumentTypeError: If the parent directory of `value` does
            not exist or if `value` is an existing directory.
    """
    path = Path(value)
    if path.is_dir():
        raise argparse.ArgumentTypeError(
            f'"{value}" is a directory, not a file'
        )
    if not path.parent.is_dir():
        raise argparse.ArgumentTypeError(
            f'the directory "{path.parent}" does not exist'
        )
    return path


def non_negative_int_argument(value: str) -> int:
    """Parses a command line argument as an integer greater or equal than 0.

    Raises:
        argparse.ArgumentTypeError: If `value` is not an integer or if it is
            negative.
    """
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'"{value}" is not a valid integer number'
        )
    if number < 0:
        raise argparse.ArgumentTypeError(
            f"the value must be greater or equal than 0; received {number}"
        )
    return number


def non_negative_float_argument(value: str) -> float:
    """Parses a command line argument as a finite float greater or equal than 0.

    Raises:
        argparse.ArgumentTypeError: If `value` is not a number, if it is not
            finite (nan or inf) or if it is negative.
    """
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f'"{value}" is not a valid number')
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError(
            f"the value must be a finite number; received {value}"
        )
    if number < 0:
        raise argparse.ArgumentTypeError(
            f"the value must be greater or equal than 0; received {number}"
        )
    return number


def fraction_argument(value: str) -> float:
    """Parses a command line argument as a number between 0 and 1.

    Raises:
        argparse.ArgumentTypeError: If `value` is not a number or if it is
            not between 0 and 1.
    """
    number = non_negative_float_argument(value)
    if number > 1:
        raise argparse.ArgumentTypeError(
            f"the value must be between 0 and 1; received {number}"
        )
    return number


def basin_argument(value: str) -> Basin:
    """Parses a command line argument as a basin (see `load_basin`).

    Raises:
        argparse.ArgumentTypeError: If the basin cannot be loaded.
    """
    try:
        return load_basin(value)
    except InvalidBasinError as error:
        raise argparse.ArgumentTypeError(str(error))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pqtools",
        description="Compare chlorophyll values from the Copernicus Marine "
        "analysis, reanalysis and satellite products.",
    )
    subparsers = parser.add_subparsers(
        dest="action",
        required=True,
        metavar="ACTION",
        help="the action to perform",
    )

    # Options shared by all the actions
    basin_options = argparse.ArgumentParser(add_help=False)
    basin_options.add_argument(
        "--basin",
        type=basin_argument,
        required=True,
        metavar="BASIN",
        help="the basin on which the statistics are computed: either "
        "bitsea:UUID, where UUID is the uuid of a basin of bitsea (e.g. "
        "bitsea:adr1), or the path of a YAML file that describes the basin",
    )

    # Options shared by all the actions that use the satellite data
    satellite_options = argparse.ArgumentParser(add_help=False)
    satellite_group = satellite_options.add_argument_group(
        "satellite dataset options"
    )
    satellite_group.add_argument(
        "--satellite-average-window",
        type=non_negative_int_argument,
        default=_DEFAULT_SATELLITE_OPTIONS.half_window_size,
        required=False,
        metavar="DAYS",
        help="half-width, in days, of the centered moving average applied "
        "to the satellite data to fill the cells left empty by clouds "
        "or missing passes. The value of each day is the average of "
        "the DAYS days before, the day itself and the DAYS days after "
        "(i.e. over 2*DAYS+1 days; e.g. 3 means a 7-day window). "
        "Use 0 to disable the average and keep the daily values "
        "(default: %(default)s)",
    )
    satellite_group.add_argument(
        "--satellite-max-qi",
        type=non_negative_float_argument,
        default=_DEFAULT_SATELLITE_OPTIONS.max_qi,
        required=False,
        metavar="QI",
        help="threshold on the quality index (QI_CHL) of the satellite "
        "data, i.e. the anomaly with respect to the climatology in "
        "units of climatological standard deviations. The satellite "
        "values with |QI_CHL| strictly greater than QI, and the ones "
        "without a quality index, are removed before any other "
        "processing. Use 0 to disable the filter "
        "(default: %(default)s)",
    )

    # Options for the satellite data when they are interpolated on the model
    # grid. The group has the same title as the previous one: argparse merges
    # them in the actions that use both the parsers
    regridding_options = argparse.ArgumentParser(add_help=False)
    regridding_options.add_argument_group(
        "satellite dataset options"
    ).add_argument(
        "--satellite-min-coverage",
        type=fraction_argument,
        default=DEFAULT_MIN_SATELLITE_COVERAGE,
        required=False,
        metavar="FRACTION",
        help="minimum fraction (between 0 and 1) of the sea area of a model "
        "cell that must be covered by valid satellite data; the cells with "
        "a lower coverage are ignored. Use 0 to keep all the cells with at "
        "least one valid value. It has no effect if the reference dataset "
        "is not the satellite (default: %(default)s)",
    )

    timeseries = subparsers.add_parser(
        COMPUTE_TIMESERIES,
        parents=[basin_options, satellite_options],
        help="compute the chlorophyll timeseries",
        description="Compute the chlorophyll timeseries.",
    )
    timeseries.add_argument(
        "--start-date",
        type=date_argument,
        required=True,
        help="first day of the period, in the format YYYY-MM-DD (inclusive)",
    )
    timeseries.add_argument(
        "--end-date",
        type=date_argument,
        required=True,
        help="last day of the period, in the format YYYY-MM-DD (inclusive)",
    )

    timeseries.add_argument(
        "-o",
        "--output-file",
        type=output_file_argument,
        required=True,
        help="path of the output file; its directory must already exist",
    )

    comparison_maps = subparsers.add_parser(
        COMPUTE_COMPARISON_MAPS,
        parents=[basin_options, satellite_options, regridding_options],
        help="compute the comparison maps",
        description="Compute the comparison maps.",
    )
    comparison_maps.add_argument(
        "--evaluated-dataset",
        type=DataSource,
        choices=[DataSource.ANALYSIS, DataSource.REANALYSIS],
        required=True,
        metavar="{analysis,reanalysis}",
        help="the model dataset to evaluate",
    )
    comparison_maps.add_argument(
        "--reference-dataset",
        type=DataSource,
        choices=list(DataSource),
        default=DataSource.SATELLITE,
        required=False,
        metavar="{" + ",".join(s.value for s in DataSource) + "}",
        help="the dataset used as reference; it must be different from "
        "--evaluated-dataset (default: %(default)s)",
    )
    comparison_maps.add_argument(
        "--start-date",
        type=date_argument,
        required=True,
        help="first day of the period, in the format YYYY-MM-DD (inclusive)",
    )
    comparison_maps.add_argument(
        "--end-date",
        type=date_argument,
        required=True,
        help="last day of the period, in the format YYYY-MM-DD (inclusive)",
    )
    comparison_maps.add_argument(
        "-o",
        "--output-file",
        type=output_file_argument,
        required=True,
        help="path of the output file; its directory must already exist",
    )

    return parser


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Reads and validates the command line arguments.

    Args:
        args: The arguments to parse; if None, `sys.argv[1:]` is used.

    Returns:
        The parsed arguments. `action` contains the name of the selected
        action.
    """
    parser = build_parser()
    parsed = parser.parse_args(args)

    if parsed.start_date > parsed.end_date:
        parser.error(
            f"--start-date ({parsed.start_date.date()}) must not be later "
            f"than --end-date ({parsed.end_date.date()})"
        )

    if (
        parsed.action == COMPUTE_COMPARISON_MAPS
        and parsed.evaluated_dataset is parsed.reference_dataset
    ):
        parser.error(
            "--reference-dataset must be different from --evaluated-dataset"
        )

    return parsed
