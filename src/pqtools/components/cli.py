import argparse
import math
from datetime import datetime
from pathlib import Path

COMPUTE_TIMESERIES = "compute_timeseries"
COMPUTE_COMPARISON_MAPS = "compute_comparison_maps"


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
        raise argparse.ArgumentTypeError(
            f'"{value}" is not a valid number'
        )
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError(
            f"the value must be a finite number; received {value}"
        )
    if number < 0:
        raise argparse.ArgumentTypeError(
            f"the value must be greater or equal than 0; received {number}"
        )
    return number


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

    timeseries = subparsers.add_parser(
        COMPUTE_TIMESERIES,
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
        "--satellite-average-window",
        type=non_negative_int_argument,
        default=3,
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

    timeseries.add_argument(
        "--satellite-max-qi",
        type=non_negative_float_argument,
        default=0.,
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

    timeseries.add_argument(
        "-o", "--output-file",
        type=output_file_argument,
        required=True,
        help="path of the output file; its directory must already exist",
    )

    subparsers.add_parser(
        COMPUTE_COMPARISON_MAPS,
        help="compute the comparison maps",
        description="Compute the comparison maps.",
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

    if parsed.action == COMPUTE_TIMESERIES and parsed.start_date > parsed.end_date:
        parser.error(
            f"--start-date ({parsed.start_date.date()}) must not be later "
            f"than --end-date ({parsed.end_date.date()})"
        )

    return parsed
