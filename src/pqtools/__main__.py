import logging
import time

from pqtools.components.cli import COMPUTE_COMPARISON_MAPS
from pqtools.components.cli import COMPUTE_TIMESERIES
from pqtools.components.cli import parse_args


def configure_logger(logger: logging.Logger | None = None, level=logging.INFO):
    if logger is None:
        logger = logging.getLogger()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # Ensure that this formatter uses local time and not UTC
    formatter.converter = time.localtime

    logger.setLevel(level)

    # Remove all handlers associated with the copernicusmarine logger
    copernicusmarine_logger = logging.getLogger("copernicusmarine")
    copernicusmarine_logger.handlers.clear()

    logging.getLogger("urllib3").setLevel(logging.ERROR)

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)

    logger.addHandler(handler)


def main() -> int:
    configure_logger()
    logger = logging.getLogger()

    args = parse_args()

    from pqtools.components.basin_data import SatelliteDatasetOptions
    from pqtools.components.output import write_netcdf

    satellite_dataset_options = SatelliteDatasetOptions(
        half_window_size=args.satellite_average_window,
        max_qi=args.satellite_max_qi,
    )

    if args.action == COMPUTE_TIMESERIES:
        from pqtools.components.compute_timeseries import compute_timeseries

        timeseries = compute_timeseries(
            start_date=args.start_date,
            end_date=args.end_date,
            basin=args.basin,
            satellite_dataset_options=satellite_dataset_options,
        )

        write_netcdf(timeseries, args.output_file)

    elif args.action == COMPUTE_COMPARISON_MAPS:
        from pqtools.components.compute_comparison_maps import (
            compute_comparison_maps,
        )

        comparison_maps = compute_comparison_maps(
            evaluated_dataset=args.evaluated_dataset,
            reference_dataset=args.reference_dataset,
            start_date=args.start_date,
            end_date=args.end_date,
            basin=args.basin,
            satellite_dataset_options=satellite_dataset_options,
        )

        write_netcdf(comparison_maps, args.output_file)

    else:
        raise ValueError(f"Unexpected action: {args.action}")

    logger.info("Execution complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
