from bitsea.basins import Basin

from pqtools.components.cli import COMPUTE_COMPARISON_MAPS
from pqtools.components.cli import COMPUTE_TIMESERIES
from pqtools.components.cli import parse_args


BASIN = Basin.load_from_uuid("adr1")

def main() -> int:
    args = parse_args()

    if args.action == COMPUTE_TIMESERIES:
        from pqtools.components.compute_timeseries import TimeseriesOptions
        from pqtools.components.compute_timeseries import compute_timeseries

        options = TimeseriesOptions(
            half_satellite_window_size=args.satellite_average_window,
            satellite_max_qi=args.satellite_max_qi,
        )

        timeseries = compute_timeseries(
            start_date=args.start_date,
            end_date=args.end_date,
            basin=BASIN,
            options=options,
        )

        timeseries.to_netcdf(args.output_file)

    elif args.action == COMPUTE_COMPARISON_MAPS:
        print("to be implemented")
    else:
        raise ValueError(f"Unexpected action: {args.action}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
