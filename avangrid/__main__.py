"""Command-line tools. Run from the project folder:

    python -m avangrid cache-build            # build (or refresh) the hourly and daily caches
    python -m avangrid cache-info             # list cached items and sizes
    python -m avangrid cache-clear            # delete the cache (rebuilt on next use)
    python -m avangrid run 70_segmentation.ipynb [more.ipynb ...]
    python -m avangrid benchmark              # time parallel vs sequential PU training
    python -m avangrid status-build           # create/refresh the customer_status table
"""

import argparse
import sys
import time

import avangrid  # noqa: F401  (sets the thread limit before scikit-learn loads)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m avangrid')
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('cache-build', help='build or refresh cached arrays')
    b.add_argument('--rebuild', action='store_true', help='rebuild even if the cache is current')
    sub.add_parser('cache-info', help='list cached items')
    sub.add_parser('cache-clear', help='delete all cached items')
    r = sub.add_parser('run', help='execute notebooks in place')
    r.add_argument('notebooks', nargs='+')
    r.add_argument('--timeout', type=int, default=3600, help='seconds allowed per cell')
    sub.add_parser('benchmark', help='time parallel vs sequential bagging PU')
    sub.add_parser('status-build', help='create or refresh the customer_status table')
    args = parser.parse_args(argv)

    if args.command == 'cache-build':
        from . import features, grids
        t = time.time()
        hd = grids.load_hourly(rebuild=args.rebuild)
        features.load_daily_signals(hd, rebuild=args.rebuild)
        print(f'Cache ready in {time.time() - t:.0f}s')
    elif args.command == 'cache-info':
        from . import cache
        import polars as pl
        with pl.Config(tbl_hide_dataframe_shape=True, fmt_str_lengths=80):
            print(cache.info())
    elif args.command == 'cache-clear':
        from . import cache
        cache.clear()
        print('Cache cleared.')
    elif args.command == 'run':
        from . import notebooks
        ok = all([notebooks.run(nb, timeout=args.timeout) for nb in args.notebooks])
        sys.exit(0 if ok else 1)
    elif args.command == 'benchmark':
        from .benchmark import run_benchmark
        run_benchmark()
    elif args.command == 'status-build':
        from .status import build_customer_status
        import polars as pl
        summary = build_customer_status()
        print('customer_status created in avangrid.db:')
        with pl.Config(tbl_hide_dataframe_shape=True):
            print(summary)


if __name__ == '__main__':
    main()
