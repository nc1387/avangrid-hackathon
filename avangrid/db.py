"""DuckDB helpers.

DuckDB 2.0 notes (these tripped up earlier notebooks):
- Read results with ``con.sql(query).pl()``. Relations no longer have ``fetchone()``.
- A result from ``con.execute(...)`` must be fully read before the same connection runs
  another query, so prefer ``con.sql(...)`` for anything you read back.
- Only one process can open the database for writing at a time. Use ``read_only=True``
  (the default here) unless you are creating tables.
"""

from contextlib import contextmanager

import duckdb
import polars as pl

from . import config


@contextmanager
def connect(read_only: bool = True):
    """Open ``avangrid.db`` with the progress bar off (it clutters notebook output)."""
    con = duckdb.connect(str(config.DB_PATH), read_only=read_only)
    try:
        con.execute('SET enable_progress_bar = false')
        yield con
    finally:
        con.close()


def sql(query: str, read_only: bool = True) -> pl.DataFrame:
    """Run one query and return the result as a polars DataFrame."""
    with connect(read_only=read_only) as con:
        return con.sql(query).pl()


def table_exists(name: str) -> bool:
    return sql(f"SELECT COUNT(*) AS n FROM duckdb_tables() WHERE table_name = '{name}'")['n'].item() > 0 or \
        sql(f"SELECT COUNT(*) AS n FROM duckdb_views() WHERE view_name = '{name}'")['n'].item() > 0


def fingerprint(checks: dict[str, str]) -> dict[str, list]:
    """Cheap summary of the tables a cached result depends on.

    ``checks`` maps a label to a query returning one row (e.g. a count and a sum). If any of
    these change, caches built from the tables are rebuilt.
    """
    with connect() as con:
        return {name: [None if v is None else float(v) for v in con.sql(q).fetchall()[0]]
                for name, q in checks.items()}
