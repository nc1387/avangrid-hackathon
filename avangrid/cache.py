"""Disk cache for derived arrays, plus a place to save experiment results.

Each cached item is a folder in ``config.CACHE_DIR`` holding one ``.npy`` file per array, any
polars tables as ``.parquet``, and ``meta.json``. The metadata records a *key*: a fingerprint
of the source tables plus a version string for the code that built it. On load the key is
compared with the current one, and the item is rebuilt if anything changed. The cache folder
is safe to delete at any time.

Arrays load with ``mmap_mode='r'``: they open instantly, are paged in from disk as they're
used, and are read-only. Use ``np.array(arr)`` if you need a writable copy.
"""

import json
import shutil
import time
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from . import config


def _folder(name: str) -> Path:
    return config.CACHE_DIR / name


def _read_meta(name: str) -> dict | None:
    p = _folder(name) / 'meta.json'
    return json.loads(p.read_text()) if p.exists() else None


def save(name: str, key: dict, arrays: dict[str, np.ndarray] | None = None,
         tables: dict[str, pl.DataFrame] | None = None, info: dict | None = None) -> None:
    folder = _folder(name)
    tmp = folder.with_name(folder.name + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for k, v in (arrays or {}).items():
        np.save(tmp / f'{k}.npy', v)
    for k, v in (tables or {}).items():
        v.write_parquet(tmp / f'{k}.parquet')
    meta = {'key': key, 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'arrays': {k: [list(v.shape), str(v.dtype)] for k, v in (arrays or {}).items()},
            'tables': sorted(tables or {}), 'info': info or {}}
    (tmp / 'meta.json').write_text(json.dumps(meta, indent=2))
    shutil.rmtree(folder, ignore_errors=True)
    tmp.rename(folder)  # swap in only once fully written


def load(name: str, mmap: bool = True) -> tuple[dict[str, np.ndarray], dict[str, pl.DataFrame], dict]:
    folder, meta = _folder(name), _read_meta(name)
    arrays = {k: np.load(folder / f'{k}.npy', mmap_mode='r' if mmap else None) for k in meta['arrays']}
    tables = {k: pl.read_parquet(folder / f'{k}.parquet') for k in meta['tables']}
    return arrays, tables, meta.get('info', {})


def get_or_build(name: str, key: dict, build: Callable[[], tuple], rebuild: bool = False, verbose: bool = True):
    """Return cached (arrays, tables, info) for ``name``, building and saving it if needed.

    ``build`` returns ``(arrays, tables, info)``.
    ``key=None`` means the source couldn't be checked (e.g. the database is locked by another
    process): an existing cache is used as-is, with a warning.
    """
    meta = _read_meta(name)
    if key is None:
        if meta is None:
            raise RuntimeError(f'{name} is not cached yet and the source data could not be read '
                               '(is another process writing to avangrid.db?). Try again once it has finished.')
        print(f'[cache] WARNING: could not check whether {name} is up to date (database busy); '
              f'using the cached copy built {meta["created"]}')
        return load(name)
    if not rebuild and meta is not None and meta['key'] == key:
        if verbose:
            print(f'[cache] loaded {name} (built {meta["created"]})')
        return load(name)
    t = time.time()
    reason = 'forced rebuild' if rebuild else ('not cached yet' if meta is None else 'source data or code changed')
    if verbose:
        print(f'[cache] building {name} ({reason}) ...')
    arrays, tables, info = build()
    save(name, key, arrays, tables, info)
    if verbose:
        print(f'[cache] built {name} in {time.time() - t:.0f}s')
    return load(name)


def info() -> pl.DataFrame:
    """List cached items with their size on disk."""
    rows = []
    if config.CACHE_DIR.exists():
        for folder in sorted(p for p in config.CACHE_DIR.iterdir() if p.is_dir() and not p.name.endswith('.tmp')):
            meta = _read_meta(folder.name) or {}
            size = sum(f.stat().st_size for f in folder.iterdir())
            rows.append({'name': folder.name, 'created': meta.get('created'), 'size_mb': round(size / 1e6, 1),
                         'arrays': ', '.join(meta.get('arrays', {}))})
    return pl.DataFrame(rows, schema={'name': pl.String, 'created': pl.String, 'size_mb': pl.Float64, 'arrays': pl.String})


def clear(name: str | None = None) -> None:
    """Delete one cached item, or the whole cache if ``name`` is None."""
    shutil.rmtree(_folder(name) if name else config.CACHE_DIR, ignore_errors=True)


# --- results of long-running experiments -------------------------------------------------

def save_results(name: str, df: pl.DataFrame) -> Path:
    """Save an experiment's results so a notebook can load them instead of recomputing."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f'{name}.parquet'
    df.write_parquet(path)
    return path


def load_results(name: str) -> pl.DataFrame:
    return pl.read_parquet(config.RESULTS_DIR / f'{name}.parquet')
