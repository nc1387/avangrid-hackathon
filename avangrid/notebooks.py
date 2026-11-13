"""Run notebooks from the command line (or a background job) without touching their kernel setting.

The notebooks are saved with the editor's kernel (``conda-base-py``), which command-line Jupyter
doesn't know about. This runs them with the standard ``python3`` kernel, from the project folder,
then puts the original kernel setting back. If a cell fails, the original notebook is left
untouched and a copy with the partial outputs is saved in ``results/failed_runs/``.
"""

import copy
import re
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from . import config


def run(path: str | Path, timeout: int = 3600, kernel_name: str = 'python3') -> bool:
    path = Path(path)
    if not path.is_absolute():
        path = config.PROJECT_ROOT / path
    nb = nbformat.read(path, as_version=4)
    original_meta = copy.deepcopy(nb.metadata)
    client = NotebookClient(nb, timeout=timeout, kernel_name=kernel_name,
                            resources={'metadata': {'path': str(config.PROJECT_ROOT)}})
    t = time.time()
    print(f'Running {path.name} ...', flush=True)
    try:
        client.execute()
    except CellExecutionError as err:
        failed_dir = config.RESULTS_DIR / 'failed_runs'
        failed_dir.mkdir(parents=True, exist_ok=True)
        nb.metadata = original_meta
        out = failed_dir / path.name
        nbformat.write(nb, out)
        message = re.sub(r'\x1b\[[0-9;]*m', '', str(err)).strip()  # drop terminal colour codes
        first_line = message.splitlines()[-1] if message else repr(err)
        print(f'FAILED after {time.time() - t:.0f}s: {first_line}')
        print(f'Original notebook unchanged; partial run saved to {out}')
        return False
    nb.metadata = original_meta
    nbformat.write(nb, path)
    print(f'Done in {time.time() - t:.0f}s; outputs saved to {path.name}')
    return True
