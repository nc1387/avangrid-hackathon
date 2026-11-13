"""Shared code for the Avangrid AMI analysis notebooks.

Import this package *before* scikit-learn (or anything that imports it), so the thread
limit below takes effect:

    import avangrid
    from avangrid import config, db, grids, features, pu

Why the thread limit: scikit-learn's gradient boosting starts one OpenMP thread per core for
every model it fits. With the many small models used for PU learning, those threads fight
each other and a run that takes about a minute single-threaded can take more than ten.
Each model therefore runs on one thread, and speed comes from fitting several models at
once in separate processes instead (see ``avangrid.pu``).
"""

import os

# Must be set before scikit-learn / NumPy's OpenMP runtime starts. setdefault leaves any value
# the user has chosen alone. Worker processes inherit it.
os.environ.setdefault('OMP_NUM_THREADS', '1')

__version__ = '0.1.0'
