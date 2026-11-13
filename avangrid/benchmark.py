"""Check that parallel bagging PU gives the same scores as sequential, and how much faster it is."""

import time

import numpy as np

from . import features, labels, pu


def run_benchmark(n_models: int = 30, seed: int = 0):
    X, _ = features.customer_features()
    from . import db
    customer_numbers = db.sql('SELECT CUSTOMER_NUMBER FROM customer_features ORDER BY 1')['CUSTOMER_NUMBER'].to_numpy()
    is_pv = labels.label_arrays(customer_numbers)['PV']
    rng = np.random.default_rng(100)
    labeled = np.zeros(len(is_pv), bool)
    labeled[rng.choice(np.where(is_pv)[0], round(0.25 * is_pv.sum()), replace=False)] = True
    print(f'PV hidden-label setup: {len(is_pv):,} customers, {labeled.sum()} labeled, {n_models} models')

    t = time.time()
    seq, _ = pu.bagging_pu(X, labeled, seed, n_models=n_models, n_jobs=1)
    t_seq = time.time() - t
    print(f'  sequential (1 process):         {t_seq:6.1f}s')

    t = time.time()
    par, _ = pu.bagging_pu(X, labeled, seed, n_models=n_models)
    t_par = time.time() - t
    print(f'  parallel ({pu.DEFAULT_JOBS} processes):         {t_par:6.1f}s  (includes starting the workers)')

    t = time.time()
    par2, _ = pu.bagging_pu(X, labeled, seed, n_models=n_models)
    t_par2 = time.time() - t
    print(f'  parallel again (workers ready): {t_par2:6.1f}s')

    same = np.allclose(seq, par, equal_nan=True) and np.allclose(par, par2, equal_nan=True)
    print(f'Speed-up: {t_seq / t_par2:.1f}x. Identical scores: {same}')
    return {'sequential_s': t_seq, 'parallel_s': t_par, 'parallel_warm_s': t_par2, 'identical': same}
