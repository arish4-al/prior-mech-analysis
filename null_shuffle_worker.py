"""Minimal worker for parallel label-shuffle null distance computation."""

import numpy as np


def null_shuffle_chunk(b, n0, ys_true, n_chunk, seed):
    """Compute n_chunk label-shuffle null condition-mean pairs."""
    rng = np.random.RandomState(seed)
    b = np.asarray(b, dtype=float)
    ys_true = np.asarray(ys_true, dtype=bool)
    out = []
    for _ in range(n_chunk):
        perm = ys_true.copy()
        rng.shuffle(perm)
        out.append(b[perm].mean(axis=0))
        out.append(b[~perm].mean(axis=0))
    return out
