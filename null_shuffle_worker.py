"""Minimal worker for parallel label-shuffle null distance computation."""

import numpy as np


def contrast_matched_shuffle_labels(contrasts, n0, rng):
    """
    Pseudo high/low labels with the same contrast histogram as the real groups.

    Within each contrast bin, randomly assign the same number of trials to
    pseudo-high as in the real high-prior group (first n0 trials in b order).
    """
    contrasts = np.asarray(contrasts)
    ntr = contrasts.shape[0]
    ys_true = np.zeros(ntr, dtype=bool)
    ys_true[:n0] = True

    pseudo = np.zeros(ntr, dtype=bool)
    for c in np.unique(contrasts):
        idx = np.where(contrasts == c)[0]
        n_high = int(ys_true[idx].sum())
        if n_high <= 0:
            continue
        if n_high >= len(idx):
            pseudo[idx] = True
            continue
        chosen = rng.choice(idx, size=n_high, replace=False)
        pseudo[chosen] = True
    return pseudo


def null_shuffle_chunk(b, n0, ys_true, n_chunk, seed):
    """Compute n_chunk unrestricted label-shuffle null condition-mean pairs."""
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


def null_shuffle_contrast_chunk(b, n0, contrasts, n_chunk, seed):
    """Compute n_chunk contrast-matched label-shuffle null condition-mean pairs."""
    rng = np.random.RandomState(seed)
    b = np.asarray(b, dtype=float)
    contrasts = np.asarray(contrasts)
    out = []
    for _ in range(n_chunk):
        pseudo = contrast_matched_shuffle_labels(contrasts, n0, rng)
        out.append(b[pseudo].mean(axis=0))
        out.append(b[~pseudo].mean(axis=0))
    return out
