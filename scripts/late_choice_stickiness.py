"""Late-session choice stickiness for ActionKernel synthetic sessions.

Stationary ActionKernel (option 1) has no explicit time-varying perseveration.
This module adds a **copy-last mixture** after AK simulate so each synthetic
draw's post-0.5 quintile mean_run matches **that real session's** mean_run
(journal ``sticky_end_of_session_exclusion.md`` §8), not a cohort median.

Geometric conversion (run length μ ≈ 1 / (1 − p_repeat)):

    p^real_q = 1 − 1/μ^real_q
    p^AK_q   = 1 − 1/μ^AK_q
    ρ_q = max(0, (p^real_q − p^AK_q) / (1 − p^AK_q))

Copy-last can only *raise* p_repeat; break-repeat (flip away from last
when AK already repeated) can only *lower* it. Quintiles are equal-count
slices of the drop-0.5 sequence (same masks as
``analyze_perseveration_counts._quintile_masks``). Time alignment is
quintile index, not raw calendar index (option-1 pseudos can be longer
than the real session).
"""
from __future__ import annotations

import numpy as np

N_QUINTILES = 5
MIN_TRIALS_PER_QUINTILE = 10
PLEFT_UNBIASED = 0.5
# One-shot copy-last recovers ~80% of the infinite-chain Δμ (no-op when the
# kernel already repeats). Gain restores a Markov sequence to the target μ.
LATE_STICKY_RHO_GAIN = 1.27


def _run_lengths(choices: np.ndarray) -> np.ndarray:
    choices = np.asarray(choices)
    if len(choices) == 0:
        return np.array([], dtype=int)
    change = np.flatnonzero(choices[1:] != choices[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(choices)]))
    return np.diff(bounds)


def mean_run(choices: np.ndarray) -> float:
    """Mean run length on valid ±1 choices (calendar order)."""
    ch = np.asarray(choices, dtype=float)
    ch = ch[np.isin(ch, (-1.0, 1.0))]
    if len(ch) == 0:
        return float('nan')
    rl = _run_lengths(ch)
    if len(rl) == 0:
        return float('nan')
    return float(np.mean(rl))


def p_repeat_from_mean_run(mu) -> float:
    mu = float(mu)
    if not np.isfinite(mu) or mu <= 1.0:
        return float('nan')
    return 1.0 - 1.0 / mu


def post05_quintile_index(
        pleft,
        n_groups: int = N_QUINTILES,
        min_per_q: int = MIN_TRIALS_PER_QUINTILE) -> np.ndarray:
    """0-based quintile on drop-0.5 trials; −1 on 0.5 trials or if too short.

    Same equal-count split as ``analyze_perseveration_counts._quintile_masks``.
    If the session has no 0.5 block, the full sequence is quintiled.
    """
    pleft = np.asarray(pleft, dtype=float).reshape(-1)
    n = len(pleft)
    q_idx = np.full(n, -1, dtype=int)
    nobias = ~np.isclose(pleft, PLEFT_UNBIASED)
    idx = np.flatnonzero(nobias)
    if len(idx) < n_groups * min_per_q:
        idx = np.arange(n)
    if len(idx) < n_groups * min_per_q:
        return q_idx
    edges = np.linspace(0, len(idx), n_groups + 1).astype(int)
    for q in range(n_groups):
        q_idx[idx[edges[q]:edges[q + 1]]] = q
    return q_idx


def quintile_mean_run(choice, pleft, n_groups: int = N_QUINTILES) -> np.ndarray:
    """Calendar mean_run inside each post-0.5 quintile (NaN if empty)."""
    choice = np.asarray(choice, dtype=float).reshape(-1)
    q_idx = post05_quintile_index(pleft, n_groups=n_groups)
    out = np.full(n_groups, np.nan)
    for q in range(n_groups):
        out[q] = mean_run(choice[q_idx == q])
    return out


def extra_repeat_probs(
        choice,
        pleft,
        target_mean_run) -> np.ndarray:
    """Per-quintile signed mix probabilities matching ``target_mean_run``.

    ``target_mean_run[q]`` is this *real* session's quintile mean_run.
    ``choice`` is the AK sequence on *its* post-0.5 quintiles.

    Positive ρ_q: copy last (raise p_repeat). Negative ρ_q: if AK already
    repeated, flip to −last (lower p_repeat). Zero if μ already matches
    or a quintile is empty.
    """
    target = np.asarray(target_mean_run, dtype=float).reshape(-1)
    n_g = int(target.size)
    if n_g < 1:
        raise ValueError('target_mean_run must be non-empty')
    mu_ak = quintile_mean_run(choice, pleft, n_groups=n_g)
    rho = np.zeros(n_g, dtype=float)
    for q in range(n_g):
        p_t = p_repeat_from_mean_run(target[q])
        p_a = p_repeat_from_mean_run(mu_ak[q])
        if not np.isfinite(p_t) or not np.isfinite(p_a):
            continue
        if p_t > p_a:
            denom = 1.0 - p_a
            if denom <= 1e-12:
                continue
            rho[q] = float(np.clip(
                LATE_STICKY_RHO_GAIN * (p_t - p_a) / denom, 0.0, 1.0))
        elif p_a > p_t:
            if p_a <= 1e-12:
                continue
            rho[q] = -float(np.clip((p_a - p_t) / p_a, 0.0, 1.0))
    return rho


def apply_late_stickiness(
        choice,
        pleft,
        rng,
        target_mean_run):
    """Signed copy-last / break-repeat so quintile mean_run matches ``target_mean_run``.

    ``choice`` / ``pleft`` are 1-d. Returns a new ±1/0 array. Timeouts (not
    ±1) are left alone and do not break the last-valid-choice chain.
    ``rng`` is a numpy Generator or RandomState. Positive ρ copies last;
    negative ρ flips a repeat to the opposite of last.
    """
    if target_mean_run is None:
        raise ValueError('target_mean_run is required (real-session quintile mean_run)')
    choice = np.asarray(choice, dtype=float).reshape(-1).copy()
    pleft = np.asarray(pleft, dtype=float).reshape(-1)
    if choice.shape != pleft.shape:
        raise ValueError(
            f'choice and pleft length mismatch: {choice.shape} vs {pleft.shape}')
    rho = extra_repeat_probs(choice, pleft, target_mean_run=target_mean_run)
    if not np.any(rho != 0):
        return choice
    n_g = len(rho)
    q_idx = post05_quintile_index(pleft, n_groups=n_g)
    rand = rng.random if hasattr(rng, 'random') else rng.rand
    last = None
    for t in range(len(choice)):
        ch = choice[t]
        if ch not in (-1.0, 1.0):
            continue
        q = int(q_idx[t])
        if last is not None and 0 <= q < n_g and rho[q] != 0:
            r = float(rho[q])
            u = float(rand())
            if r > 0 and u < r:
                choice[t] = last
                ch = last
            elif r < 0 and ch == last and u < -r:
                choice[t] = -last
                ch = -last
        last = ch
    return choice


def apply_late_stickiness_rows(choice, pleft, rng, target_mean_run):
    """Apply ``apply_late_stickiness`` to each row of a (n_sim, n_trials) array."""
    if target_mean_run is None:
        raise ValueError('target_mean_run is required (real-session quintile mean_run)')
    choice = np.asarray(choice, dtype=float)
    pleft = np.asarray(pleft, dtype=float)
    if choice.ndim == 1:
        return apply_late_stickiness(choice, pleft, rng, target_mean_run)
    out = np.array(choice, dtype=float, copy=True)
    if pleft.ndim == 1:
        for i in range(out.shape[0]):
            out[i] = apply_late_stickiness(out[i], pleft, rng, target_mean_run)
        return out
    if pleft.shape != out.shape:
        raise ValueError(
            f'choice and pleft shape mismatch: {out.shape} vs {pleft.shape}')
    for i in range(out.shape[0]):
        out[i] = apply_late_stickiness(out[i], pleft[i], rng, target_mean_run)
    return out
