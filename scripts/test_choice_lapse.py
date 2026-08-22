#!/usr/bin/env python
"""Smoke for post-decision choice_lapse (ε=0 identity, ε=1 ⊥ stim, M aligned)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulate_recovery import apply_choice_lapse, attach_choice_lapse, choice_lapse_dir_tag


def _fake_results(choices, sides, steps=2):
    M_rows = []
    choice_sides = []
    trial_sides = []
    asig = []
    for c, s in zip(choices, sides):
        # Left (−1): M0 > M1; right (+1): M0 < M1; timeout: equal.
        if c == -1:
            row = [0.8, 0.2]
            act = 0.5
        elif c == 1:
            row = [0.2, 0.8]
            act = -0.5
        else:
            row = [0.5, 0.5]
            act = 0.0
        M_rows.append(np.tile(row, (steps, 1)))
        choice_sides.append(np.full(steps, float(c)))
        trial_sides.append(np.full(steps, float(s)))
        asig.append(np.full(steps, act))
    return {
        "choices": list(choices),
        "correct_action_taken": [int(c == s) for c, s in zip(choices, sides)],
        "trial_sides": trial_sides,
        "choice_sides": choice_sides,
        "M": np.concatenate(M_rows, axis=0),
        "action_signal": np.concatenate(asig),
    }


def _choice_matches_m(results):
    choices = results["choices"]
    lens = [len(ts) for ts in results["trial_sides"]]
    M = np.asarray(results["M"], float)
    off = 0
    for ch, m_i in zip(choices, lens):
        sl = slice(off, off + m_i)
        d = float(np.mean(M[sl, 0] - M[sl, 1]))
        if ch == -1:
            assert d > 0, (ch, d)
        elif ch == 1:
            assert d < 0, (ch, d)
        off += m_i


def test_eps0_identity():
    rng = np.random.RandomState(0)
    choices = [1, -1, 1, 0, -1]
    sides = [1, -1, -1, 1, 1]
    results = _fake_results(choices, sides)
    m0 = results["M"].copy()
    out = apply_choice_lapse(results, rng, 0.0)
    assert out["choices"] == choices
    assert out["correct_action_taken"] == [1, 1, 0, 0, 0]
    assert out["choice_sides"][0][0] == 1
    assert np.array_equal(out["M"], m0)


def test_eps1_independent():
    rng = np.random.RandomState(1)
    n = 4000
    sides = [1 if i % 2 == 0 else -1 for i in range(n)]
    choices = list(sides)
    results = _fake_results(choices, sides)
    apply_choice_lapse(results, rng, 1.0)
    new = np.asarray(results["choices"])
    tside = np.asarray(sides)
    acc = float(np.mean(new == tside))
    assert abs(acc - 0.5) < 0.03, acc
    assert abs(float(np.mean(new == 1)) - 0.5) < 0.03
    correct = np.asarray(results["correct_action_taken"])
    assert np.all(correct == (new == tside).astype(int))
    _choice_matches_m(results)
    for i, ch in enumerate(results["choices"]):
        if ch in (-1, 1):
            assert results["choice_sides"][i][0] == ch


def test_timeouts_untouched():
    rng = np.random.RandomState(2)
    results = _fake_results([0, 0], [1, -1])
    m0 = results["M"].copy()
    apply_choice_lapse(results, rng, 1.0)
    assert results["choices"] == [0, 0]
    assert np.array_equal(results["M"], m0)


def test_cache_tag():
    mp = {"g_s": 0.0}
    attach_choice_lapse(mp, 0.0)
    assert "choice_lapse" not in mp
    assert "choice_lapse_align_m" not in mp
    assert choice_lapse_dir_tag(0.0) == ""
    attach_choice_lapse(mp, 0.1)
    assert mp["choice_lapse"] == 0.1
    assert mp["choice_lapse_align_m"] is True
    assert choice_lapse_dir_tag(0.1) == "_lapse0.1"


if __name__ == "__main__":
    test_eps0_identity()
    test_eps1_independent()
    test_timeouts_untouched()
    test_cache_tag()
    print("choice_lapse smoke OK")
