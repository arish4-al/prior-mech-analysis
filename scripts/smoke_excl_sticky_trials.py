#!/usr/bin/env python
"""
Smoke-test late+perseveration trial exclusion (no neural data required for
the mask self-test; optional get_d_vars if insertion_cache exists).

  python scripts/smoke_excl_sticky_trials.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _self_test_mask():
    import block_analysis_allsplits as ba

    n = 100
    choice = np.ones(n)
    # Break runs so mid segment is not glued to the perseveration epoch.
    choice[40:42] = -1.0
    stim_left = np.zeros(n, dtype=bool)
    stim_left[:20] = True
    stim_left[20:40] = False  # right stim while choosing left → pers
    stim_left[40:42] = False
    stim_left[42:] = True
    contrast = np.full(n, 0.25)
    cl = np.where(stim_left, contrast, np.nan)
    cr = np.where(stim_left, np.nan, contrast)

    trials = {
        'choice': choice,
        'contrastLeft': cl,
        'contrastRight': cr,
    }
    drop, info = ba.sticky_trial_exclusion_mask(trials, late_frac=0.2, min_run=10)
    assert info.get('pers_mode') == 'tail', info
    assert info['n_late'] == 20, info
    # Tail-of-run: keep first 9 of the 40-trial poorly-explained run, drop 9..39
    assert not drop[:9].any(), drop[:9]
    assert drop[9:40].all(), drop[9:40]
    assert not drop[42:80].any(), drop[42:80]
    assert drop[80:].all(), drop[80:]
    assert info['n_perseveration'] == 31, info  # 40 - 9
    print('  mask self-test OK (tail-of-run)', info)


def main():
    # Import after path setup; ONE init may be slow.
    print('Self-test sticky exclusion mask …', flush=True)
    _self_test_mask()

    cache = Path(os.environ.get(
        'ONE_CACHE_DIR',
        Path.home() / 'Downloads/ONE/alyx.internationalbrainlab.org',
    ))
    caches = sorted((cache / 'manifold' / 'insertion_cache').glob('*.npy'))
    if not caches:
        print('No insertion_cache — mask self-test only.', flush=True)
        print('SMOKE PASSED (self-test)', flush=True)
        return

    from one.api import ONE
    import block_analysis_allsplits as ba

    ba.one = ONE(cache_dir=str(cache), mode='local')
    ba.configure_excl_sticky_output_dirs(str(cache))
    nrand = int(os.environ.get('SMOKE_NRAND', '10'))
    split = 'choice_stim_l_block_l'
    for fpath in caches:
        c = np.load(fpath, allow_pickle=True).item()
        pid = c.get('pid')
        if not pid:
            continue
        try:
            D = ba.get_d_vars(
                split, pid, control=True, nrand=nrand, cached=c,
                exclude_sticky_trials=True)
        except ba.InsufficientTrials as exc:
            print(f'  skip {fpath.name}: {exc}', flush=True)
            continue
        if not isinstance(D, dict) or 'trial_exclusion' not in D:
            raise SystemExit(f'Missing trial_exclusion in return: {type(D)}')
        print(f'OK {split} null_scheme={D.get("null_scheme")} '
              f'excl={D["trial_exclusion"]}', flush=True)
        print('SMOKE PASSED', flush=True)
        return
    raise SystemExit('SMOKE FAILED: no insertion completed')


if __name__ == '__main__':
    main()
