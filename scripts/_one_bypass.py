"""Import this BEFORE `model_functions` to skip ONE in the model-fitting path.

`model_functions` only needs ONE to resolve the local cache dir. Constructing ONE
in every loky worker re-reads/rewrites ``~/.one`` params and can race
(BrokenProcessPool / JSONDecodeError). This module resolves the cache dir once and
sets ``PRIOR_MECH_NO_ONE=1`` so `model_functions` (in this process and in workers,
which inherit the env) takes ``ONE_CACHE_DIR`` directly instead of building ONE.

Resolution order:
  1. If ``ONE_CACHE_DIR`` is already set (e.g. SLURM worker / ORCD), trust it — no
     ONE, no network. This is what makes it safe on internet-less compute nodes.
  2. Otherwise construct ONE ONCE in this (single, non-racing) process to read
     ``one.cache_dir``, export it, and continue.
  3. If ONE can't be constructed, do nothing and let `model_functions` build ONE
     the normal way (fallback; unchanged behavior).
"""
from __future__ import annotations

import os

_TRUTHY = ("1", "true", "yes", "on")


def ensure_one_bypass():
    already = (
        os.environ.get("PRIOR_MECH_NO_ONE", "").strip().lower() in _TRUTHY
        and os.environ.get("ONE_CACHE_DIR")
    )
    if already:
        return

    if not os.environ.get("ONE_CACHE_DIR"):
        try:
            from one.api import ONE

            o = ONE(
                base_url="https://openalyx.internationalbrainlab.org",
                password="international",
                silent=True,
            )
            os.environ["ONE_CACHE_DIR"] = str(o.cache_dir)
            del o
        except Exception:
            # Leave ONE to be constructed normally by model_functions.
            return

    if os.environ.get("ONE_CACHE_DIR"):
        os.environ.setdefault("PRIOR_MECH_NO_ONE", "1")


ensure_one_bypass()
