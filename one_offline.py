"""Keep Slurm jobs off the Alyx database.

Every ``ONE`` used by analysis code should come from ``make_one``. Inside a
Slurm job (``SLURM_JOB_ID`` set) it opens ONE in ``mode='local'``, so trials,
spikes and cache tables are read from the ONE cache on disk. It also blocks
HTTP requests to ``*.internationalbrainlab.org``, so a call that would reach
Alyx raises ``AlyxBlockedError`` instead of quietly querying the server.

Incident 2026-11-02: in ONE's default remote mode, every ``load_trials_and_mask``
sent two Alyx requests (``/sessions?id=``, ``/sessions/<eid>``). The standard-splits
act prior and the sticky θ table reloaded the full session for every insertion
in every shard (~120k requests over two days, throttling other users).

Overrides, for a deliberate one-off job only:
  ONE_MODE=remote          open ONE online (the guard still applies in Slurm)
  ALLOW_ALYX_IN_SLURM=1    lift the guard as well
On the laptop (no ``SLURM_JOB_ID``) the default stays ``remote``, unless
``ONE_MODE=local`` or ``BLOCK_ALYX=1`` is set.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

_TRUTHY = ('1', 'true', 'yes', 'on')
BLOCKED_DOMAIN = 'internationalbrainlab.org'


class AlyxBlockedError(BaseException):
    '''An HTTP request to Alyx was attempted where Alyx is off limits.

    A ``BaseException``, so the analysis loops' ``except Exception`` cannot
    turn it into a skipped insertion: the job stops.
    '''


def _env_true(name):
    return os.environ.get(name, '').strip().lower() in _TRUTHY


def in_slurm_job():
    return bool(os.environ.get('SLURM_JOB_ID'))


def alyx_blocked():
    '''True when HTTP to Alyx must raise (Slurm jobs, or ``BLOCK_ALYX=1``).'''
    if _env_true('ALLOW_ALYX_IN_SLURM'):
        return False
    return in_slurm_job() or _env_true('BLOCK_ALYX')


def one_mode():
    mode = os.environ.get('ONE_MODE', '').strip().lower()
    if mode:
        return mode
    return 'local' if alyx_blocked() else 'remote'


_guard_installed = False


def install_alyx_guard():
    '''Make every ``requests`` call to Alyx raise. Idempotent.'''
    global _guard_installed
    if _guard_installed:
        return
    import requests.sessions

    real_request = requests.sessions.Session.request

    def guarded_request(self, method, url, *args, **kwargs):
        host = urlparse(str(url)).hostname or ''
        if host == BLOCKED_DOMAIN or host.endswith('.' + BLOCKED_DOMAIN):
            raise AlyxBlockedError(
                f'Alyx request blocked ({method} {url}). Cluster jobs read the '
                'ONE cache on disk only; see one_offline.py.')
        return real_request(self, method, url, *args, **kwargs)

    requests.sessions.Session.request = guarded_request
    _guard_installed = True


def is_missing_local_data(exc, one):
    '''True when an offline ONE could not find a file or dataset on disk.

    Online, ONE would have downloaded it. Offline, the loops must not treat
    this as a bad insertion and skip it, or results change silently.
    '''
    from one.alf.exceptions import ALFError
    if not getattr(one, 'offline', False):
        return False
    if isinstance(exc, ALFError):
        return True
    # ONE/brainbox calls that only work online, e.g. SpikeSortingLoader
    # without eid/pname, or pid2eid without an insertions table.
    msg = str(exc).lower()
    return ('local mode' in msg or 'remote connection' in msg
            or 'offline' in msg)


def make_one(**kwargs):
    '''``ONE(**kwargs)``, opened offline and guarded inside Slurm jobs.'''
    from one.api import ONE

    if alyx_blocked():
        install_alyx_guard()
    mode = kwargs.pop('mode', None) or one_mode()
    if mode == 'local':
        # A username/password makes AlyxClient log in on construction.
        kwargs.pop('username', None)
        kwargs.pop('password', None)
    return ONE(mode=mode, **kwargs)


if alyx_blocked():
    install_alyx_guard()
