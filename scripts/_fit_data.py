"""
Canonical fit targets for weights / joint / retinal drivers (repo-local).

Files live in ``<repo>/fit_targets/`` (copied from paper-brain-wide-map /
``model_test.ipynb`` — nested ``mean_data_results``, paper prior curves,
``avg_mean_R``). Drivers must load from this directory — not ONE
``manifold/res`` (historically flat / incomplete) or a sibling paper checkout.

``ensure_fit_data_links`` refreshes cwd symlinks so ``loss_prior_effect`` and
similar cwd-relative loaders keep working. Fail closed if ``I``/``M``
``mean_traj`` is not nested ``{stim, choice}`` (2026-08-12c).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# scripts/_fit_data.py → repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
FIT_TARGETS_DIR = REPO_ROOT / "fit_targets"

FIT_MEAN_NAME = "mean_data_results.npy"
FIT_PRIOR_NAMES = (
    "data_act_block_duringstim.npy",
    "data_act_block_duringchoice.npy",
)
FIT_AVG_MEAN_R = "avg_mean_R.npy"
FIT_TARGET_NAMES = (FIT_MEAN_NAME,) + FIT_PRIOR_NAMES + (FIT_AVG_MEAN_R,)


def resolve_fit_targets_dir(explicit=None):
    """Return the repo ``fit_targets`` directory (must contain mean_data)."""
    if explicit is not None:
        d = Path(explicit).resolve()
    else:
        d = FIT_TARGETS_DIR
    if not (d / FIT_MEAN_NAME).is_file():
        raise FileNotFoundError(
            f"Missing {FIT_MEAN_NAME} under {d}. "
            "Expected repo fit_targets/ (notebook nested targets)."
        )
    return d


# Back-compat aliases used by plot_best_fit_results / older call sites.
def resolve_paper_data_dir():
    """Deprecated name — returns repo fit_targets dir."""
    try:
        return resolve_fit_targets_dir()
    except FileNotFoundError:
        return None


PAPER_DIR_CANDIDATES = (FIT_TARGETS_DIR,)


def _force_symlink(dst: Path, src: Path):
    src = src.resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    if dst.is_symlink() or dst.exists():
        try:
            if dst.resolve() == src:
                return False
        except FileNotFoundError:
            pass
        dst.unlink()
    dst.symlink_to(src)
    return True


def validate_mean_data_results(mean_data_results, source="mean_data_results.npy"):
    """Fail closed if traj targets lack nested stim/choice windows."""
    for vn in ("I", "M"):
        if vn not in mean_data_results:
            raise ValueError(f"{source}: missing key {vn!r}")
        mt = mean_data_results[vn].get("mean_traj")
        if not (isinstance(mt, dict) and "stim" in mt and "choice" in mt):
            keys = list(mt.keys())[:12] if isinstance(mt, dict) else type(mt)
            raise ValueError(
                f"{source}: {vn}.mean_traj must be nested {{'stim','choice'}} "
                f"(repo fit_targets / model_test.ipynb format); got keys={keys}. "
                "Flat manifold/res copies score only I-post (M baseline needs stim) "
                "and understate L_w vs the notebook diagnostics (2026-08-12c)."
            )


def ensure_fit_data_links(
    pth_res=None,
    paper_dir=None,
    fit_targets_dir=None,
    require_avg_mean_r=False,
    mean_and_prior=True,
):
    """
    Symlink fit targets from ``<repo>/fit_targets`` into cwd.

    ``pth_res`` / ``paper_dir`` are ignored for source selection (kept for call
    compatibility). Sources are always repo ``fit_targets``.
    """
    del pth_res  # no longer used as a data source
    targets = resolve_fit_targets_dir(fit_targets_dir or paper_dir)
    cwd = Path.cwd()
    refreshed = []

    names = []
    if mean_and_prior:
        names.append(FIT_MEAN_NAME)
        names.extend(FIT_PRIOR_NAMES)
    if require_avg_mean_r:
        names.append(FIT_AVG_MEAN_R)

    for name in names:
        src = targets / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing {src}")
        if _force_symlink(cwd / name, src):
            refreshed.append(name)

    if refreshed:
        print(f"[fit-data] refreshed cwd links from {targets}: {', '.join(refreshed)}")

    return {
        "fit_targets_dir": str(targets),
        "paper_dir": str(targets),  # back-compat key
        "mean_data": str((cwd / FIT_MEAN_NAME).resolve())
        if mean_and_prior and (cwd / FIT_MEAN_NAME).exists() else None,
        "mean_origin": "repo",
        "prior_origin": "repo" if mean_and_prior else None,
        "avg_mean_R": str((cwd / FIT_AVG_MEAN_R).resolve())
        if require_avg_mean_r else None,
        "refreshed": refreshed,
    }


def load_validated_mean_data(path=None):
    """Load mean_data_results (cwd link, explicit path, or fit_targets) + validate."""
    if path is not None:
        mean_path = Path(path)
    else:
        cwd_mean = Path.cwd() / FIT_MEAN_NAME
        mean_path = cwd_mean if cwd_mean.exists() else (
            resolve_fit_targets_dir() / FIT_MEAN_NAME
        )
    mean_path = mean_path.resolve()
    mean_data = np.load(mean_path, allow_pickle=True).flat[0]
    validate_mean_data_results(mean_data, source=str(mean_path))
    return mean_path, mean_data


def load_avg_mean_r(path=None):
    """Load avg_mean_R from explicit path, cwd, or fit_targets."""
    if path is not None:
        p = Path(path)
    else:
        cwd_p = Path.cwd() / FIT_AVG_MEAN_R
        p = cwd_p if cwd_p.exists() else (resolve_fit_targets_dir() / FIT_AVG_MEAN_R)
    p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    return p, np.load(p, allow_pickle=True).flat[0]
