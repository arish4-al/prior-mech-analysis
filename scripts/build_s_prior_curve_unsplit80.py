#!/usr/bin/env python
"""Build the S prior-distance curve for the joint ``full`` variant.

Cell-weighted ``(obs − mean(nulls)) / n_splits`` over the **intersection** of

1. unsplit 80 ms act-prior label-shuffle FDR@0.01, and
2. SC sensory regions (``sc_duringstim_regtype`` 0 = stim or 0.1 = stim_early).

Same as ``load_group(..., correction='simple')``: the combined ``regde`` is a
**sum** of split Euclidean curves, so divide by ``n_splits`` (2 for
``act_block_duringstim_{l,r}``) before the cell-weighted average. Then the
``t ≤ 80`` ms prefix.

Writes next to the earlystim table that defined the FDR set
(alyx ``meta/``) and a copy in repo ``fit_targets/`` for the fitter.

  conda activate iblenv
  python scripts/build_s_prior_curve_unsplit80.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from summarize_prior_earlystim import (  # noqa: E402
    UNSPLIT_SHUFFLE,
    _combined_name,
    _default_meta,
    _default_res,
    _obs_nulls,
    n_bins_le_tmax,
)

WINDOW_MS = 150.0
T_MAX_MS = 80.0
ALPHA = 0.01
NCLUS_SPLIT = "act_block_duringstim_l"

REPO_ROOT = Path(__file__).resolve().parents[1]
FIT_TARGETS = REPO_ROOT / "fit_targets"
REGTYPE_CSV = REPO_ROOT / "data" / "stimchoice_act_regtype_regions_p_mean_c_0.01.csv"
SIDECAR_NAME = "data_act_block_duringstim_s_unsplit80.npy"
SENSORY_REGTYPES = (0.0, 0.1)


def _nclus_map(pth_res: Path, split: str) -> dict[str, int]:
    d = np.load(pth_res / f"{split}.npy", allow_pickle=True).flatten()[0]
    out = {}
    for reg, rec in d.items():
        if isinstance(rec, dict) and "nclus" in rec:
            out[reg] = int(rec["nclus"])
    return out


def _sensory_regions(regtype_csv: Path) -> list[str]:
    rt = pd.read_csv(regtype_csv)
    mask = rt["sc_duringstim_regtype"].isin(SENSORY_REGTYPES)
    return [str(r) for r in rt.loc[mask, "region"].tolist()]


def build_curve(
    pth_res: Path,
    meta_dir: Path,
    alpha: float,
    t_max_ms: float,
    regtype_csv: Path,
):
    csv = meta_dir / f"table_act_block_earlystim_{t_max_ms:g}ms_p_mean.csv"
    if not csv.is_file():
        raise FileNotFoundError(
            f"{csv} missing — run scripts/summarize_prior_earlystim.py first"
        )
    if not Path(regtype_csv).is_file():
        raise FileNotFoundError(regtype_csv)
    df = pd.read_csv(csv)
    arm = df[(df["conditioning"] == "unsplit") & (df["null"] == "shuffle")]
    if arm.empty:
        raise RuntimeError(f"no unsplit/shuffle rows in {csv}")
    hits = arm.loc[arm["p_mean_c"] <= alpha].sort_values(
        ["p_mean_c", "region"], kind="mergesort"
    )
    fdr_regs = [str(r) for r in hits["region"].tolist()]
    if not fdr_regs:
        raise RuntimeError(f"no FDR@{alpha:g} hits in unsplit/shuffle 80 ms")
    sensory = _sensory_regions(regtype_csv)
    sensory_set = set(sensory)
    regs = [r for r in fdr_regs if r in sensory_set]
    if not regs:
        raise RuntimeError(
            f"empty intersection: FDR@{alpha:g} unsplit-80 ({len(fdr_regs)}) "
            f"∩ sensory regtype {SENSORY_REGTYPES} ({len(sensory)})"
        )

    splits = list(UNSPLIT_SHUFFLE)
    combined = _combined_name(splits)
    regde_path = pth_res / f"combined_regde_{'_'.join(splits)}.npy"
    if not regde_path.is_file():
        raise FileNotFoundError(regde_path)
    print(f"regde={regde_path}")
    print(
        f"FDR@{alpha:g} unsplit-80={len(fdr_regs)}  "
        f"sensory(regtype {SENSORY_REGTYPES})={len(sensory)}  "
        f"intersection={len(regs)}: {', '.join(regs)}"
    )

    regde = np.load(regde_path, allow_pickle=True).item()
    sample = _obs_nulls(next(iter(regde.values())))[0]
    n_full = int(np.asarray(sample).reshape(-1).shape[0])
    n_keep = n_bins_le_tmax(n_full, WINDOW_MS, t_max_ms)
    nclus = _nclus_map(pth_res, NCLUS_SPLIT)

    weighted_diff_80 = None
    weighted_diff_150 = None
    weighted_obs_150 = None
    weighted_null_150 = None
    n_cells = 0
    used = []
    skipped = []
    n_splits = len(splits)
    for reg in regs:
        if reg not in regde:
            skipped.append((reg, "not_in_combine"))
            continue
        n = int(nclus.get(reg, 0))
        if n <= 0:
            skipped.append((reg, "nclus=0"))
            continue
        obs, nulls = _obs_nulls(regde[reg])
        obs = np.asarray(obs, dtype=float)[:n_full] / n_splits
        null_mean = np.mean(np.asarray(nulls, dtype=float)[:, :n_full], axis=0) / n_splits
        diff_150 = obs - null_mean
        diff_80 = diff_150[:n_keep]
        weighted_diff_80 = diff_80 * n if weighted_diff_80 is None else weighted_diff_80 + diff_80 * n
        weighted_diff_150 = (
            diff_150 * n if weighted_diff_150 is None else weighted_diff_150 + diff_150 * n
        )
        weighted_obs_150 = obs * n if weighted_obs_150 is None else weighted_obs_150 + obs * n
        weighted_null_150 = (
            null_mean * n if weighted_null_150 is None else weighted_null_150 + null_mean * n
        )
        n_cells += n
        used.append(reg)
    del regde

    if weighted_diff_80 is None or n_cells == 0:
        raise RuntimeError("no regions contributed to the S curve")
    r_stim = weighted_diff_80 / n_cells
    r_stim_150 = weighted_diff_150 / n_cells
    r_obs = weighted_obs_150 / n_cells
    r_null = weighted_null_150 / n_cells
    payload = {
        "regs_stim": used,
        "regs_stim_requested": regs,
        "regs_fdr_unsplit80": fdr_regs,
        "regs_sensory": sensory,
        "skipped": skipped,
        "r_stim": np.asarray(r_stim, dtype=float),
        "r_stim_150ms": np.asarray(r_stim_150, dtype=float),
        "r_obs": np.asarray(r_obs, dtype=float),
        "r_null": np.asarray(r_null, dtype=float),
        "n_keep": int(n_keep),
        "n_full": int(n_full),
        "t_max_ms": float(t_max_ms),
        "window_ms": float(WINDOW_MS),
        "alpha": float(alpha),
        "n_cells": int(n_cells),
        "n_regions": len(used),
        "conditioning": "unsplit",
        "null": "shuffle",
        "prior": "act",
        "splits": splits,
        "n_splits": n_splits,
        "nclus_split": NCLUS_SPLIT,
        "source_combine": combined,
        "source_regde": str(regde_path),
        "source_fdr_csv": str(csv),
        "source_regtype_csv": str(Path(regtype_csv).resolve()),
        "sensory_regtypes": list(SENSORY_REGTYPES),
        "region_rule": "unsplit80_fdr ∩ stim|stim_early",
    }
    return payload


def plot_curve(payload: dict, out_png: Path):
    r = np.asarray(payload["r_stim"], dtype=float)
    t = np.linspace(0.0, float(payload["t_max_ms"]), r.size)
    fig, ax = plt.subplots(figsize=(4.2, 2.4), dpi=150, facecolor="white")
    ax.set_facecolor("white")
    ax.plot(t, r, color="C0", lw=2.0)
    ax.set_xlabel("time from stimOn (ms)")
    ax.set_ylabel(r"$d^{\mathrm{prior}}_{\mathrm{S}}(t)$")
    ax.set_title(
        f"S prior (unsplit 80 ms ∩ stim/stim_early, "
        f"FDR@{payload['alpha']:g}, n={payload['n_regions']}/"
        f"{payload['n_cells']} cells)"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", type=Path, default=_default_res())
    ap.add_argument("--meta-dir", type=Path, default=_default_meta())
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--t-max-ms", type=float, default=T_MAX_MS)
    ap.add_argument(
        "--fit-targets", type=Path, default=FIT_TARGETS,
        help="also write the sidecar here for joint/weights drivers",
    )
    ap.add_argument(
        "--regtype-csv", type=Path, default=REGTYPE_CSV,
        help="SC duringstim labels (stim=0, stim_early=0.1)",
    )
    args = ap.parse_args()

    payload = build_curve(
        args.res, args.meta_dir, args.alpha, args.t_max_ms, args.regtype_csv,
    )
    print(
        f"curve n_reg={payload['n_regions']} n_cells={payload['n_cells']} "
        f"bins={payload['n_keep']}  "
        f"min={float(np.min(payload['r_stim'])):.4f} "
        f"max={float(np.max(payload['r_stim'])):.4f} "
        f"end={float(payload['r_stim'][-1]):.4f}"
    )
    if payload["skipped"]:
        print(f"skipped {len(payload['skipped'])}: {payload['skipped']}")

    args.meta_dir.mkdir(parents=True, exist_ok=True)
    meta_npy = args.meta_dir / SIDECAR_NAME
    np.save(meta_npy, payload, allow_pickle=True)
    plot_curve(payload, args.meta_dir / SIDECAR_NAME.replace(".npy", ".png"))
    print(f"wrote {meta_npy}")

    if args.fit_targets is not None:
        args.fit_targets.mkdir(parents=True, exist_ok=True)
        fit_npy = args.fit_targets / SIDECAR_NAME
        np.save(fit_npy, payload, allow_pickle=True)
        print(f"wrote {fit_npy}")

    summary = {
        k: (v if not isinstance(v, np.ndarray) else f"array{tuple(v.shape)}")
        for k, v in payload.items()
        if k not in (
            "r_stim", "r_stim_150ms", "r_obs", "r_null",
            "regs_fdr_unsplit80", "regs_sensory",
        )
    }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
