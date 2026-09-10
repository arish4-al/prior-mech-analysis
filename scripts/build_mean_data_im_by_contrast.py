"""Build data I/M traj by contrast (notebook RMS recipe).

Pooled I/M in ``paper-brain-wide-map/model_test.ipynb`` (cell that writes
``mean_data_results.npy``) and the older
``paper-brain-wide-map/get_data_for_fitting.py``:

  * load ``concat_act_normFalse.npy``
  * keep cells whose Beryl acronym is in ``int_regs`` (I) or ``move_regs`` (M)
  * for each of 8 stim + 8 choice keys, slice ``concat`` with ``sum_for_key``
  * store **RMS** (not the cell-mean)::

        rms = sqrt(nansum(subset**2, 0) / n_cells / T_BIN)

    ``T_BIN = 0.0125``. The notebook does **not** subtract ``rms[0]`` for I/M
    (``get_data_for_fitting.py`` does; we follow the notebook / fit_targets).

S already uses ``concat_by_contrast_act_noshuffle`` with the same RMS.
This script applies that recipe to I/M. Preferred source is the all-cell
noshuffle concat (80 keys = 8 stim + 8 choice × 5 contrasts). That file is
not on this laptop. Fallback: act-prior ``shuffleTrue_{integrator,move_init}``
concats — ``concat`` is still the real PETH (shuffle is only
``distance_controls``); keys are the **correct-only** 4+4 subset
(``block_pairs_by_contrast``).

Writes next to the concat (ONE ``dmn/res/mean_data_im_by_contrast/``), not
repo ``figs/``. Does not replace ``fit_targets/mean_data_results.npy``.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from iblatlas.regions import BrainRegions

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _fit_data import load_validated_mean_data  # noqa: E402

T_BIN = 0.0125
CONTRASTS = (1.0, 0.25, 0.125, 0.0625, 0.0)
STIM_KEYS = (
    "stimLbLcL", "stimLbRcL", "stimRbLcR", "stimRbRcR",
    "stimLbLcR", "stimRbLcL", "stimLbRcR", "stimRbRcL",
)
CHOICE_KEYS = (
    "sLbLchoiceL", "sLbRchoiceL", "sRbLchoiceR", "sRbRchoiceR",
    "sRbRchoiceL", "sLbRchoiceR", "sLbLchoiceR", "sRbLchoiceL",
)

DMN = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/dmn/res"
)
PREFERRED = DMN / "concat_by_contrast_act_noshuffle_normFalse.npy"
FALLBACK_I = DMN / "concat_by_contrast_act_normFalse_shuffleTrue_integrator.npy"
FALLBACK_M = DMN / "concat_by_contrast_act_normFalse_shuffleTrue_move_init.npy"
OUT = DMN / "mean_data_im_by_contrast"


def sum_for_key(length_dict, key, after=False):
    """Notebook / ``dmn_ari.sum_for_key``: running sum in dict order."""
    key = str(key)
    total = 0
    found = False
    for k, v in length_dict.items():
        if str(k) == key:
            found = True
            if after:
                total += int(v)
            break
        total += int(v)
    if not found:
        raise KeyError(key)
    return total


def rms_curve(subset):
    """Notebook I/M ``mean_traj``: RMS across cells, no t0 subtract."""
    n = subset.shape[0]
    return np.sqrt((np.nansum(subset ** 2, axis=0) / n) / T_BIN)


def load_concat(path: Path):
    d = np.load(path, allow_pickle=True).flat[0]
    if "concat" not in d or d["concat"] is None:
        raise KeyError(f"{path.name}: no concat (only concat_z / distances)")
    lengths = {str(k): int(v) for k, v in d["len"].items()}
    return d, lengths


def acronyms(ids):
    br = BrainRegions()
    return np.asarray(br.id2acronym(ids, mapping="Beryl"))


def filter_cells(d, regs):
    acs = acronyms(d["ids"])
    keep = np.array([a in regs for a in acs], dtype=bool)
    return keep, acs


def slice_key(concat, lengths, key):
    start = sum_for_key(lengths, key, after=False)
    end = sum_for_key(lengths, key, after=True)
    return concat[:, start:end]


def keys_present(lengths, bases):
    out = defaultdict(list)
    for c in CONTRASTS:
        for base in bases:
            k = f"{base}_{c}"
            if k in lengths:
                out[c].append(base)
    return dict(out)


def build_one(path, regs, label):
    d, lengths = load_concat(path)
    keep, acs = filter_cells(d, set(regs))
    concat = np.asarray(d["concat"], float)[keep]
    n = int(keep.sum())
    print(
        f"{label}: {path.name}  cells {len(keep)} → {n} in {len(regs)} regs  "
        f"keys={len(lengths)}",
        flush=True,
    )
    if n < 10:
        raise RuntimeError(f"{label}: only {n} cells after region filter")

    stim_ok = keys_present(lengths, STIM_KEYS)
    choice_ok = keys_present(lengths, CHOICE_KEYS)
    mean_traj = {}
    n_used = {}
    for c in CONTRASTS:
        stim = {}
        choice = {}
        for base in stim_ok.get(c, []):
            stim[base] = rms_curve(slice_key(concat, lengths, f"{base}_{c}"))
        for base in choice_ok.get(c, []):
            choice[base] = rms_curve(slice_key(concat, lengths, f"{base}_{c}"))
        mean_traj[c] = {"stim": stim, "choice": choice}
        n_used[c] = {"stim": list(stim), "choice": list(choice)}
        print(
            f"  c={c}: stim {len(stim)}/{len(STIM_KEYS)}  "
            f"choice {len(choice)}/{len(CHOICE_KEYS)}",
            flush=True,
        )
    return {
        "regs": list(regs),
        "n_cells": n,
        "n_cells_file": int(len(keep)),
        "source": str(path),
        "keys_used": n_used,
        "mean_traj": mean_traj,
        "acs_kept": sorted(set(acs[keep].tolist())),
    }


def collapse_choice(traj_win, ch_is_left):
    """Same collapse as ``_data_mean_and_baseline``: mean over ts,sp at one choice."""
    if ch_is_left:
        want = [k for k in traj_win if k.endswith("cL") or k.endswith("choiceL")]
    else:
        want = [k for k in traj_win if k.endswith("cR") or k.endswith("choiceR")]
    curves = [np.asarray(traj_win[k], float) for k in want if k in traj_win]
    if not curves:
        return None
    T = min(c.size for c in curves)
    return np.mean([c[:T] for c in curves], axis=0)


def plot_crf(payload, out_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.4), sharex="col")
    for row, vn in enumerate(("I", "M")):
        mt = payload[vn]["mean_traj"]
        for col, win in enumerate(("stim", "choice")):
            ax = axes[row, col]
            for ch_left, name, color in ((True, "choice L", "C0"), (False, "choice R", "C1")):
                ys = []
                for c in CONTRASTS:
                    curve = collapse_choice(mt[c][win], ch_left)
                    if curve is None or curve.size <= 15:
                        ys.append(np.nan)
                    else:
                        ys.append(float(np.nanmean(curve[15:])))
                ax.plot(CONTRASTS, ys, "-o", color=color, label=name)
            ax.set_title(f"{vn} {win}  mean RMS after bin 15")
            ax.set_xscale("symlog", linthresh=0.05)
            ax.legend(frameon=False, fontsize=8)
            if row == 1:
                ax.set_xlabel("|contrast|")
            if col == 0:
                ax.set_ylabel("RMS")
    fig.tight_layout()
    for ext in (".svg", ".png"):
        fig.savefig(out_dir / f"im_crf_amp{ext}", bbox_inches="tight",
                    facecolor="white", transparent=False)
    plt.close(fig)

    for vn in ("I", "M"):
        mt = payload[vn]["mean_traj"]
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), sharey=False)
        for ax, win in zip(axes, ("stim", "choice")):
            for c in CONTRASTS:
                curve = collapse_choice(mt[c][win], True)
                if curve is None:
                    continue
                x = np.linspace(0.0, 200.0 if win == "stim" else 150.0, curve.size)
                ax.plot(x, curve, label=f"c={c}", lw=1.2)
            ax.set_title(f"{vn} {win}  choice-L collapse")
            ax.set_xlabel("time (ms)")
            ax.legend(frameon=False, fontsize=7, ncol=2)
        axes[0].set_ylabel("RMS")
        fig.tight_layout()
        for ext in (".svg", ".png"):
            fig.savefig(out_dir / f"{vn}_traces_by_contrast{ext}", bbox_inches="tight",
                        facecolor="white", transparent=False)
        plt.close(fig)


def _to_native(obj):
    if isinstance(obj, dict):
        return {str(k) if not isinstance(k, float) else k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main():
    _, mean_data = load_validated_mean_data()
    int_regs = list(mean_data["I"]["regs"])
    move_regs = list(mean_data["M"]["regs"])
    OUT.mkdir(parents=True, exist_ok=True)

    if PREFERRED.is_file():
        print(f"preferred all-cell file: {PREFERRED}")
        i_src = m_src = PREFERRED
        source_note = "all_cell_noshuffle"
    else:
        print(
            f"preferred missing: {PREFERRED.name}\n"
            f"  fallback I={FALLBACK_I.name}\n"
            f"  fallback M={FALLBACK_M.name}\n"
            "  (concat is real PETH; keys are correct-only 4+4)",
            flush=True,
        )
        i_src, m_src = FALLBACK_I, FALLBACK_M
        source_note = "raster_restricted_correct_only"

    payload = {
        "meta": {
            "recipe": "model_test.ipynb I/M RMS (T_BIN=0.0125, no t0 subtract)",
            "t_bin": T_BIN,
            "source_note": source_note,
            "preferred_missing": not PREFERRED.is_file(),
            "stim_keys_canonical": list(STIM_KEYS),
            "choice_keys_canonical": list(CHOICE_KEYS),
            "contrasts": list(CONTRASTS),
        },
        "I": build_one(i_src, int_regs, "I"),
        "M": build_one(m_src, move_regs, "M"),
    }
    out_npy = OUT / "mean_data_results_by_contrast.npy"
    np.save(out_npy, payload, allow_pickle=True)
    plot_crf(payload, OUT)
    meta_path = OUT / "meta.json"
    slim = {
        "meta": payload["meta"],
        "I": {k: payload["I"][k] for k in ("n_cells", "n_cells_file", "source", "keys_used")},
        "M": {k: payload["M"][k] for k in ("n_cells", "n_cells_file", "source", "keys_used")},
        "out": str(out_npy),
    }
    meta_path.write_text(json.dumps(slim, indent=2, default=str))
    print(f"wrote {out_npy}")
    print(f"wrote plots in {OUT}")


if __name__ == "__main__":
    main()
