"""I/M mean RMS from the insertion cache (no concat, no nulls, no act-prior).

What the fit actually scores (``_data_mean_and_baseline``): I/M RMS in
the stim and choice windows, collapsed to **choice L vs R**. This driver
builds those curves, by contrast and pooled, without the 8 stim×prior×choice
notebook keys.

Regions default to the current stim×choice table
(``data/stimchoice_act_regtype_regions_p_mean_c_0.01.csv``):

  * I = duringstim ∨ duringchoice label ``integrator`` (60)
  * M = duringstim ∨ duringchoice label ``move`` (23)

That is a subset of the old ``fit_targets`` lists (81 / 26). The extras
are unclassified or stim-early only — not current I/M. ``--regs
fit_targets`` restores the old lists.

Per insertion: keep I/M cells, bin **once** per alignment, slice by
choice × contrast. RMS = ``sqrt(sum(x²) / n / T_BIN)``.

Writes ``manifold/mean_data_im_from_cache/``. Does not replace
``fit_targets/mean_data_results.npy``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brainbox.singlecell import bin_spikes2D
from iblatlas.regions import BrainRegions

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _fit_data import load_validated_mean_data  # noqa: E402

T_BIN = 0.0125
STS = 0.002
CONTRASTS = (1.0, 0.25, 0.125, 0.0625, 0.0)
CHOICES = (("L", 1), ("R", -1))
REGTYPE_CSV = ROOT / "data" / "stimchoice_act_regtype_regions_p_mean_c_0.01.csv"
DEFAULT_CACHE = Path.home() / (
    "Downloads/ONE/alyx.internationalbrainlab.org/manifold/insertion_cache"
)


def n_bins(pre, post, t_bin=T_BIN, sts=STS):
    return int(t_bin // sts) * int(round((pre + post) / t_bin))


def bin_aligned(times, spike_clu, cluster_ids, events, pre, post):
    events = np.asarray(events, dtype=float)
    if events.size == 0:
        return np.zeros((0, len(cluster_ids), n_bins(pre, post)))
    st = int(T_BIN // STS)
    bis = []
    for ts in range(st):
        bi, _ = bin_spikes2D(
            times, spike_clu, cluster_ids,
            events + ts * STS, pre, post, T_BIN,
        )
        bis.append(bi)
    ntr, nn, nbin = bis[0].shape
    ar = np.zeros((ntr, nn, st * nbin), dtype=np.float64)
    for ts in range(st):
        ar[:, :, ts::st] = bis[ts]
    return ar


class RmsAcc:
    def __init__(self, nbin):
        self.nbin = int(nbin)
        self.sumsq = {}
        self.n = {}

    def add(self, key, cell_means):
        x = np.asarray(cell_means, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        if x.size == 0:
            return
        x = x[~np.isnan(x).all(axis=1)]
        if x.shape[0] == 0:
            return
        if key not in self.sumsq:
            self.sumsq[key] = np.zeros(self.nbin, dtype=np.float64)
            self.n[key] = 0
        self.sumsq[key] += np.nansum(x ** 2, axis=0)
        self.n[key] += int(x.shape[0])

    def rms(self, key):
        n = self.n.get(key, 0)
        if n == 0:
            return None
        return np.sqrt(self.sumsq[key] / n / T_BIN)


def load_sc_im_regs(csv_path=REGTYPE_CSV):
    """Current stim×choice I/M: union of duringstim / duringchoice labels."""
    df = pd.read_csv(csv_path)
    int_regs, move_regs = set(), set()
    for col in ("duringstim_label", "duringchoice_label"):
        int_regs |= set(df.loc[df[col] == "integrator", "region"])
        move_regs |= set(df.loc[df[col] == "move", "region"])
    if int_regs & move_regs:
        raise ValueError(f"I/M overlap: {sorted(int_regs & move_regs)}")
    return int_regs, move_regs


def load_fit_target_regs():
    _, md = load_validated_mean_data()
    return set(md["I"]["regs"]), set(md["M"]["regs"])


def contrast_mag(trials):
    cl = trials["contrastLeft"].to_numpy(dtype=float)
    cr = trials["contrastRight"].to_numpy(dtype=float)
    return np.nanmax(np.stack([cl, cr], axis=0), axis=0)


def subset_spikes(spikes, clusters, keep):
    cid_all = np.asarray(clusters["cluster_id"])
    cid = cid_all[keep]
    spk_cid = cid_all[np.asarray(spikes["clusters"])]
    in_keep = np.isin(spk_cid, cid)
    return np.asarray(spikes["times"])[in_keep], spk_cid[in_keep], cid


def process_insertion(cache, int_regs, move_regs, br, acc, stim_pre_post,
                      min_trials):
    acs = np.asarray(br.id2acronym(cache["clusters"]["atlas_id"], mapping="Beryl"))
    is_i = np.array([a in int_regs for a in acs], dtype=bool)
    is_m = np.array([a in move_regs for a in acs], dtype=bool)
    keep = is_i | is_m
    n_i, n_m = int(is_i.sum()), int(is_m.sum())
    if n_i + n_m == 0:
        return 0, 0, 0
    times, spk_cid, cid = subset_spikes(cache["spikes"], cache["clusters"], keep)
    node = np.where(np.array([a in int_regs for a in acs[keep]]), "I", "M")

    stim_trials = cache["trials"]["saturation_stim_plus04"]
    move_trials = cache["trials"]["saturation_move_minus02"]
    stim_ev = stim_trials["stimOn_times"].to_numpy(dtype=float)
    move_ev = move_trials["firstMovement_times"].to_numpy(dtype=float)
    ok_s, ok_m = np.isfinite(stim_ev), np.isfinite(move_ev)
    b_stim = bin_aligned(times, spk_cid, cid, stim_ev[ok_s], *stim_pre_post)
    b_choice = bin_aligned(times, spk_cid, cid, move_ev[ok_m], 0.15, 0.0)
    stim_trials = stim_trials.iloc[np.flatnonzero(ok_s)].reset_index(drop=True)
    move_trials = move_trials.iloc[np.flatnonzero(ok_m)].reset_index(drop=True)
    cmag_s = contrast_mag(stim_trials)
    cmag_m = contrast_mag(move_trials)

    def _add(win, binned, trials, cmag, contrast, tag):
        ch = trials["choice"].to_numpy(dtype=int)
        for ch_name, ch_val in CHOICES:
            mask = ch == ch_val
            if contrast is not None:
                mask = mask & np.isclose(cmag, contrast)
            if int(mask.sum()) < min_trials:
                continue
            mean = binned[mask].mean(axis=0)
            for vn in ("I", "M"):
                sel = node == vn
                if sel.any():
                    acc[vn][win].add((tag, ch_name), mean[sel])

    for c in CONTRASTS:
        _add("stim", b_stim, stim_trials, cmag_s, c, c)
        _add("choice", b_choice, move_trials, cmag_m, c, c)
    _add("stim", b_stim, stim_trials, cmag_s, None, "all")
    _add("choice", b_choice, move_trials, cmag_m, None, "all")
    return n_i, n_m, 1


def get_curve(mt, tag, win, ch):
    d = mt.get(tag, {}).get(win, {})
    arr = d.get(ch)
    return None if arr is None else np.asarray(arr, float)


def plot_crf(payload, out_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.4), sharex="col")
    for row, vn in enumerate(("I", "M")):
        mt = payload[vn]["mean_traj"]
        for col, win in enumerate(("stim", "choice")):
            ax = axes[row, col]
            for ch, color in (("L", "C0"), ("R", "C1")):
                ys = []
                for c in CONTRASTS:
                    curve = get_curve(mt, c, win, ch)
                    ys.append(
                        float(np.nanmean(curve[15:]))
                        if curve is not None and curve.size > 15 else np.nan
                    )
                ax.plot(CONTRASTS, ys, "-o", color=color, label=f"choice {ch}")
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
            for c in list(CONTRASTS) + ["all"]:
                curve = get_curve(mt, c, win, "L")
                if curve is None:
                    continue
                ax.plot(
                    np.linspace(0.0, 150.0, curve.size), curve,
                    label=f"c={c}", lw=1.2, ls="--" if c == "all" else "-",
                )
            ax.set_title(f"{vn} {win}  choice L")
            ax.set_xlabel("time (ms)")
            ax.legend(frameon=False, fontsize=7, ncol=2)
        axes[0].set_ylabel("RMS")
        fig.tight_layout()
        for ext in (".svg", ".png"):
            fig.savefig(
                out_dir / f"{vn}_traces_by_contrast{ext}",
                bbox_inches="tight", facecolor="white", transparent=False,
            )
        plt.close(fig)


def collapse_fit_targets(mt_win, ch_is_left):
    if ch_is_left:
        want = [k for k in mt_win if str(k).endswith("cL") or str(k).endswith("choiceL")]
    else:
        want = [k for k in mt_win if str(k).endswith("cR") or str(k).endswith("choiceR")]
    curves = [np.asarray(mt_win[k], float) for k in want if k in mt_win]
    if not curves:
        return None
    T = min(c.size for c in curves)
    return np.mean([c[:T] for c in curves], axis=0)


def compare_fit_targets(payload, mean_data, out_dir: Path):
    lines = []
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.0), sharex="col")
    for row, vn in enumerate(("I", "M")):
        for col, win in enumerate(("stim", "choice")):
            ax = axes[row, col]
            ours = get_curve(payload[vn]["mean_traj"], "all", win, "L")
            tgt = collapse_fit_targets(mean_data[vn]["mean_traj"][win], True)
            if ours is None or tgt is None:
                continue
            if tgt.size > ours.size:
                tgt = tgt[: ours.size]
            T = min(ours.size, tgt.size)
            x = np.linspace(0.0, 150.0, T)
            ax.plot(x, tgt[:T], color="0.4", label="fit_targets (all-c)")
            ax.plot(x, ours[:T], color="C3", label="cache (all-c)")
            r = np.corrcoef(tgt[:T], ours[:T])[0, 1] if T > 2 else np.nan
            scale = (
                float(np.nanmean(ours[:T]) / np.nanmean(tgt[:T]))
                if np.nanmean(tgt[:T]) else np.nan
            )
            ax.set_title(f"{vn} {win}  r={r:.3f}  scale={scale:.2f}")
            ax.legend(frameon=False, fontsize=8)
            lines.append(
                f"{vn} {win}: r={r:.4f} scale={scale:.3f} "
                f"ours_mean={float(np.nanmean(ours[:T])):.3f} "
                f"tgt_mean={float(np.nanmean(tgt[:T])):.3f}"
            )
            if row == 1:
                ax.set_xlabel("time (ms)")
            if col == 0:
                ax.set_ylabel("RMS")
    fig.tight_layout()
    for ext in (".svg", ".png"):
        fig.savefig(out_dir / f"allcontrast_vs_fit_targets{ext}",
                    bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--n-insertions", type=int, default=None)
    p.add_argument("--min-trials", type=int, default=1)
    p.add_argument("--stim-post", type=float, default=0.15)
    p.add_argument(
        "--regs", choices=("sc", "fit_targets"), default="sc",
        help="sc = current stim×choice integrator/move; "
             "fit_targets = old 81/26 lists",
    )
    p.add_argument("--regtype-csv", type=Path, default=REGTYPE_CSV)
    args = p.parse_args()

    cache_dir = args.cache_dir.expanduser().resolve()
    files = sorted(cache_dir.glob("*.npy"))
    if args.n_insertions is not None:
        files = files[: args.n_insertions]
    if not files:
        raise SystemExit(f"No insertion_cache/*.npy in {cache_dir}")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = cache_dir.parent / "mean_data_im_from_cache"
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ft_I, ft_M = load_fit_target_regs()
    sc_I, sc_M = load_sc_im_regs(args.regtype_csv)
    if args.regs == "sc":
        int_regs, move_regs = sc_I, sc_M
    else:
        int_regs, move_regs = ft_I, ft_M
    print(
        f"regs={args.regs}  I={len(int_regs)} M={len(move_regs)}  "
        f"(sc {len(sc_I)}/{len(sc_M)}; fit_targets {len(ft_I)}/{len(ft_M)})",
        flush=True,
    )
    print(f"  I extras vs sc (dropped if --regs sc): {sorted(ft_I - sc_I)}")
    print(f"  M extras vs sc (dropped if --regs sc): {sorted(ft_M - sc_M)}")

    _, mean_data = load_validated_mean_data()
    br = BrainRegions()
    nbin_stim = n_bins(0.0, args.stim_post)
    nbin_choice = n_bins(0.15, 0.0)
    acc = {
        "I": {"stim": RmsAcc(nbin_stim), "choice": RmsAcc(nbin_choice)},
        "M": {"stim": RmsAcc(nbin_stim), "choice": RmsAcc(nbin_choice)},
    }

    n_i = n_m = n_used = 0
    t0 = time.time()
    print(f"cache {cache_dir}  n={len(files)}  stim_post={args.stim_post} "
          f"bins={nbin_stim}/{nbin_choice}", flush=True)

    for i, f in enumerate(files, 1):
        cache = np.load(f, allow_pickle=True).item()
        ni, nm, used = process_insertion(
            cache, int_regs, move_regs, br, acc,
            (0.0, args.stim_post), args.min_trials,
        )
        n_i += ni
        n_m += nm
        n_used += used
        print(f"  [{i}/{len(files)}] {f.name}  I={ni} M={nm}", flush=True)

    payload = {
        "meta": {
            "recipe": "insertion_cache I/M RMS, choice×contrast, no act-prior",
            "t_bin": T_BIN,
            "sts": STS,
            "stim_window": [0.0, args.stim_post],
            "choice_window": [0.15, 0.0],
            "n_bins": {"stim": nbin_stim, "choice": nbin_choice},
            "cache_dir": str(cache_dir),
            "n_insertions": n_used,
            "n_insertions_listed": len(files),
            "n_cells": {"I": n_i, "M": n_m},
            "min_trials": args.min_trials,
            "regs_source": args.regs,
            "regtype_csv": str(args.regtype_csv),
            "elapsed_s": time.time() - t0,
        },
    }
    for vn in ("I", "M"):
        mean_traj = {}
        n_used_k = {}
        for tag in list(CONTRASTS) + ["all"]:
            stim, choice = {}, {}
            for ch, _ in CHOICES:
                r = acc[vn]["stim"].rms((tag, ch))
                if r is not None:
                    stim[ch] = r
                r = acc[vn]["choice"].rms((tag, ch))
                if r is not None:
                    choice[ch] = r
            mean_traj[tag] = {"stim": stim, "choice": choice}
            n_used_k[str(tag)] = {
                "stim": {ch: acc[vn]["stim"].n.get((tag, ch), 0) for ch, _ in CHOICES},
                "choice": {ch: acc[vn]["choice"].n.get((tag, ch), 0) for ch, _ in CHOICES},
            }
        payload[vn] = {
            "regs": sorted(int_regs if vn == "I" else move_regs),
            "n_cells": n_i if vn == "I" else n_m,
            "mean_traj": mean_traj,
            "n_used": n_used_k,
        }

    out_npy = out_dir / "mean_data_results_by_contrast.npy"
    np.save(out_npy, payload, allow_pickle=True)
    plot_crf(payload, out_dir)
    cmp_lines = compare_fit_targets(payload, mean_data, out_dir)

    print("\nChoice-L, mean RMS after bin 15:")
    print(f"{'|c|':>8} {'I stim':>8} {'I choice':>8} {'M stim':>8} {'M choice':>8}")
    crf = {}
    for c in list(CONTRASTS) + ["all"]:
        row = {}
        for vn in ("I", "M"):
            for win in ("stim", "choice"):
                curve = get_curve(payload[vn]["mean_traj"], c, win, "L")
                row[f"{vn}_{win}"] = (
                    float(np.nanmean(curve[15:]))
                    if curve is not None and curve.size > 15 else np.nan
                )
        crf[str(c)] = row
        print(f"{str(c):>8} {row['I_stim']:8.3f} {row['I_choice']:8.3f} "
              f"{row['M_stim']:8.3f} {row['M_choice']:8.3f}")
    print("\nvs fit_targets all-contrast (choice-L collapse):")
    for line in cmp_lines:
        print(" ", line)

    slim = {
        "meta": payload["meta"],
        "I": {"n_cells": payload["I"]["n_cells"], "n_regs": len(payload["I"]["regs"]),
              "regs": payload["I"]["regs"], "n_used": payload["I"]["n_used"]},
        "M": {"n_cells": payload["M"]["n_cells"], "n_regs": len(payload["M"]["regs"]),
              "regs": payload["M"]["regs"], "n_used": payload["M"]["n_used"]},
        "crf_choiceL_after15": crf,
        "vs_fit_targets": cmp_lines,
        "out": str(out_npy),
    }
    (out_dir / "meta.json").write_text(json.dumps(slim, indent=2, default=str))
    print(f"\nwrote {out_npy}")
    print(f"wrote plots in {out_dir}")
    if n_used < 50:
        print(
            "\nNOTE: laptop smoke cache only (7 insertions). Full BWM on ORCD:\n"
            "  python scripts/build_mean_data_im_from_cache.py "
            "--cache-dir $ONE_CACHE/manifold/insertion_cache\n"
        )


if __name__ == "__main__":
    main()
