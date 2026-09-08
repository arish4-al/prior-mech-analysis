#!/usr/bin/env python
"""Prior-distance curves for sensory regions with unsplit 80 ms FDR.

Same shuffle-overlay style as ``plot_regional_distance`` /
``plot_average_distance_over_regions`` (and
``plot_mixed_region_distance_trajectories``). Region set is the
unsplit-80 FDR@0.01 ∩ stim/stim_early list from
``build_s_prior_curve_unsplit80.py``.

Writes into the alyx analysis folder next to the earlystim tables:

  <alyx>/manifold/figs/earlystim80_sensory_prior/

  conda activate iblenv
  python scripts/plot_earlystim_sensory_prior_distance.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_s_prior_curve_unsplit80 import (  # noqa: E402
    NCLUS_SPLIT,
    REGTYPE_CSV,
    _nclus_map,
    build_curve,
)
from plot_mixed_region_distance_trajectories import (  # noqa: E402
    _compute_p_gain,
    _draw_panel,
)
from summarize_prior_earlystim import (  # noqa: E402
    UNSPLIT_SHUFFLE,
    _default_meta,
    _default_res,
    _obs_nulls,
    n_bins_le_tmax,
)

WINDOW_MS = 150.0
T_MAX_MS = 80.0
ALPHA = 0.01
TIMEFRAME = "act_block_duringstim_unsplit"
OUT_SUBDIR = "earlystim80_sensory_prior"


def _default_out_dir() -> Path:
    return Path.home() / (
        "Downloads/ONE/alyx.internationalbrainlab.org/manifold/figs"
        f"/{OUT_SUBDIR}"
    )


def _slice_r(r: np.ndarray, n_keep: int) -> np.ndarray:
    return np.asarray(r, dtype=float)[:, :n_keep]


def _p_on_slice(r: np.ndarray) -> dict:
    """Raw p_mean / p_offset / p_gain on an already-sliced (1+U, T) stack."""
    p_mean = float(np.mean(np.mean(r, axis=1) >= np.mean(r[0])))
    mean_first5 = np.mean(r[:, :5], axis=1)
    p_offset = float(np.mean(mean_first5 >= mean_first5[0]))
    p_gain, _ = _compute_p_gain(r, {"p_offset": p_offset}, ALPHA)
    return {
        "p_mean": p_mean,
        "p_offset": p_offset,
        "p_gain": float(p_gain),
    }


def _load_unsplit_r(af, reg: str) -> np.ndarray:
    d_all, r_all, _, _ = af.load_combined_data(TIMEFRAME, dist="de")
    if reg not in r_all:
        raise KeyError(reg)
    r = r_all[reg]
    r = np.concatenate(
        [np.asarray(r[0]).reshape(1, -1), np.asarray(r[1])], axis=0
    )
    r = r / len(af.run_align[TIMEFRAME])
    return r, d_all.get(reg, {})


def _fdr_map(meta_dir: Path, alpha: float) -> dict[str, float]:
    csv = meta_dir / f"table_act_block_earlystim_{T_MAX_MS:g}ms_p_mean.csv"
    df = pd.read_csv(csv)
    arm = df[(df["conditioning"] == "unsplit") & (df["null"] == "shuffle")]
    return {
        str(row.region): float(row.p_mean_c)
        for row in arm.itertuples(index=False)
        if float(row.p_mean_c) <= alpha
    }


def plot_one_region(
    times,
    r,
    d,
    title: str,
    out_path: Path,
    *,
    show_y: bool = True,
):
    fig, axs = plt.subplots(
        1, 2, sharey=True, figsize=(6, 4), dpi=250,
        gridspec_kw={"width_ratios": [6, 1]},
    )
    _draw_panel(
        axs[0], times, r, d, title,
        ptype="p_mean_c", alpha=ALPHA, n_shuf_show=40,
        plot_offset=True, plot_gain=True, show_y=show_y,
    )
    axs[0].set_xticks([0, 40, 80])
    axs[1].hist(
        np.mean(r[1:], axis=1), density=True, bins=20,
        color="silver", orientation="horizontal",
    )
    axs[1].axhline(y=np.mean(r[0]), c="black", linestyle="--")
    if d.get("p_offset", 1) <= ALPHA:
        axs[1].hist(
            np.mean(r[1:, :5], axis=1), density=True, bins=20,
            color="#5f7ea3", orientation="horizontal", alpha=0.5,
        )
        axs[1].axhline(y=np.mean(r[0, :5]), c="blue", linestyle="--")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].spines["bottom"].set_visible(False)
    axs[1].tick_params(axis="y", left=False, labelleft=False)
    axs[1].tick_params(axis="x", bottom=False, labelbottom=False)
    axs[1].set_facecolor("none")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, transparent=True, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_combined(times, r_avg, d, out_path: Path, n_reg: int, n_cells: int):
    fig, axs = plt.subplots(
        1, 2, sharey=True, figsize=(6, 4), dpi=250,
        gridspec_kw={"width_ratios": [6, 1]},
    )
    title = (
        rf"sensory combined $d^{{\mathrm{{prior}},s}}$ "
        rf"(80 ms, n={n_reg}/{n_cells} cells)"
    )
    _draw_panel(
        axs[0], times, r_avg, d, title,
        ptype="p_mean", alpha=ALPHA, n_shuf_show=40,
        plot_offset=True, plot_gain=True, show_y=True,
    )
    axs[0].set_xticks([0, 40, 80])

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    diff = r_avg[0] - np.mean(r_avg[1:], axis=0)
    shuf = r_avg[1:] - np.mean(r_avg[1:], axis=0, keepdims=True)
    lo = np.percentile(shuf, 100 * (ALPHA / 2), axis=0)
    hi = np.percentile(shuf, 100 * (1 - ALPHA / 2), axis=0)
    ax_ins = inset_axes(
        axs[0], width="22%", height="22%", loc="upper right", borderpad=1.2,
    )
    ax_ins.fill_between(times, lo, hi, color="gray", alpha=0.3, linewidth=0)
    ax_ins.plot(times, diff, linewidth=1.2, c="C0")
    ax_ins.set_xticks([])
    ax_ins.set_ylim(-0.05, 0.2)
    ax_ins.set_yticks([0, 0.1, 0.2])
    ax_ins.set_facecolor("none")
    for spine in ("top", "right"):
        ax_ins.spines[spine].set_visible(False)

    axs[1].hist(
        np.mean(r_avg[1:], axis=1), density=True, bins=20,
        color="silver", orientation="horizontal",
    )
    axs[1].axhline(y=np.mean(r_avg[0]), c="C0", linestyle="--")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].spines["bottom"].set_visible(False)
    axs[1].tick_params(axis="y", left=False, labelleft=False)
    axs[1].tick_params(axis="x", bottom=False, labelbottom=False)
    axs[1].set_facecolor("none")
    fig.tight_layout()
    fig.savefig(out_path, transparent=True, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _subtract_shuffle_mean(r: np.ndarray) -> np.ndarray:
    """Each trace minus the per-bin mean of that region's shuffles."""
    return r - np.mean(r[1:], axis=0, keepdims=True)


def _annotate_p(ax, d: dict):
    p_val = float(d.get("p_mean_c", d.get("p_mean", np.nan)))
    ptype = "p_mean_c" if "p_mean_c" in d else "p_mean"
    ax.text(
        0.04, 0.96, f"{ptype}={p_val:.4f}",
        transform=ax.transAxes,
        color="red" if (np.isfinite(p_val) and p_val <= ALPHA) else "black",
        fontsize=11, ha="left", va="top",
    )
    y_ann = 0.84
    if "p_gain" in d:
        pg = float(d["p_gain"])
        ax.text(
            0.04, y_ann, f"p_gain={pg:.4f}",
            transform=ax.transAxes,
            color="red" if pg <= ALPHA else "purple",
            fontsize=9, ha="left", va="top",
        )
        y_ann -= 0.12
    if "p_offset" in d:
        po = float(d["p_offset"])
        ax.text(
            0.04, y_ann, f"p_offset={po:.4f}",
            transform=ax.transAxes,
            color="red" if po <= ALPHA else "#5f7ea3",
            fontsize=9, ha="left", va="top",
        )


def _plot_residual_traces(ax, times, r_b: np.ndarray, n_shuf_show: int = 25):
    n_show = min(n_shuf_show, r_b.shape[0] - 1)
    for j in range(1, n_show + 1):
        ax.plot(times, r_b[j], c="gray", alpha=0.2, linewidth=0.5)
    ax.plot(times, r_b[0], c="black", linewidth=1.5)
    ax.axhline(0, color="k", lw=0.5, alpha=0.35)
    ax.axvline(0, color="k", lw=0.6, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_grid(panels: list[tuple], times, out_path: Path):
    n = len(panels)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), dpi=160, sharex=True,
    )
    axes = np.atleast_2d(axes)
    for i, (reg, r, d) in enumerate(panels):
        ax = axes[i // ncols, i % ncols]
        _draw_panel(
            ax, times, r, d, reg,
            ptype="p_mean_c", alpha=ALPHA, n_shuf_show=25,
            plot_offset=True, plot_gain=True, show_y=(i % ncols == 0),
        )
        ax.set_xticks([0, 40, 80])
        ax.set_xlabel("")
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle(
        "Unsplit 80 ms act-prior, sensory FDR@0.01", fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_grid_baseline_sub(panels: list[tuple], times, out_path: Path):
    """Same 13-region grid; each trace minus that region's shuffle mean(t)."""
    n = len(panels)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), dpi=160,
        sharex=True, sharey=True,
    )
    axes = np.atleast_2d(axes)
    for i, (reg, r, d) in enumerate(panels):
        ax = axes[i // ncols, i % ncols]
        _plot_residual_traces(ax, times, _subtract_shuffle_mean(r))
        _annotate_p(ax, d)
        ax.set_title(reg, fontsize=12)
        ax.set_xticks([0, 40, 80])
        ax.set_xlabel("")
        if i % ncols == 0:
            ax.set_ylabel(r"$d_{\mathrm{euc}}-\overline{\mathrm{shuffle}}$", fontsize=10)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle(
        "Unsplit 80 ms act-prior, sensory FDR@0.01 "
        r"(obs and shuffles minus region shuffle mean at each $t$)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_combined_baseline_sub(
    times, r_avg_b, d, out_path: Path, n_reg: int, n_cells: int,
):
    """Cell-weighted mean of per-region (curve − shuffle mean(t))."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=250)
    _plot_residual_traces(ax, times, r_avg_b, n_shuf_show=40)
    _annotate_p(ax, d)
    ax.set_xticks([0, 40, 80])
    ax.set_xlabel("time from stimOn (ms)", fontsize=10)
    ax.set_ylabel(r"$d_{\mathrm{euc}}-\overline{\mathrm{shuffle}}$", fontsize=10)
    ax.set_title(
        rf"sensory combined $d^{{\mathrm{{prior}},s}}$ "
        rf"(shuffle-mean subtracted, then cell-weighted; n={n_reg}/{n_cells} cells)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", type=Path, default=_default_res())
    ap.add_argument("--meta-dir", type=Path, default=_default_meta())
    ap.add_argument("--out-dir", type=Path, default=_default_out_dir())
    ap.add_argument("--regtype-csv", type=Path, default=REGTYPE_CSV)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument(
        "--only-baseline-grid",
        action="store_true",
        help="Only write shuffle-mean-subtracted grid and cell-weighted avg",
    )
    args = ap.parse_args()

    payload = build_curve(
        args.res, args.meta_dir, args.alpha, T_MAX_MS, args.regtype_csv,
    )
    regs = list(payload["regs_stim"])
    fdr = _fdr_map(args.meta_dir, args.alpha)

    import analysis_functions as af
    af.pth_res = Path(args.res)
    nclus = _nclus_map(args.res, NCLUS_SPLIT)

    sample_r, _ = _load_unsplit_r(af, regs[0])
    n_keep = n_bins_le_tmax(sample_r.shape[1], WINDOW_MS, T_MAX_MS)
    times = np.linspace(0.0, T_MAX_MS, n_keep)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"res={args.res}")
    print(f"out={out_dir}")
    print(f"regions ({len(regs)}): {', '.join(regs)}")

    weighted = None
    n_cells = 0
    used = []
    grid = []
    for reg in regs:
        r_full, _ = _load_unsplit_r(af, reg)
        r = _slice_r(r_full, n_keep)
        d = _p_on_slice(r)
        d["p_mean_c"] = float(fdr.get(reg, d["p_mean"]))
        n = int(nclus.get(reg, 0))
        if not args.only_baseline_grid:
            plot_one_region(
                times, r, d, rf"{reg}  $d^{{\mathrm{{prior}},s}}$ (80 ms)",
                out_dir / f"{reg}_{TIMEFRAME}_p_mean_c_dist.svg",
            )
        print(f"  {reg}: p_c={d['p_mean_c']:.4g}  p={d['p_mean']:.4g}  "
              f"off={d['p_offset']:.4g}  gain={d['p_gain']:.4g}  nclus={n}")
        if n > 0:
            weighted = r * n if weighted is None else weighted + r * n
            n_cells += n
            used.append(reg)
        grid.append((reg, r, d))

    r_avg = weighted / n_cells
    d_avg = _p_on_slice(r_avg)
    if not args.only_baseline_grid:
        plot_combined(
            times, r_avg, d_avg,
            out_dir / f"combined_sensory_{TIMEFRAME}_p_mean_c_dist_avg.svg",
            n_reg=len(used), n_cells=n_cells,
        )
        print(
            f"combined: n_reg={len(used)} n_cells={n_cells}  "
            f"p={d_avg['p_mean']:.4g}  off={d_avg['p_offset']:.4g}  "
            f"gain={d_avg['p_gain']:.4g}"
        )
        plot_grid(grid, times, out_dir / "sensory_earlystim80_grid.png")
    plot_grid_baseline_sub(
        grid, times, out_dir / "sensory_earlystim80_grid_baseline_sub.png",
    )
    weighted_b = None
    n_cells_b = 0
    n_used_b = 0
    for reg, r, _d in grid:
        n = int(nclus.get(reg, 0))
        if n <= 0:
            continue
        r_b = _subtract_shuffle_mean(r)
        weighted_b = r_b * n if weighted_b is None else weighted_b + r_b * n
        n_cells_b += n
        n_used_b += 1
    r_avg_b = weighted_b / n_cells_b
    d_avg_b = dict(d_avg)
    plot_combined_baseline_sub(
        times, r_avg_b, d_avg_b,
        out_dir / "sensory_earlystim80_combined_baseline_sub.png",
        n_reg=n_used_b, n_cells=n_cells_b,
    )
    print(
        f"combined residual: n_reg={n_used_b} n_cells={n_cells_b}  "
        f"min={float(r_avg_b[0].min()):.4f} max={float(r_avg_b[0].max()):.4f}  "
        f"max-min={float(r_avg_b[0].max()-r_avg_b[0].min()):.4f}"
    )
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
