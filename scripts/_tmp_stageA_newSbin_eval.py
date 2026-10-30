"""Stage A newSbin vs newS: multi-stim-seed eval under the current S loss (2026-10-29).

Per final: 5 stim seeds (bps=20), per-contrast SSE / SNR / R², c=1 bump.
Writes multiseed.json + overlay figs into the newSbin ranking folder.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_retinal_fit_s as P  # noqa: E402
import model_functions as mf  # noqa: E402

M = Path.home() / "Downloads/ONE/alyx.internationalbrainlab.org/models"
OUT = M / "retinal_s_fit_plots_retinal_newSbin_bps20_seed12345" / "multiseed"
SEEDS = [12345, 1, 2, 3, 4]


def run(jp, seed):
    mp, meta, ret = P.mp_from_retinal_json(jp)
    sbo = int(mf.STEPS_BEFORE_OBS_DURATION_MS / mp["dt"])
    mo = int(mf.MAX_OBS_DURATION_MS / mp["dt"])
    st, ts, _, tsd, bsd = mf.create_stimuli(
        20, mf.trials_per_block_param, mf.block_side_probs, mf.num_stimulus_strength,
        mf.min_stimulus_strength, mf.max_stimulus_strength, mf.min_trials_per_block,
        mf.max_trials_per_block, mo, sbo, rng=np.random.RandomState(seed), **mp)
    res = mf.run_model("data", st, ts, tsd, bsd, 20, steps_before_obs=sbo,
                       verbose=False, backend="numba", **mp)
    return mf.mean_S_by_contrast(res, sbo)


def main():
    _, R = P.load_avg_mean_r("fit_targets_integ/opt2_shuffleslope/avg_mean_R.npy")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag in ("newSbin", "newS"):
        for jp in sorted(M.glob(f"retinal_run_fr_retinal_{tag}_retinal_masknone_s*/retinal_final_*.json")):
            seed_tag = jp.parent.name.split("_masknone_")[1]
            per_seed = []
            for s in SEEDS:
                S = run(jp, s)
                L = mf.compute_sse_stim_right(S, R, baseline_R=0)
                m1 = S[(1, 1.0)][1] - S[(1, 1.0)][0]
                per_seed.append({
                    "seed": s, "L": L["total_loss"], "sse": L["total_sse"],
                    "snr": L["total_snr_loss"], "r2": L["total_gof_r2"],
                    "c1_peak": float(np.max(m1[:40])), "c1_peak_step": int(np.argmax(m1[:40])),
                    "per_c": {str(c): {k: v[k] for k in ("sse", "snr_loss", "snr_model", "snr_data", "gof_r2")}
                              for c, v in L["per_contrast"].items()}})
                if s == SEEDS[0]:
                    (OUT / f"{tag}_{seed_tag}").mkdir(parents=True, exist_ok=True)
                    mf.plot_S_diff_by_contrast_side_with_data(
                        S, {}, R, baseline=0, save_dir=str(OUT / f"{tag}_{seed_tag}"),
                        ylim=[-0.1, 0.7], yticks=None, r2=L["total_gof_r2"])
                    import matplotlib.pyplot as plt
                    plt.close("all")
            row = {"tag": tag, "seed": seed_tag, "json": str(jp),
                   "recorded": json.loads(jp.read_text()).get("loss"), "per_seed": per_seed}
            for k in ("L", "sse", "snr", "r2", "c1_peak"):
                v = np.array([p[k] for p in per_seed])
                row[k + "_mean"], row[k + "_sd"] = float(v.mean()), float(v.std())
            rows.append(row)
            print(f"{tag:8s} {seed_tag:5s} rec {row['recorded']:.3f}  L {row['L_mean']:.3f}±{row['L_sd']:.3f}"
                  f"  sse {row['sse_mean']:.3f}±{row['sse_sd']:.3f}  snr {row['snr_mean']:.3f}±{row['snr_sd']:.3f}"
                  f"  R2 {row['r2_mean']:.3f}±{row['r2_sd']:.3f}  c1pk {row['c1_peak_mean']:.4f}", flush=True)
    d1 = np.asarray(R[1.0], float)
    print("data c1 peak (first 40 bins)", float(d1[:40].max()), int(d1[:40].argmax()))
    (OUT / "multiseed.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
