"""Resume act-prior RT plots for unfinished full im150stim seeds."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _tmp_perf_rt_model_vs_data import BEHAVIOR_ACT, plot_one_json  # noqa: E402

BASE = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models"
)
OUT = BASE / "stageB_hold_s89_full_s_prior_1e12_plot_summary.json"
SEEDS = (101, 303, 333)
PREFIX = "weights_run_fj_stageB_hold_s89_full_im150stim_full_masknone"
TS = re.compile(r"(\d{8}-\d{6})")


def newest_final(d: Path) -> Path:
    finals = list(d.glob("weights_final_*.json"))
    if not finals:
        raise FileNotFoundError(d)

    def key(p):
        m = TS.search(p.name)
        return (m.group(1) if m else "", p.stat().st_mtime)

    return max(finals, key=key)


def main():
    behavior = np.load(BEHAVIOR_ACT, allow_pickle=True).item()
    print(f"loaded {BEHAVIOR_ACT}", flush=True)
    rows = []
    for seed in SEEDS:
        run = BASE / f"{PREFIX}_s{seed}"
        print(f"\n[rt] im150stim s{seed}", flush=True)
        r = plot_one_json(f"im150stim_s{seed}", newest_final(run), run, behavior)
        r["arm"] = "im150stim"
        r["seed"] = seed
        rows.append(r)
        print(
            f"  s{seed} perf {r['perf_r2']:.3f} RT {r['rt_r2']:.3f} "
            f"split {r['rt_split_r2']:.3f} "
            f"({r['rt_split_r2_con']:.3f}/{r['rt_split_r2_inc']:.3f})",
            flush=True,
        )

    done = {
        7: dict(
            arm="im150stim", seed=7, perf_r2=0.449, rt_r2=-5.734,
            rt_split_r2=-9.192, rt_split_r2_con=-4.206, rt_split_r2_inc=-14.527,
        ),
        12: dict(
            arm="im150stim", seed=12, perf_r2=0.645, rt_r2=-4.591,
            rt_split_r2=-8.367, rt_split_r2_con=-2.912, rt_split_r2_inc=-14.204,
        ),
        34: dict(
            arm="im150stim", seed=34, perf_r2=0.347, rt_r2=-10.418,
            rt_split_r2=-20.629, rt_split_r2_con=-8.359, rt_split_r2_inc=-33.759,
        ),
        45: dict(
            arm="im150stim", seed=45, perf_r2=0.816, rt_r2=-0.497,
            rt_split_r2=-1.086, rt_split_r2_con=-0.188, rt_split_r2_inc=-2.046,
        ),
        89: dict(
            arm="im150stim", seed=89, perf_r2=0.644, rt_r2=-7.417,
            rt_split_r2=-11.895, rt_split_r2_con=-5.501, rt_split_r2_inc=-18.738,
        ),
    }
    for r in rows:
        done[int(r["seed"])] = r
    rt_im = [done[s] for s in (7, 12, 34, 45, 89, 101, 303, 333)]
    if OUT.is_file():
        prev = json.loads(OUT.read_text())
        prev["rt"] = [
            r for r in prev.get("rt", []) if r.get("arm") != "im150stim"
        ] + rt_im
        OUT.write_text(json.dumps(prev, indent=2, default=str))
        print(f"\nmerged im150stim RT into {OUT}")
    else:
        OUT.write_text(json.dumps({"traj": [], "rt": rt_im}, indent=2, default=str))
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
