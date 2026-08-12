"""
Build Stage A→B hybrid JSON: WEIGHTS_REL W/g/d/θ ∪ Stage-A retinal.

Output is joint21 and loadable by ``run_fit_joint.py --resume-json``.

Example (defaults = local openalyx WEIGHTS_REL + shared-stim best s89 retinal)::

  PYTHONPATH=. python scripts/build_stage_b_hybrid.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fit_joint import (
    D_JOINT,
    reconstruct_theta_joint_from_json,
    unpack_joint,
    write_stage_b_hybrid_json,
)

MODELS = Path.home() / "Downloads/ONE/openalyx.internationalbrainlab.org/models"
DEFAULT_WEIGHTS = (
    MODELS
    / "weights_run_20251125_182058"
    / "weights_2stagelocalrefine_loss0p4044_20251125-195255.json"
)
DEFAULT_RETINAL = (
    MODELS
    / "retinal_run_fr_retinal_masknone_s89"
    / "retinal_final_loss0p3712_20260811-171342.json"
)
DEFAULT_OUT = MODELS / "stage_b_hybrid_WEIGHTS_REL_retinal_s89.json"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build Stage-B hybrid joint JSON.")
    ap.add_argument("--weights-json", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--retinal-json", type=Path, default=DEFAULT_RETINAL)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    if not args.weights_json.is_file():
        raise SystemExit(f"missing weights JSON: {args.weights_json}")
    if not args.retinal_json.is_file():
        raise SystemExit(f"missing retinal JSON: {args.retinal_json}")

    out_path, payload = write_stage_b_hybrid_json(
        args.weights_json, args.retinal_json, args.out,
    )
    # Round-trip check.
    th = reconstruct_theta_joint_from_json(payload)
    assert th.size == D_JOINT
    u = unpack_joint(th)
    print(json.dumps({
        "out": str(out_path),
        "npy": str(out_path.with_suffix(".npy")),
        "layout": payload["layout"],
        "handoff": payload["handoff"],
        "source_weights": payload["source_weights"],
        "source_retinal": payload["source_retinal"],
        "recorded_weights_loss": payload["recorded_weights_loss"],
        "recorded_retinal_loss": payload["recorded_retinal_loss"],
        "W": payload["W"],
        "g": payload["g"],
        "d": payload["d"],
        "theta": payload["theta"],
        "g_s": payload["g_s"],
        "d_s": payload["d_s"],
        "retinal": payload["retinal"],
        "roundtrip_ok": bool(abs(u["alpha_w"] - payload["retinal"]["alpha_w"]) < 1e-9),
    }, indent=2))
    return out_path


if __name__ == "__main__":
    main()
