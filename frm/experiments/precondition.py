"""PRECONDITION (run before anything): gaze -> answer-context distance distribution.

If almost no samples have answer-context FAR from gaze, Exp 1 / 2a will look flat
because the DATA lacks signal, not because FRM is bad. This measures whether the
premise ("relevance != proximity") is even present in WearVQA gaze.

NOTE: our gaze is question-derived (placed at the answer cue), so we EXPECT most
mass to be near gaze -> this quantifies exactly how weak the far-context signal is.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np

import config as C
from labels import load_labels
from geometry import importance_centroid_distance


def run(save_plot=True):
    metas, G, _ = load_labels()
    dists = np.array([
        importance_centroid_distance(m["imp_answer"], m["g_idx"], m["cand_mask"])
        for m in metas
    ])
    far_frac = float((dists >= C.FAR_THRESH).mean())
    summary = {
        "n": len(dists),
        "far_thresh_cells": C.FAR_THRESH,
        "far_fraction": far_frac,
        "dist_mean": float(dists.mean()),
        "dist_median": float(np.median(dists)),
        "dist_p90": float(np.percentile(dists, 90)),
    }
    # per question type
    per = {}
    for qt in sorted(set(m["question_type"] for m in metas)):
        idx = [i for i, m in enumerate(metas) if m["question_type"] == qt]
        per[qt] = {"n": len(idx), "far_fraction": float((dists[idx] >= C.FAR_THRESH).mean()),
                   "dist_mean": float(dists[idx].mean())}
    summary["per_question_type"] = per
    print(json.dumps(summary, indent=2))
    json.dump(summary, open(f"{C.RESULTS_DIR}/precondition.json", "w"), indent=2)

    if save_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(dists, bins=24, color="#4C78A8", alpha=0.85)
        ax.axvline(C.FAR_THRESH, color="#E45756", ls="--",
                   label=f"FAR thresh = {C.FAR_THRESH} (far={far_frac:.1%})")
        ax.set_xlabel("answer-context centroid distance from gaze (grid cells)")
        ax.set_ylabel("# samples")
        ax.set_title("Precondition: gaze->context distance distribution")
        ax.legend(); fig.tight_layout()
        fig.savefig(f"{C.RESULTS_DIR}/precondition_hist.png", dpi=140)
        print("saved precondition_hist.png")
    return summary


if __name__ == "__main__":
    run()
