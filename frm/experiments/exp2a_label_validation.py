"""Exp 2a — label validation: answer- vs question-grounding (cheap, NO training).

Empirical proof of "relevance != proximity". Uses only the cached teacher labels.

  * imp_question grounds NEAR gaze (deictic words point at the fixated thing).
  * imp_answer reaches FAR (the context the answer actually needed).

We measure, over candidate cells, how importance mass distributes with distance
from the gaze cell. If the answer curve carries more far-distance mass than the
question curve, the two DIVERGE -> FRM (which predicts answer-context) is needed.

Run this FIRST. No VLM forward here (labels already cached).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from scipy.stats import spearmanr, wilcoxon

import config as C
from labels import load_labels
from geometry import cell_distance_to_gaze, importance_centroid_distance


def _weighted_mean_dist(imp, g_idx, cm):
    imp = np.asarray(imp, float) * cm
    d = cell_distance_to_gaze(g_idx)
    return (imp * d).sum() / imp.sum() if imp.sum() > 0 else 0.0


def run(save_plot=True):
    metas, G, _ = load_labels()
    maxd = np.sqrt(2) * (C.GRID - 1)
    bins = np.linspace(0, maxd, 9)
    ans_prof = np.zeros(len(bins) - 1)
    q_prof = np.zeros(len(bins) - 1)
    a_wm, q_wm = [], []
    pooled_d, pooled_a, pooled_q = [], [], []

    for m in metas:
        cm = m["cand_mask"]
        d = cell_distance_to_gaze(m["g_idx"])[cm]
        a = np.asarray(m["imp_answer"], float)[cm]
        q = np.asarray(m["imp_question"], float)[cm]
        a = a / a.sum() if a.sum() > 0 else a
        q = q / q.sum() if q.sum() > 0 else q
        # per-distance-bin importance mass (averaged over samples)
        idx = np.clip(np.digitize(d, bins) - 1, 0, len(bins) - 2)
        for b in range(len(bins) - 1):
            ans_prof[b] += a[idx == b].sum()
            q_prof[b] += q[idx == b].sum()
        a_wm.append(_weighted_mean_dist(m["imp_answer"], m["g_idx"], cm))
        q_wm.append(_weighted_mean_dist(m["imp_question"], m["g_idx"], cm))
        pooled_d += d.tolist(); pooled_a += a.tolist(); pooled_q += q.tolist()

    ans_prof /= len(metas); q_prof /= len(metas)
    a_wm, q_wm = np.array(a_wm), np.array(q_wm)
    rho_a = spearmanr(pooled_d, pooled_a).correlation
    rho_q = spearmanr(pooled_d, pooled_q).correlation
    try:
        w_p = float(wilcoxon(a_wm, q_wm).pvalue)
    except Exception:
        w_p = float("nan")

    summary = {
        "n": len(metas),
        "answer_weighted_mean_dist": float(a_wm.mean()),
        "question_weighted_mean_dist": float(q_wm.mean()),
        "divergence(answer-question)": float(a_wm.mean() - q_wm.mean()),
        "spearman_dist_vs_imp_answer": float(rho_a),
        "spearman_dist_vs_imp_question": float(rho_q),
        "wilcoxon_p(answer>question dist)": w_p,
        "interpretation": ("answer reaches farther than question -> proof FRM needed"
                           if a_wm.mean() > q_wm.mean()
                           else "answer NOT farther than question -> weak/absent signal "
                                "(expected for question-derived WearVQA gaze)"),
    }
    print(json.dumps(summary, indent=2))
    json.dump(summary, open(f"{C.RESULTS_DIR}/exp2a.json", "w"), indent=2)

    if save_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        centers = 0.5 * (bins[:-1] + bins[1:])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(centers, ans_prof, "-o", color="#4C78A8", label="imp_answer (far context)")
        ax.plot(centers, q_prof, "-s", color="#E45756", label="imp_question (grounds near)")
        ax.set_xlabel("distance from gaze (grid cells)")
        ax.set_ylabel("mean importance mass per bin")
        ax.set_title("Exp 2a: answer- vs question-grounding by distance")
        ax.legend(); fig.tight_layout()
        fig.savefig(f"{C.RESULTS_DIR}/exp2a_divergence.png", dpi=140)
        print("saved exp2a_divergence.png")
    return summary


if __name__ == "__main__":
    run()
