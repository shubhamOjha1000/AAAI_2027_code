"""Exp 7 — FRM capacity / architecture sweep.

How much capacity does FRM need? Sweep CAPACITY_GRID = (n_heads, depth):
  (1,1) minimal single-head  ->  (8,1) multi-head  ->  (8,2) 2-layer stacked.

Metric: OOF Spearman(student, teacher) + top-n recall + TRAIN/VAL KL gap
(overfit check). Pick the SMALLEST architecture that does not underfit.

VLM-free (ranking only) -> cheap. Runs on CPU.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np

import config as C
from labels import load_labels
from splits import make_folds
from train_frm import train_one, predict
from frm_model import FRM, masked_kl, param_count
from metrics import spearman_over_cand, topn_recall

import torch


def _val_kl(model, metas, G, rows, device):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for r in rows:
            m = metas[r]
            Gb = torch.tensor(np.asarray(G[m["row"]], np.float32), device=device).unsqueeze(0)
            gaze = Gb[:, m["g_idx"], :]
            cand = torch.tensor(m["cand_mask"], dtype=torch.bool, device=device).unsqueeze(0)
            p = torch.tensor(np.asarray(m["imp_answer"], np.float32), device=device).unsqueeze(0)
            s = model(gaze, Gb, cand)
            tot += float(masked_kl(p, s, cand))
    return tot / max(1, len(rows))


def run(seeds=(0, 1, 2), log=True):
    metas, G, _ = load_labels()
    d = metas[0]["d"]
    folds = make_folds(metas)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []
    for (nh, dep) in C.CAPACITY_GRID:
        sp, rc, tr_kl, va_kl = [], [], [], []
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            for tr, va in folds:
                model = train_one(metas, G, tr, d, n_heads=nh, depth=dep)
                s_va = predict(model, metas, G, va)
                for j, r in enumerate(va):
                    m = metas[r]
                    sp.append(spearman_over_cand(s_va[j], m["imp_answer"], m["cand_mask"]))
                    rc.append(topn_recall(s_va[j], m["imp_answer"], m["cand_mask"]))
                tr_kl.append(_val_kl(model, metas, G, tr, device))
                va_kl.append(_val_kl(model, metas, G, va, device))
        n_params = param_count(FRM(d, C.D_H, nh, dep))
        row = {
            "n_heads": nh, "depth": dep, "params": int(n_params),
            "spearman": float(np.nanmean(sp)), "topn_recall": float(np.nanmean(rc)),
            "train_kl": float(np.mean(tr_kl)), "val_kl": float(np.mean(va_kl)),
            "overfit_gap": float(np.mean(va_kl) - np.mean(tr_kl)),
        }
        results.append(row)
        if log:
            print(json.dumps(row))

    json.dump(results, open(f"{C.RESULTS_DIR}/exp7.json", "w"), indent=2)
    best = _pick_smallest(results)
    print("\nRecommended (smallest not-underfitting):", json.dumps(best))
    return results


def _pick_smallest(results):
    """Smallest params within 1% Spearman of the best."""
    best_sp = max(r["spearman"] for r in results)
    ok = [r for r in results if r["spearman"] >= best_sp - 0.01 * abs(best_sp)]
    return min(ok, key=lambda r: r["params"])


if __name__ == "__main__":
    run()
