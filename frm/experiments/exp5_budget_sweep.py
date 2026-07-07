"""Exp 5 — selection budget sweep (n for KEEP_g).

How many context tokens to keep? Sweep n in BUDGETS; report the accuracy-vs-tokens
curve (find the knee). Tokens = kept global cells (fovea ∪ top-n). Compares FRM
against eccentricity at every budget.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np

import config as C
from labels import load_labels
from splits import make_folds
from train_frm import oof_scores
from data import load_samples
from metrics import far_mask
from geometry import fovea_region


def run(preserve_limit=120, seeds=C.SEEDS, log=True):
    metas, G, _ = load_labels()
    d = metas[0]["d"]
    # reuse Exp 1 OOF scores if present, else train
    cache = f"{C.RESULTS_DIR}/exp1_oof.npy"
    if os.path.exists(cache):
        frm = np.load(cache)
    else:
        frm = oof_scores(metas, G, d, make_folds(metas), seeds=seeds, log=log)

    far = far_mask(metas)
    rng = np.random.default_rng(0)
    rows = rng.choice(np.where(far)[0], size=min(preserve_limit, int(far.sum())),
                      replace=False).tolist()

    from vlm_teacher import VLMTeacher
    from preserve_eval import evaluate
    teacher = VLMTeacher()
    samples = load_samples()

    curve = {"budgets": C.BUDGETS, "frm": [], "eccentricity": [], "tokens": []}
    for n in C.BUDGETS:
        res = evaluate(teacher, metas, frm, rows, samples, n=n,
                       methods=["frm", "eccentricity"])
        # mean kept-token count across the subset (fovea ∪ n)
        tok = np.mean([len(set(fovea_region(metas[r]["g_idx"])) |
                           set(range(n))) for r in rows])  # upper bound; report n+|fovea|
        curve["frm"].append(res["methods"]["frm"]["answer_kept_acc"])
        curve["eccentricity"].append(res["methods"]["eccentricity"]["answer_kept_acc"])
        curve["tokens"].append(float(tok))
        print(f"n={n}: frm={curve['frm'][-1]:.3f} ecc={curve['eccentricity'][-1]:.3f}")

    json.dump(curve, open(f"{C.RESULTS_DIR}/exp5.json", "w"), indent=2)
    _plot(curve)
    return curve


def _plot(curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["budgets"], curve["frm"], "-o", color="#4C78A8", label="FRM")
    ax.plot(curve["budgets"], curve["eccentricity"], "-s", color="#E45756", label="Eccentricity")
    ax.set_xlabel("KEEP_g budget n (context cells)")
    ax.set_ylabel("answer-kept accuracy (far subset)")
    ax.set_title("Exp 5: accuracy vs budget")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"{C.RESULTS_DIR}/exp5_budget.png", dpi=140)
    print("saved exp5_budget.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preserve_limit", type=int, default=120)
    run(**vars(ap.parse_args()))
