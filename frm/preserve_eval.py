"""Shared answer-preservation evaluation (used by Exp 1 and Exp 5).

Given a trained-FRM score table + baseline scorers, for each sample build the
teacher bundle, keep only (fovea ∪ top-n) global tokens per method, and ask the
FROZEN VLM whether it still answers (retained logprob + generation match).

This is the expensive, GPU-bound part of evaluation -> supports a row subset.
"""
import numpy as np
from tqdm import tqdm

import config as C
from data import load_image
from deploy import keep_from_scores
from baselines import eccentricity_scores, random_scores, full_g_scores


def _key_to_sample(data_samples):
    return {s["key"]: s for s in data_samples}


def evaluate(teacher, metas, frm_scores, rows, data_samples, n=C.KEEP_N, methods=None):
    """Return per-method aggregates over `rows`.

    frm_scores : [N,K] student logits aligned to meta['row'] (use OOF).
    methods    : subset of {"frm","eccentricity","random","full_g"}.
    """
    methods = methods or ["frm", "eccentricity", "random", "full_g"]
    by_key = _key_to_sample(data_samples)
    rng = np.random.default_rng(0)
    agg = {m: {"retained": [], "gen_keep": [], "gen_full": []} for m in methods}

    for r in tqdm(rows, desc=f"preserve@{n}"):
        m = metas[r]
        s = by_key[m["key"]]
        img = load_image(s["image_path"])
        bundle = teacher.build_bundle(img, s["question"], s["answer"])
        G = bundle["G"]
        g_idx = m["g_idx"]
        score_map = {
            "frm": frm_scores[m["row"]],
            "eccentricity": eccentricity_scores(g_idx),
            "random": random_scores(g_idx, rng=rng),
            "full_g": full_g_scores(g_idx),
        }
        for meth in methods:
            keep = keep_from_scores(score_map[meth], g_idx, n=n, include_fovea=True)
            res = teacher.answer_preservation(bundle, G, keep)
            agg[meth]["retained"].append(res["retained"])
            agg[meth]["gen_keep"].append(res["gen_match_keep"])
            agg[meth]["gen_full"].append(res["gen_match_full"])
    return _summarize(agg, n)


def _summarize(agg, n):
    out = {"n": n, "methods": {}}
    for meth, d in agg.items():
        ret = np.array(d["retained"], float)
        gk = np.array(d["gen_keep"], float)
        out["methods"][meth] = {
            "n_samples": int(len(ret)),
            "mean_retained_logprob": float(ret.mean()) if len(ret) else float("nan"),
            "answer_kept_acc": float(gk.mean()) if len(gk) else float("nan"),
        }
    if agg:
        any_m = next(iter(agg))
        gf = np.array(agg[any_m]["gen_full"], float)
        out["full_G_answer_acc"] = float(gf.mean()) if len(gf) else float("nan")
    return out
