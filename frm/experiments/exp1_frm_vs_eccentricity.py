"""Exp 1 — FRM vs Eccentricity (THE decisive / make-or-break).

Does the LEARNED FRM beat the cheap geometric baseline, ESPECIALLY on far context?

Cheap proxies (all samples, no VLM): Spearman(FRM, teacher), top-n recall.
Decisive metric (VLM, GPU): answer-preservation @ n=KEEP_N, stratified NEAR/FAR.

Success/kill: FRM > Eccentricity on the FAR subset -> FRM justified.
              FRM ~ Eccentricity -> drop FRM, fall back to eccentricity.
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
from metrics import spearman_over_cand, topn_recall, far_mask
from baselines import eccentricity_scores, random_scores


def cheap_proxies(metas, frm_scores):
    """VLM-free ranking proxies, overall and on the FAR subset."""
    far = far_mask(metas)
    rows_all = np.arange(len(metas))
    def block(rows):
        sp_f, sp_e, rc_f, rc_e = [], [], [], []
        for r in rows:
            m = metas[r]
            t = m["imp_answer"]; cm = m["cand_mask"]
            ecc = eccentricity_scores(m["g_idx"])
            sp_f.append(spearman_over_cand(frm_scores[m["row"]], t, cm))
            sp_e.append(spearman_over_cand(ecc, t, cm))
            rc_f.append(topn_recall(frm_scores[m["row"]], t, cm))
            rc_e.append(topn_recall(ecc, t, cm))
        f = lambda x: float(np.nanmean(x))
        return {"frm_spearman": f(sp_f), "ecc_spearman": f(sp_e),
                "frm_topn_recall": f(rc_f), "ecc_topn_recall": f(rc_e)}
    return {"overall": block(rows_all), "far_subset": block(np.where(far)[0]),
            "near_subset": block(np.where(~far)[0]), "far_count": int(far.sum())}


def run(preserve_limit=120, seeds=C.SEEDS, log=True):
    metas, G, _ = load_labels()
    d = metas[0]["d"]
    folds = make_folds(metas)
    print("training FRM (OOF)...")
    frm = oof_scores(metas, G, d, folds, seeds=seeds, label_key="imp_answer", log=log)
    np.save(f"{C.RESULTS_DIR}/exp1_oof.npy", frm)

    proxies = cheap_proxies(metas, frm)
    print(json.dumps(proxies, indent=2))

    # ---- decisive: answer-preservation on a balanced far/near subset ----
    far = far_mask(metas)
    far_rows = np.where(far)[0]
    near_rows = np.where(~far)[0]
    rng = np.random.default_rng(0)
    k = min(preserve_limit // 2, len(far_rows))
    sub_far = rng.choice(far_rows, size=k, replace=False) if k else np.array([], int)
    sub_near = rng.choice(near_rows, size=min(preserve_limit - k, len(near_rows)),
                          replace=False)
    result = {"proxies": proxies, "preserve": {}}

    try:
        from vlm_teacher import VLMTeacher
        from preserve_eval import evaluate
        teacher = VLMTeacher()
        samples = load_samples()
        for name, rows in [("far", sub_far), ("near", sub_near)]:
            if len(rows) == 0:
                continue
            result["preserve"][name] = evaluate(
                teacher, metas, frm, rows.tolist(), samples, n=C.KEEP_N)
            print(name, json.dumps(result["preserve"][name], indent=2))
    except Exception as e:
        result["preserve_error"] = f"{type(e).__name__}: {e} (needs GPU + VLM)"
        print("answer-preservation skipped:", result["preserve_error"])

    json.dump(result, open(f"{C.RESULTS_DIR}/exp1.json", "w"), indent=2)
    _verdict(result)
    return result


def _verdict(result):
    fp = result.get("preserve", {}).get("far", {}).get("methods", {})
    if "frm" in fp and "eccentricity" in fp:
        a = fp["frm"]["answer_kept_acc"]; b = fp["eccentricity"]["answer_kept_acc"]
        print(f"\nVERDICT (far subset): FRM answer-kept={a:.3f} vs Ecc={b:.3f} -> "
              + ("FRM JUSTIFIED" if a > b + 0.02 else "FRM ~ Ecc -> consider dropping"))
    else:
        pf = result["proxies"]["far_subset"]
        print(f"\nVERDICT (far proxy): FRM recall={pf['frm_topn_recall']:.3f} vs "
              f"Ecc={pf['ecc_topn_recall']:.3f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preserve_limit", type=int, default=120)
    run(**vars(ap.parse_args()))
