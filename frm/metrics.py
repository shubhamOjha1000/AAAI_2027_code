"""Shared metrics for the 4 experiments.

  * ranking      : Spearman(student r, teacher imp) over candidate cells.
  * answer-pres. : does the frozen VLM still answer given only KEEP_g tokens?
                   (needs the VLM — implemented in vlm_teacher.answer_preservation)
  * stratify     : split samples into NEAR / FAR context by importance-centroid
                   distance from gaze (the FAR subset is where FRM must win).
"""
import numpy as np
from scipy.stats import spearmanr

import config as C
from geometry import importance_centroid_distance


def spearman_over_cand(student_scores, teacher_imp, cand_mask):
    """Spearman rank corr on candidate cells only. Returns nan if degenerate."""
    idx = np.where(cand_mask)[0]
    s = np.asarray(student_scores)[idx]
    t = np.asarray(teacher_imp)[idx]
    if len(idx) < 3 or np.allclose(s, s[0]) or np.allclose(t, t[0]):
        return float("nan")
    rho, _ = spearmanr(s, t)
    return float(rho)


def far_mask(metas, imp_key="imp_answer", thresh=C.FAR_THRESH):
    """Boolean array over samples: True where answer-context is FAR from gaze.

    metas: list of dicts each with g_idx and imp arrays; imp_key selects which
    teacher label defines "context" (imp_answer by default).
    """
    out = []
    for m in metas:
        cm = np.asarray(m["cand_mask"], dtype=bool)
        dist = importance_centroid_distance(m[imp_key], m["g_idx"], cm)
        out.append(dist >= thresh)
    return np.array(out, dtype=bool)


def topn_recall(student_scores, teacher_imp, cand_mask, n=C.KEEP_N):
    """Fraction of the teacher's top-n candidate cells recovered by the student's top-n.

    A cheap, VLM-free proxy for "did we keep the right context".
    """
    idx = np.where(cand_mask)[0]
    if len(idx) == 0:
        return float("nan")
    n = min(n, len(idx))
    t_top = set(idx[np.argsort(-np.asarray(teacher_imp)[idx])[:n]])
    s_top = set(idx[np.argsort(-np.asarray(student_scores)[idx])[:n]])
    return len(t_top & s_top) / n
