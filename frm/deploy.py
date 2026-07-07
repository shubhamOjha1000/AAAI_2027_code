"""Phase 3 — deploy (question-free, cached).

    r      = FRM(gaze_tok, G)         # [K]  no question needed
    KEEP_g = top_n(r, n=KEEP_N)       # relevant context cells
    MERGE_g = rest                    # -> Stage 2c eccentricity merge (stand-in)

The fovea region (Stage 2a) is always available at high res, so the effective
kept set fed downstream is fovea ∪ KEEP_g.
"""
import numpy as np
import torch

import config as C
from geometry import candidate_mask, fovea_region


@torch.no_grad()
def frm_relevance(model, G_row, g_idx, device=None):
    """G_row [K,d] numpy -> relevance r [K] (fovea cells = 0)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    Gb = torch.tensor(np.asarray(G_row, np.float32), device=device).unsqueeze(0)
    gaze = Gb[:, g_idx, :]
    cand = torch.tensor(candidate_mask(g_idx), dtype=torch.bool, device=device).unsqueeze(0)
    r = model.relevance(gaze, Gb, cand)[0].cpu().numpy()
    return r


def keep_from_scores(scores, g_idx, n=C.KEEP_N, include_fovea=True):
    """Top-n candidate cells by score, optionally unioned with the fovea region."""
    cm = candidate_mask(g_idx)
    valid = np.where(cm)[0]
    n = min(n, len(valid))
    keep = valid[np.argsort(-np.asarray(scores)[valid])[:n]].tolist()
    if include_fovea:
        keep = sorted(set(keep) | set(fovea_region(g_idx)))
    return keep
