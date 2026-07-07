"""Selection baselines that share FRM's interface: score every candidate cell,
then KEEP_g = top-n. All operate over the SAME candidate set (fovea excluded).

  eccentricity : the geometric rival — keep cells NEAREST the gaze cell
                 (score = -distance). This is the hypothesis FRM must beat:
                 "just keep what's spatially close" vs "keep what's relevant".
  random       : the floor.
  full_g       : the ceiling — keep all candidates (no selection).
"""
import numpy as np

import config as C
from geometry import candidate_mask, cell_distance_to_gaze


def eccentricity_scores(g_idx, ring=C.FOVEA_RING):
    """Higher = keep. Nearest-to-gaze cells score highest; fovea cells -> -inf."""
    m = candidate_mask(g_idx, ring)
    d = cell_distance_to_gaze(g_idx)
    s = -d
    s[~m] = -np.inf
    return s


def random_scores(g_idx, ring=C.FOVEA_RING, rng=None):
    rng = rng or np.random.default_rng()
    m = candidate_mask(g_idx, ring)
    s = rng.random(C.K)
    s[~m] = -np.inf
    return s


def full_g_scores(g_idx, ring=C.FOVEA_RING):
    """Ceiling: every candidate kept (all equal, high)."""
    m = candidate_mask(g_idx, ring)
    s = np.zeros(C.K)
    s[~m] = -np.inf
    return s


def topn_keep(scores, n):
    """Indices of the top-n cells by score (drops -inf/fovea)."""
    valid = np.where(np.isfinite(scores))[0]
    if len(valid) <= n:
        return valid.tolist()
    order = valid[np.argsort(-scores[valid])]
    return order[:n].tolist()
