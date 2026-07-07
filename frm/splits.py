"""Cross-validation splits. Group = image id (scene) so no image leaks across a
fold boundary; stratify on question_type to keep type balance in each fold.
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

import config as C


def make_folds(metas, n_folds=C.N_FOLDS, seed=0):
    y = np.array([m["question_type"] for m in metas])
    groups = np.array([m["id"] for m in metas])          # one image = one group
    X = np.zeros(len(metas))
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for tr, va in sgkf.split(X, y, groups):
        folds.append((tr.tolist(), va.tolist()))
    return folds
