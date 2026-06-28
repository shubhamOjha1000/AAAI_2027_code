"""Step 3 — build stratified + grouped CV folds.

Stratify by label (so each fold keeps ~the same few DETAIL positives) and group
by scene_id (so near-duplicate questions on the same image never straddle
train/test -> no leakage). With the current pilot every image is its own group,
so this reduces to stratified k-fold; the grouping matters once multiple
questions share a frame.
"""
import json
import numpy as np
import config as C


def build_folds():
    with open(C.LABELS) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    with open(C.FEAT_IDS) as f:
        feat_ids = json.load(f)

    # align rows to the feature order
    by_key = {f"{r['question_type']}||{r['id']}": r for r in rows}
    rows = [by_key[k] for k in feat_ids]

    y = np.array([1 if r["label"] == "DETAIL" else 0 for r in rows])
    groups = np.array([r["scene_id"] for r in rows])
    X = np.zeros((len(rows), 1))

    try:
        from sklearn.model_selection import StratifiedGroupKFold
        skf = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=0)
        splitter = skf.split(X, y, groups)
    except Exception:
        from sklearn.model_selection import StratifiedKFold
        print("[splits] StratifiedGroupKFold unavailable -> StratifiedKFold")
        skf = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=0)
        splitter = skf.split(X, y)

    fold = np.full(len(rows), -1, dtype=int)
    for fi, (_, test_idx) in enumerate(splitter):
        fold[test_idx] = fi

    assert (fold >= 0).all(), "every sample must be assigned to a fold"
    out = {"feat_ids": feat_ids, "fold": fold.tolist()}
    with open(C.FOLDS, "w") as f:
        json.dump(out, f)

    print(f"folds -> {C.FOLDS}")
    for fi in range(C.N_FOLDS):
        m = fold == fi
        print(f"  fold {fi}: n={m.sum()}  DETAIL={int(y[m].sum())}")
    return out


if __name__ == "__main__":
    build_folds()
