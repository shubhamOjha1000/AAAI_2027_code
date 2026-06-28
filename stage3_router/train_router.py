"""Step 6 — train the router with stratified-grouped CV and collect OOF predictions.

For each seed x fold:
  - hold out the test fold; split the rest into inner-train / inner-val
  - train the MLP on inner-train with a class-balanced sampler (+ BCE pos_weight)
  - early-stop on inner-val PR-AUC (threshold-free, imbalance-robust)
  - tune the decision threshold tau on inner-val to MAXIMISE routed end-to-end
    accuracy (tie-break: fewer tokens) -> tau never sees the test fold
  - predict probabilities on the test fold (out-of-fold)

Aggregates probabilities over seeds; writes per-sample OOF probs + the mean
operating threshold. A logistic-regression variant (hidden=0) is also reported.
"""
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

import config as C
from router_model import RouterMLP
from router_dataset import make_loader


def load_feature_matrix():
    """Assemble the router input per FEATURE_MODE (text / visual / text+visual)."""
    text = np.load(C.FEATURES)
    if C.FEATURE_MODE == "text":
        return text
    vis = np.load(C.VISUAL_FEATURES)
    if C.FEATURE_MODE == "visual":
        return vis
    return np.concatenate([text, vis], axis=1)   # text+visual


def _load_aligned():
    feats = load_feature_matrix()
    folds = json.load(open(C.FOLDS))
    feat_ids = folds["feat_ids"]
    fold = np.array(folds["fold"])
    rows = {f"{r['question_type']}||{r['id']}": r
            for r in (json.loads(l) for l in open(C.LABELS) if l.strip())}
    rows = [rows[k] for k in feat_ids]
    y = np.array([1 if r["label"] == "DETAIL" else 0 for r in rows], dtype=np.int64)
    bc = np.array([r["B_correct"] for r in rows], dtype=bool)
    cc = np.array([r["C_correct"] for r in rows], dtype=bool)
    bt = np.array([r["B_tokens"] for r in rows], dtype=float)
    ct = np.array([r["C_tokens"] for r in rows], dtype=float)
    qt = [r["question_type"] for r in rows]
    return feats, y, fold, feat_ids, qt, bc, cc, bt, ct


def routed_acc_tokens(prob, tau, bc, cc, bt, ct):
    detail = prob > tau                       # route to fovea
    correct = np.where(detail, cc, bc)
    tokens = np.where(detail, ct, bt)
    return correct.mean(), tokens.mean()


def tune_tau(prob, bc, cc, bt, ct):
    best, best_tau = (-1, 1e9), 0.5
    for tau in C.TAU_GRID:
        acc, tok = routed_acc_tokens(prob, tau, bc, cc, bt, ct)
        score = (acc, -tok)                   # max accuracy, tie-break fewer tokens
        if score > best:
            best, best_tau = score, tau
    return best_tau


def train_one(Xtr, ytr, Xva, yva, hidden, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = RouterMLP(Xtr.shape[1], hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(C.POS_WEIGHT)))
    loader = make_loader(Xtr, ytr, balanced=True)
    Xva_t = torch.as_tensor(Xva, dtype=torch.float32)

    best_ap, best_state, patience = -1.0, None, 0
    for _ in range(C.EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pva = torch.sigmoid(model(Xva_t)).numpy()
        ap = average_precision_score(yva, pva) if yva.sum() > 0 else 0.0
        if ap > best_ap:
            best_ap, best_state, patience = ap, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= C.EARLY_STOP_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run(hidden):
    feats, y, fold, feat_ids, qt, bc, cc, bt, ct = _load_aligned()
    N = len(y)
    prob_sum = np.zeros(N); prob_cnt = np.zeros(N); taus = []

    for seed in C.SEEDS:
        for f in range(C.N_FOLDS):
            te = fold == f
            tv = ~te
            idx_tv = np.where(tv)[0]
            ytv = y[idx_tv]
            # inner train/val split (stratified) for early stop + tau tuning
            tr_i, va_i = train_test_split(idx_tv, test_size=C.INNER_VAL_FRAC,
                                          stratify=ytv if ytv.sum() >= C.N_FOLDS else None,
                                          random_state=seed)
            model = train_one(feats[tr_i], y[tr_i], feats[va_i], y[va_i], hidden, seed)
            with torch.no_grad():
                pva = torch.sigmoid(model(torch.as_tensor(feats[va_i], dtype=torch.float32))).numpy()
                pte = torch.sigmoid(model(torch.as_tensor(feats[te], dtype=torch.float32))).numpy()
            taus.append(tune_tau(pva, bc[va_i], cc[va_i], bt[va_i], ct[va_i]))
            prob_sum[te] += pte; prob_cnt[te] += 1

    prob = prob_sum / np.maximum(prob_cnt, 1)
    operating_tau = float(np.mean(taus))
    return feat_ids, qt, y, prob, operating_tau


def main():
    # main model (MLP) -> OOF file; LogReg reported for reference
    feat_ids, qt, y, prob, tau = run(hidden=C.HIDDEN)
    with open(C.OOF, "w") as fo:
        for k, q, yi, pi in zip(feat_ids, qt, y, prob):
            fo.write(json.dumps({"key": k, "question_type": q,
                                 "label": "DETAIL" if yi == 1 else "GIST",
                                 "prob": float(pi)}) + "\n")
    json.dump({"operating_tau": tau, "feature_mode": C.FEATURE_MODE,
               "model": f"MLP(hidden={C.HIDDEN})", "encoder": C.ENCODER,
               "seeds": C.SEEDS, "folds": C.N_FOLDS},
              open(C.ROUTER_META, "w"), indent=2)
    print(f"[{C.FEATURE_MODE}] OOF probs -> {C.OOF}   operating_tau={tau:.3f}")

    # reference: logistic-regression router (simplest head)
    _, _, _, prob_lr, tau_lr = run(hidden=0)
    np.save(C.OOF_LOGREG, prob_lr)
    print(f"[{C.FEATURE_MODE}] LogReg OOF probs saved   operating_tau={tau_lr:.3f}")


if __name__ == "__main__":
    main()
