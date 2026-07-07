"""Phase 2 — FRM training (cheap, fast; FRM is the only thing updated).

Listwise KL distillation: teacher p = imp_answer[cand] normalized, student =
softmax(FRM logits) over the SAME candidate set (fovea excluded, Section 3.1).

Marginalization is IMPLICIT: samples that share (image, gaze) but differ in
question feed the SAME input with DIFFERENT imp_answer -> the KL drives the
student to their mean E_q[imp | gaze, G] (the anticipatory context predictor).

Returns out-of-fold student scores so eval is leakage-free. Runs fine on CPU
(the model is ~0.5-few M params) but will use GPU if present.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import config as C
from frm_model import FRM, masked_kl, param_count


class _DS(Dataset):
    def __init__(self, metas, G, rows, label_key="imp_answer"):
        self.metas, self.G, self.rows, self.k = metas, G, rows, label_key

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        m = self.metas[self.rows[i]]
        G = torch.tensor(np.asarray(self.G[m["row"]], dtype=np.float32))   # [K,d]
        gaze = G[m["g_idx"]]                                               # [d]
        cand = torch.tensor(m["cand_mask"], dtype=torch.bool)             # [K]
        p = torch.tensor(np.asarray(m[self.k], dtype=np.float32))         # [K]
        return gaze, G, cand, p


def _pos_weight_sampler(metas, rows, thresh_key="imp_answer"):
    """Balance far-context (rare, the signal) vs near-context samples per batch."""
    from geometry import importance_centroid_distance
    far = []
    for r in rows:
        m = metas[r]
        d = importance_centroid_distance(m[thresh_key], m["g_idx"], m["cand_mask"])
        far.append(d >= C.FAR_THRESH)
    far = np.array(far)
    w = np.where(far, (~far).sum() + 1, far.sum() + 1).astype(np.float64)
    w = w / w.sum()
    return WeightedRandomSampler(torch.tensor(w), num_samples=len(rows), replacement=True)


def train_one(metas, G, train_rows, d, n_heads=C.N_HEADS, depth=C.DEPTH,
              label_key="imp_answer", epochs=C.EPOCHS, device=None, log=False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = FRM(d, C.D_H, n_heads, depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    ds = _DS(metas, G, train_rows, label_key)
    sampler = _pos_weight_sampler(metas, train_rows, label_key)
    dl = DataLoader(ds, batch_size=C.BATCH_SIZE, sampler=sampler)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for gaze, Gb, cand, p in dl:
            gaze, Gb, cand, p = gaze.to(device), Gb.to(device), cand.to(device), p.to(device)
            s = model(gaze, Gb, cand)
            loss = masked_kl(p, s, cand)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        if log and ep % 25 == 0:
            print(f"  ep{ep:3d}  kl={tot/len(dl):.4f}")
    return model


@torch.no_grad()
def predict(model, metas, G, rows, device=None):
    """Student logits [len(rows), K] for the given rows."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    out = np.full((len(rows), C.K), -np.inf, np.float32)
    for i, r in enumerate(rows):
        m = metas[r]
        Gb = torch.tensor(np.asarray(G[m["row"]], np.float32), device=device).unsqueeze(0)
        gaze = Gb[:, m["g_idx"], :]
        cand = torch.tensor(m["cand_mask"], dtype=torch.bool, device=device).unsqueeze(0)
        s = model(gaze, Gb, cand)[0].cpu().numpy()
        out[i] = s
    return out


def oof_scores(metas, G, d, folds, n_heads=C.N_HEADS, depth=C.DEPTH,
               label_key="imp_answer", seeds=C.SEEDS, log=False):
    """Out-of-fold student logits averaged over seeds. [N, K]."""
    N = len(metas)
    acc = np.zeros((N, C.K), np.float64)
    cnt = np.zeros(N, np.float64)
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        for fi, (tr, va) in enumerate(folds):
            model = train_one(metas, G, tr, d, n_heads, depth, label_key, log=log)
            s = predict(model, metas, G, va)
            acc[va] += s; cnt[va] += 1
            if log:
                print(f"seed{seed} fold{fi} done")
    cnt[cnt == 0] = 1
    return (acc / cnt[:, None]).astype(np.float32)


if __name__ == "__main__":
    from labels import load_labels
    from splits import make_folds
    metas, G, _ = load_labels()
    d = metas[0]["d"]
    print("FRM params:", param_count(FRM(d)))
    folds = make_folds(metas)
    oof = oof_scores(metas, G, d, folds, log=True)
    np.save(f"{C.RESULTS_DIR}/oof_scores.npy", oof)
    print("saved OOF scores")
