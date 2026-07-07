"""Phase 1 — offline label generation (one-time, expensive, Colab GPU).

For every gazed WearVQA sample:
  G          = frozen encoder global tokens            [K, d]   -> G.fp16.dat memmap
  imp_answer = attention rollout over answer positions [K]      -> imp.npz
  imp_question (control), imp_context = relu(ans-question)      -> imp.npz
  (optionally) imp_loo on a small subset to validate rollout

Cache dominates total cost. Resumable: skips samples already in meta.jsonl.
"""
import json
import os

import numpy as np
from tqdm import tqdm

import config as C
from data import load_samples, load_image
from geometry import candidate_mask


def _done_keys(meta_path):
    if not os.path.exists(meta_path):
        return set()
    return {json.loads(l)["key"] for l in open(meta_path) if l.strip()}


def generate(limit=None, loo_validate=C.LOO_VALIDATE_N):
    from vlm_teacher import VLMTeacher

    samples = load_samples()
    if limit:
        samples = samples[:limit]
    done = _done_keys(C.META_JSONL)
    todo = [s for s in samples if s["key"] not in done]
    print(f"total={len(samples)} done={len(done)} todo={len(todo)}")

    teacher = VLMTeacher()
    N = len(samples)
    # memmap for G: [N, K, d]. d discovered from the model.
    d = teacher.d
    G_mm = np.memmap(C.G_MEMMAP, dtype=np.float16, mode="r+" if os.path.exists(C.G_MEMMAP) else "w+",
                     shape=(N, C.K, d))
    key_to_row = {s["key"]: i for i, s in enumerate(samples)}

    imp_a = _load_or_zeros("imp_answer", N)
    imp_q = _load_or_zeros("imp_question", N)
    imp_c = _load_or_zeros("imp_context", N)
    imp_l = _load_or_zeros("imp_loo", N)

    meta_f = open(C.META_JSONL, "a")
    for si, s in enumerate(tqdm(todo, desc="label-gen")):
        row = key_to_row[s["key"]]
        img = load_image(s["image_path"])
        bundle = teacher.build_bundle(img, s["question"], s["answer"])
        a, q = teacher.rollout_importance(bundle)
        c = np.maximum(0.0, a - q)

        G_mm[row] = bundle["G"].astype(np.float16)
        imp_a[row], imp_q[row], imp_c[row] = a, q, c

        do_loo = si < loo_validate
        if do_loo:
            imp_l[row] = teacher.loo_importance(bundle, bundle["G"])

        cm = candidate_mask(s["g_idx"]).tolist()
        meta_f.write(json.dumps({
            "key": s["key"], "id": s["id"], "row": row,
            "question_type": s["question_type"], "g_idx": s["g_idx"],
            "x_norm": s["x_norm"], "y_norm": s["y_norm"],
            "question": s["question"], "answer": s["answer"],
            "cand_mask": cm, "has_loo": bool(do_loo), "d": d,
        }) + "\n")
        meta_f.flush()
        if si % 25 == 0:
            _save_imp(imp_a, imp_q, imp_c, imp_l)
    G_mm.flush()
    _save_imp(imp_a, imp_q, imp_c, imp_l)
    meta_f.close()
    print("done. labels at", C.LABELS_DIR)


def _load_or_zeros(name, N):
    if os.path.exists(C.IMP_NPZ):
        z = np.load(C.IMP_NPZ)
        if name in z and z[name].shape[0] == N:
            return z[name].copy()
    return np.zeros((N, C.K), np.float32)


def _save_imp(a, q, c, l):
    np.savez(C.IMP_NPZ, imp_answer=a, imp_question=q, imp_context=c, imp_loo=l)


# ---- convenience loader used by every experiment ----
def load_labels():
    """Return (metas, G_memmap, imp_dict). Aligns rows via meta['row']."""
    metas = [json.loads(l) for l in open(C.META_JSONL) if l.strip()]
    N = max(m["row"] for m in metas) + 1
    d = metas[0]["d"]
    G = np.memmap(C.G_MEMMAP, dtype=np.float16, mode="r", shape=(N, C.K, d))
    imp = dict(np.load(C.IMP_NPZ))
    for m in metas:
        import numpy as _np
        m["cand_mask"] = _np.asarray(m["cand_mask"], dtype=bool)
        for k in ("imp_answer", "imp_question", "imp_context", "imp_loo"):
            m[k] = imp[k][m["row"]]
    return metas, G, imp


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    generate(**vars(ap.parse_args()))
