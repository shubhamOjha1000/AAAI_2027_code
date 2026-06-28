"""Step 2 — encode each question into a fixed feature vector (run once).

Because the router input is just the question text and the backbone is frozen,
we precompute all embeddings once and cache them; router training then runs in
seconds on CPU over these cached vectors.

Encoder: a small sentence-transformer (CPU-friendly). Swap C.ENCODER to use the
VLM's own text embedder for deployment-consistency in the scaled study.
"""
import json
import numpy as np
import config as C


def main():
    with open(C.LABELS) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    # fixed, reproducible order
    ids = [f"{r['question_type']}||{r['id']}" for r in rows]
    questions = [r["question"] for r in rows]

    from sentence_transformers import SentenceTransformer
    print(f"loading encoder: {C.ENCODER}")
    model = SentenceTransformer(C.ENCODER, device="cpu")
    feats = model.encode(questions, batch_size=64, convert_to_numpy=True,
                         show_progress_bar=True, normalize_embeddings=True)
    feats = np.asarray(feats, dtype=np.float32)

    np.save(C.FEATURES, feats)
    with open(C.FEAT_IDS, "w") as f:
        json.dump(ids, f)
    print(f"features {feats.shape} -> {C.FEATURES}")
    print(f"ids ({len(ids)}) -> {C.FEAT_IDS}")


if __name__ == "__main__":
    main()
