"""Step 2b (Lever 2) — pooled VISUAL features per sample (run once, CPU).

For each sample we embed two views with CLIP and concatenate:
  - GLOBAL : the whole frame            -> "what's in the scene" (gist)
  - FOVEA  : a gaze-centred crop (~25%) -> "what is being looked at"

Rationale: the question-only router failed partly because detail-need is
image-dependent (the same question is GIST on one frame, DETAIL on another).
These features let the router see whether the fixated content actually needs
high resolution. Aligned to the same order as features.npy (FEAT_IDS).

Needs the dataset on Drive at C.DATASET_DIR/<qtype>/<id>.{jpg,json}.
"""
import os
import json
import math
import numpy as np
from PIL import Image
import config as C


def _l2(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-8)


def _gaze_crop(img, x_norm, y_norm, pct):
    W, H = img.size
    side = max(1, int(math.sqrt((pct / 100.0) * W * H)))
    cx, cy = x_norm * W, y_norm * H
    x1 = int(max(0, min(cx - side / 2, W - side)))
    y1 = int(max(0, min(cy - side / 2, H - side)))
    return img.crop((x1, y1, x1 + side, y1 + side))


def main():
    with open(C.FEAT_IDS) as f:
        feat_ids = json.load(f)

    globals_, foveas = [], []
    missing = 0
    for key in feat_ids:
        qt, sid = key.split("||")
        base = os.path.join(C.DATASET_DIR, qt, sid)
        try:
            img = Image.open(base + ".jpg").convert("RGB")
            gx, gy = 0.5, 0.5
            if os.path.exists(base + ".json"):
                g = json.load(open(base + ".json")).get("gaze", {})
                gx, gy = float(g.get("x_norm", 0.5)), float(g.get("y_norm", 0.5))
        except Exception:
            missing += 1
            img = Image.new("RGB", (364, 364), (0, 0, 0))
            gx, gy = 0.5, 0.5
        globals_.append(img)
        foveas.append(_gaze_crop(img, gx, gy, C.FOVEA_CROP_PCT))
    if missing:
        print(f"[warn] {missing} images not found under {C.DATASET_DIR} (used blank fallback)")

    from sentence_transformers import SentenceTransformer
    print(f"loading CLIP: {C.CLIP_ENCODER}")
    clip = SentenceTransformer(C.CLIP_ENCODER, device="cpu")

    blocks = []
    if C.USE_GLOBAL_VIS:
        ge = clip.encode(globals_, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
        blocks.append(_l2(np.asarray(ge, dtype=np.float32)))
    if C.USE_FOVEA_VIS:
        fe = clip.encode(foveas, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
        blocks.append(_l2(np.asarray(fe, dtype=np.float32)))

    vis = np.concatenate(blocks, axis=1).astype(np.float32)
    np.save(C.VISUAL_FEATURES, vis)
    print(f"visual features {vis.shape} -> {C.VISUAL_FEATURES}  "
          f"(global={C.USE_GLOBAL_VIS}, fovea={C.USE_FOVEA_VIS})")


if __name__ == "__main__":
    main()
