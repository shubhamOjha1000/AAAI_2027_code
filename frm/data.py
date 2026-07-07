"""WearVQA gaze-only dataset loader.

Layout: DATA_DIR/<question_type>/<id>.{jpg,json}
JSON schema (relevant keys):
  question : str
  response : str          # the gold answer (full sentence)
  gaze     : {x_norm, y_norm, ...}

Yields lightweight sample dicts. Image bytes are loaded lazily (only Phase-1
label generation and answer-preservation actually open the JPEGs).
"""
import json
import os

import config as C
from geometry import gaze_to_cell


def iter_samples(data_dir=C.DATA_DIR):
    """Iterate every gazed sample as a dict (no image loaded yet)."""
    for qtype in sorted(os.listdir(data_dir)):
        qd = os.path.join(data_dir, qtype)
        if not os.path.isdir(qd):
            continue
        for fn in sorted(f for f in os.listdir(qd) if f.endswith(".json")):
            sid = fn[:-5]
            d = json.load(open(os.path.join(qd, fn)))
            g = d.get("gaze")
            if not g:
                continue
            gx, gy, g_idx = gaze_to_cell(g["x_norm"], g["y_norm"])
            yield {
                "id": sid,
                "key": f"{qtype}||{sid}",
                "question_type": qtype,
                "image_path": os.path.join(qd, sid + ".jpg"),
                "question": d.get("question", ""),
                "answer": d.get("response", ""),
                "x_norm": g["x_norm"],
                "y_norm": g["y_norm"],
                "g_idx": int(g_idx),
                "gx": int(gx),
                "gy": int(gy),
            }


def load_samples(data_dir=C.DATA_DIR):
    return list(iter_samples(data_dir))


def load_image(path):
    from PIL import Image
    return Image.open(path).convert("RGB")
