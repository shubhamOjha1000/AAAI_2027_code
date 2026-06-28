"""Central config for the Stage-3 question-conditioned router pilot.

All paths are relative to this folder so the pipeline is self-contained and
runs unchanged on Colab CPU after cloning the repo.

FEATURE_MODE (env-overridable) selects the router input:
  "text"        -> question embedding only            (the original pilot)
  "visual"      -> pooled image features only          (global + gaze-crop CLIP)
  "text+visual" -> concat of the two                   (Lever 2: image-aware router)
Outputs are tagged by mode so the three runs don't clobber each other.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- inputs (bundled) ----
VERDICTS = os.path.join(DATA_DIR, "partition5_compare_verdicts.jsonl")   # per-(sample,algo) correctness
RESPONSES = os.path.join(DATA_DIR, "partition5_compare_responses.jsonl")  # per-(sample,algo) n_partitions

# ---- dataset on Drive (for visual features): <qtype>/<id>.{jpg,json} ----
DATASET_DIR = os.environ.get("DATASET_DIR", "/content/drive/MyDrive/wearvqa_gaze_only")

# ---- feature mode ----
FEATURE_MODE = os.environ.get("FEATURE_MODE", "text+visual")   # text | visual | text+visual
_TAG = FEATURE_MODE.replace("+", "_")

# ---- mode-independent intermediates ----
LABELS = os.path.join(OUT_DIR, "router_labels.jsonl")
FEATURES = os.path.join(OUT_DIR, "features.npy")          # text (question) embeddings
FEAT_IDS = os.path.join(OUT_DIR, "feature_ids.json")
VISUAL_FEATURES = os.path.join(OUT_DIR, "visual_features.npy")  # pooled image embeddings
FOLDS = os.path.join(OUT_DIR, "folds.json")

# ---- mode-tagged outputs ----
OOF = os.path.join(OUT_DIR, f"oof_{_TAG}.jsonl")
OOF_LOGREG = os.path.join(OUT_DIR, f"oof_logreg_{_TAG}.npy")
ROUTER_META = os.path.join(OUT_DIR, f"router_meta_{_TAG}.json")
METRICS = os.path.join(OUT_DIR, f"metrics_{_TAG}.json")
PARETO_PNG = os.path.join(OUT_DIR, f"pareto_{_TAG}.png")

# ---- token accounting ----
SEQ = 64                       # visual tokens per partition (SmolVLM2-256M)

# ---- label definition ----
# DETAIL (positive) iff global-only is WRONG and gaze-foveated is RIGHT.
GLOBAL_ALGO = "B_global_only"
FOVEA_ALGO = "C_gaze_foveated"
FULL_ALGO = "A_full"

# ---- encoders (CPU-friendly) ----
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"   # question text -> 384-d
CLIP_ENCODER = "clip-ViT-B-32"                        # image -> 512-d (via sentence-transformers)
USE_GLOBAL_VIS = True                                 # embed the whole frame (scene gist)
USE_FOVEA_VIS = True                                  # embed the gaze-centred crop (what's fixated)
FOVEA_CROP_PCT = 25.0                                 # crop area = 25% of the image, centred on gaze

# ---- cross-validation ----
N_FOLDS = 5
SEEDS = [0, 1, 2, 3, 4]        # average OOF over seeds for stability

# ---- torch router (tiny MLP) ----
HIDDEN = 128
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 150
BATCH_SIZE = 32                # ~16 pos / 16 neg per batch via the balanced sampler
POS_RATIO = 0.5                # target positive fraction per batch (WeightedRandomSampler)
POS_WEIGHT = 1.0               # BCE pos_weight; ~1 because the sampler already balances
EARLY_STOP_PATIENCE = 25       # on inner-val PR-AUC (average precision)
INNER_VAL_FRAC = 0.2           # of the training folds, for early-stop + tau tuning

# ---- decision threshold sweep (Pareto) ----
TAU_GRID = [round(i / 100, 2) for i in range(5, 100, 5)]   # 0.05 .. 0.95
