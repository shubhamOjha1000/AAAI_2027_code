"""Central config for the Stage-3 question-conditioned router pilot.

All paths are relative to this folder so the pipeline is self-contained and
runs unchanged on Colab CPU after cloning the repo.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- inputs (bundled) ----
VERDICTS = os.path.join(DATA_DIR, "partition5_compare_verdicts.jsonl")   # per-(sample,algo) correctness
RESPONSES = os.path.join(DATA_DIR, "partition5_compare_responses.jsonl")  # per-(sample,algo) n_partitions

# ---- intermediate / outputs ----
LABELS = os.path.join(OUT_DIR, "router_labels.jsonl")
FEATURES = os.path.join(OUT_DIR, "features.npy")
FEAT_IDS = os.path.join(OUT_DIR, "feature_ids.json")
FOLDS = os.path.join(OUT_DIR, "folds.json")
OOF = os.path.join(OUT_DIR, "oof_predictions.jsonl")
METRICS = os.path.join(OUT_DIR, "metrics.json")
PARETO_PNG = os.path.join(OUT_DIR, "pareto.png")

# ---- token accounting ----
SEQ = 64                       # visual tokens per partition (SmolVLM2-256M)
# token cost of each fixed strategy comes from n_partitions in RESPONSES.

# ---- label definition ----
# DETAIL (positive) iff global-only is WRONG and gaze-foveated is RIGHT.
GLOBAL_ALGO = "B_global_only"
FOVEA_ALGO = "C_gaze_foveated"
FULL_ALGO = "A_full"

# ---- question encoder (CPU-friendly) ----
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"   # 384-d, ~80MB, fast on CPU

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
