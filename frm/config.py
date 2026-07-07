"""Central config for the FRM (Foveal Relevance Module) — Stage 2b.

Everything is a plain module-level constant so the pipeline is self-contained and
runs unchanged on Colab GPU after cloning the repo + mounting Drive.

Design decisions (locked with the user):
  * Dataset : WearVQA gaze_only (proof-of-concept). Gaze is question-derived, so
              we ALSO run the anti-leakage perturbation check and report it (the
              spec warns this may collapse to "gaze = answer box").
  * VLM     : SmolVLM2-2.2B-Instruct, FROZEN. Teacher labels + encoder both from it.
  * Compute : everything on Colab GPU (label-gen is the expensive part).
  * Scope   : standalone FRM box. Stage 2a (fovea) exclusion is a fixed 1-ring
              rule here; Stage 2c (eccentricity merge) is a mean-pool stand-in.
"""
import os

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))

# WearVQA gaze-only dataset on Drive: <qtype>/<id>.{jpg,json}
DATA_DIR = os.environ.get("DATA_DIR", "/content/drive/MyDrive/wearvqa_gaze_only")

# where Phase-1 cached labels + trained checkpoints live (persist on Drive)
OUT_DIR = os.environ.get("FRM_OUT_DIR", "/content/drive/MyDrive/frm_out")
LABELS_DIR = os.path.join(OUT_DIR, "labels")          # cached G + imp_* per sample
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(OUT_DIR, "results")
for _d in (OUT_DIR, LABELS_DIR, CKPT_DIR, RESULTS_DIR):
    try:                       # on Colab the Drive path is writable; skip if not mounted yet
        os.makedirs(_d, exist_ok=True)
    except OSError:
        pass

# single-file artifacts written by Phase 1 (labels.py)
G_MEMMAP = os.path.join(LABELS_DIR, "G.fp16.dat")     # [N, K, D] float16 memmap
META_JSONL = os.path.join(LABELS_DIR, "meta.jsonl")   # one row/sample: id, qtype, g_idx, imp_*, question, answer
IMP_NPZ = os.path.join(LABELS_DIR, "imp.npz")         # imp_answer/question/context/loo [N, K]

# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
MODEL_ID = os.environ.get("MODEL_ID", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
DTYPE = "bfloat16"          # for the frozen VLM on GPU
DEVICE = "cuda"             # label-gen + answer-preservation need GPU

# --------------------------------------------------------------------------- #
# global-thumbnail grid  (SmolVLM2-2.2B: 81 image tokens/tile on a 9x9 grid)
# --------------------------------------------------------------------------- #
K = 81                     # global tokens (matches Stage-3 SEQ=81 for the 2.2B)
GRID = 9                   # sqrt(K); g_idx = gy*GRID + gx
# D (hidden dim) is read dynamically from the model at run time (~2048 for the 2.2B).
# We assert the model really yields K=81 tokens and store the observed D in meta.

# --------------------------------------------------------------------------- #
# fovea / FRM partition (Section 3.1) — the gaze cell + its 1-ring are EXCLUDED
# from FRM's keys/candidates AND from the renormalized teacher label.
# --------------------------------------------------------------------------- #
FOVEA_RING = int(os.environ.get("FOVEA_RING", "1"))   # 1 -> exclude 3x3 around gaze cell

# --------------------------------------------------------------------------- #
# teacher (label generator) — attention rollout is the default; LOO validates a subset
# --------------------------------------------------------------------------- #
TEACHER_METHOD = os.environ.get("TEACHER_METHOD", "rollout")  # rollout | loo
ROLLOUT_RESIDUAL = 0.5     # A_tilde = 0.5*A.mean(heads) + 0.5*I
LOO_VALIDATE_N = 50        # how many samples to also label with LOO (Spearman vs rollout)

# --------------------------------------------------------------------------- #
# student (FRM cross-attention head — the ONLY trained part)
# --------------------------------------------------------------------------- #
D_H = int(os.environ.get("D_H", "256"))     # projection dim
N_HEADS = int(os.environ.get("N_HEADS", "8"))
DEPTH = int(os.environ.get("DEPTH", "1"))   # 1 = single cross-attn block (Exp 7 sweeps this)

# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #
LAMBDA_ANS = float(os.environ.get("LAMBDA_ANS", "0.0"))  # answer-preservation aux (0 = pure distillation)

# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = int(os.environ.get("EPOCHS", "200"))
BATCH_SIZE = 64
SEEDS = [0, 1, 2, 3, 4]
N_FOLDS = 5                # StratifiedGroupKFold, group = scene/image id (no leakage across split)

# --------------------------------------------------------------------------- #
# deploy / experiments
# --------------------------------------------------------------------------- #
KEEP_N = 12                        # default KEEP_g budget (top-n relevant context)
BUDGETS = [4, 8, 12, 16, 24]       # Exp 5 sweep
CAPACITY_GRID = [                  # Exp 7 sweep: (n_heads, depth)
    (1, 1),   # minimal: single-head depth-1
    (8, 1),   # multi-head depth-1
    (8, 2),   # 2-layer stacked
]
# far-context stratification: a sample is "FAR" if its answer-importance centroid
# is at least FAR_THRESH grid cells from the gaze cell (that is where FRM must win).
FAR_THRESH = float(os.environ.get("FAR_THRESH", "2.0"))
