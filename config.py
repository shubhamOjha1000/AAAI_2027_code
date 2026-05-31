"""Central configuration for the WearVQA benchmark pipeline."""
import os
from pathlib import Path

# Root paths. On Colab the code is cloned from GitHub while the dataset lives
# directly on Drive, so the dataset is *not* under PROJECT_ROOT. Override any of
# these from the notebook by exporting the matching env var before importing this
# module:
#   WEARVQA_DATASET_DIR   e.g. /content/drive/MyDrive/wearvqa
#   WEARVQA_RESULTS_DIR   e.g. /content/drive/MyDrive/wearvqa_results
#   WEARVQA_MAX_SAMPLES   integer, or "none"/"all" for the full test set
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(os.environ.get("WEARVQA_DATASET_DIR", PROJECT_ROOT / "wearvqa"))
RESULTS_DIR = Path(os.environ.get("WEARVQA_RESULTS_DIR", PROJECT_ROOT / "results"))
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
JUDGMENTS_DIR = RESULTS_DIR / "judgments"
REPORTS_DIR = RESULTS_DIR / "reports"

# Smoke-test cap. Set to None for the full 1,500-sample public test set.
_env_max = os.environ.get("WEARVQA_MAX_SAMPLES")
if _env_max is None:
    MAX_SAMPLES = 10
elif _env_max.strip().lower() in {"none", "all", ""}:
    MAX_SAMPLES = None
else:
    MAX_SAMPLES = int(_env_max)

# Inference settings.
BATCH_SIZE = 1
MAX_NEW_TOKENS = 256
DO_SAMPLE = False

# Model registry. Keys are the names used by run_model("<key>").
MODEL_REGISTRY = {
    "minicpm_v_4_6": {
        "hf_id": "openbmb/MiniCPM-V-4_6",
        "display_name": "MiniCPM-V 4.6 (1.3B)",
        "dtype": "bfloat16",
    },
    "smolvlm2": {
        "hf_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "display_name": "SmolVLM2 (2.2B)",
        "dtype": "bfloat16",
    },
    "qwen2_5_vl": {
        "hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "display_name": "Qwen2.5-VL (3B)",
        "dtype": "bfloat16",
    },
    "phi_3_5_v": {
        "hf_id": "microsoft/Phi-3.5-vision-instruct",
        "display_name": "Phi-3.5-Vision (4.2B)",
        "dtype": "bfloat16",
    },
}

# LLM-as-judge config.
JUDGE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
JUDGE_LOAD_IN_4BIT = True
JUDGE_MAX_NEW_TOKENS = 256
JUDGE_BATCH_SIZE = 1

# Quality-issue field names from the WearVQA JSON schema.
QUALITY_FIELDS = [
    "is_not_zoomed_in",
    "is_leveling",
    "is_cut_off",
    "is_blur",
    "is_low_light",
    "is_occluded",
]
