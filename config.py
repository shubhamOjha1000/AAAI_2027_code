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
# "runner" selects the adapter class in runners/run_inference.py. "load_in_4bit"
# (optional) routes the model through bitsandbytes 4-bit for T4-sized memory.
MODEL_REGISTRY = {
    # --- SmolVLM / SmolVLM2 family (Idefics3) ---
    "smolvlm_256m": {
        "hf_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "display_name": "SmolVLM-256M",
        "dtype": "float16",
        "runner": "smolvlm",
    },
    "smolvlm_500m": {
        "hf_id": "HuggingFaceTB/SmolVLM-500M-Instruct",
        "display_name": "SmolVLM-500M",
        "dtype": "float16",
        "runner": "smolvlm",
    },
    "smolvlm": {
        "hf_id": "HuggingFaceTB/SmolVLM-Instruct",
        "display_name": "SmolVLM (2.2B)",
        "dtype": "float16",
        "runner": "smolvlm",
    },
    "smolvlm2": {
        "hf_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "display_name": "SmolVLM2 (2.2B)",
        "dtype": "float16",
        "runner": "smolvlm",
    },
    # --- MiniCPM-V 4.6 (modern apply_chat_template API; needs transformers>=5.7) ---
    "minicpm_v_4_6": {
        "hf_id": "openbmb/MiniCPM-V-4.6",
        "display_name": "MiniCPM-V 4.6 (1.3B)",
        "dtype": "float16",
        "runner": "minicpm_v_4_6",
    },
    # --- InternVL2.5 ---
    "internvl2_5_1b": {
        "hf_id": "OpenGVLab/InternVL2_5-1B",
        "display_name": "InternVL2.5-1B",
        "dtype": "float16",
        "runner": "internvl2_5",
    },
    "internvl2_5_2b": {
        "hf_id": "OpenGVLab/InternVL2_5-2B",
        "display_name": "InternVL2.5-2B",
        "dtype": "float16",
        "runner": "internvl2_5",
    },
    # --- LLaVA-NeXT (v1.6); full fp16 (fits an A100) ---
    "llava_next_mistral_7b": {
        "hf_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "display_name": "LLaVA-1.6 Mistral-7B",
        "dtype": "float16",
        "runner": "llava_next",
    },
    "llava_next_vicuna_13b": {
        "hf_id": "llava-hf/llava-v1.6-vicuna-13b-hf",
        "display_name": "LLaVA-1.6 Vicuna-13B",
        "dtype": "float16",
        "runner": "llava_next",
    },
}

# LLM-as-judge config. Full fp16 (fits an A100); set to True to 4-bit on smaller GPUs.
JUDGE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
JUDGE_LOAD_IN_4BIT = False
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
