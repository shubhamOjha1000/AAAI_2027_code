"""Run the Qwen2.5-7B LLM-as-judge on all prediction JSONL files."""
import time
from pathlib import Path
from typing import Iterable, Optional

from tqdm.auto import tqdm

from .. import config
from ..judge.qwen_judge import QwenJudge
from ..utils.io import append_jsonl, ensure_dir, read_jsonl


def _prediction_files(predictions_dir: Path) -> list[Path]:
    return sorted(p for p in Path(predictions_dir).glob("*.jsonl"))


def judge_all(
    predictions_dir: Optional[Path] = None,
    judgments_dir: Optional[Path] = None,
    judge: Optional[QwenJudge] = None,
) -> Path:
    predictions_dir = Path(predictions_dir or config.PREDICTIONS_DIR)
    judgments_dir = Path(judgments_dir or config.JUDGMENTS_DIR)
    ensure_dir(judgments_dir)
    out_path = judgments_dir / "qwen_judge.jsonl"
    if out_path.exists():
        out_path.unlink()

    owns_judge = judge is None
    if owns_judge:
        judge = QwenJudge()
        print(f"[judge] loading {judge.model_id} (4bit={judge.load_in_4bit})")
        judge.load()

    try:
        for pred_file in _prediction_files(predictions_dir):
            print(f"[judge] scoring {pred_file.name}")
            for pred in tqdm(list(read_jsonl(pred_file)), desc=pred_file.stem):
                if pred.get("error") or not pred.get("prediction"):
                    rec = {
                        "model": pred["model"],
                        "sample_id": pred["sample_id"],
                        "correct": False,
                        "reason": "empty/error prediction",
                    }
                else:
                    j = judge.judge(
                        question=pred["question"],
                        ground_truth=pred["ground_truth"],
                        prediction=pred["prediction"],
                    )
                    rec = {
                        "model": pred["model"],
                        "sample_id": pred["sample_id"],
                        "correct": bool(j["correct"]),
                        "reason": j["reason"],
                    }
                append_jsonl(out_path, rec)
    finally:
        if owns_judge:
            judge.unload()

    print(f"[judge] done -> {out_path}")
    return out_path
