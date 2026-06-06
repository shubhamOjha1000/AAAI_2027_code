"""Run the LLM-as-judge ensemble over all prediction JSONL files.

Each judge in config.JUDGE_REGISTRY runs sequentially (load -> score every
prediction file -> unload) and writes its own judgments/<judge_key>.jsonl.
"""
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm

from .. import config
from ..judge.llm_judge import LLMJudge
from ..utils.io import append_jsonl, ensure_dir, read_jsonl


def _prediction_files(predictions_dir: Path, models=None) -> list[Path]:
    files = sorted(p for p in Path(predictions_dir).glob("*.jsonl"))
    if models:
        wanted = set(models)
        files = [p for p in files if p.stem in wanted]
    return files


def _judge_one(judge_key, entry, predictions_dir, judgments_dir, models):
    out_path = judgments_dir / f"{judge_key}.jsonl"
    if out_path.exists():
        out_path.unlink()

    judge = LLMJudge(model_id=entry["model_id"], load_in_4bit=entry["load_in_4bit"])
    print(f"[{judge_key}] loading {judge.model_id} (4bit={judge.load_in_4bit})")
    judge.load()
    try:
        for pred_file in _prediction_files(predictions_dir, models):
            print(f"[{judge_key}] scoring {pred_file.name}")
            for pred in tqdm(list(read_jsonl(pred_file)), desc=f"{judge_key}:{pred_file.stem}"):
                if pred.get("error") or not pred.get("prediction"):
                    rec = {"model": pred["model"], "sample_id": pred["sample_id"],
                           "judge": judge_key, "correct": False, "reason": "empty/error prediction"}
                else:
                    j = judge.judge(
                        question=pred["question"],
                        ground_truth=pred["ground_truth"],
                        prediction=pred["prediction"],
                    )
                    rec = {"model": pred["model"], "sample_id": pred["sample_id"],
                           "judge": judge_key, "correct": bool(j["correct"]), "reason": j["reason"]}
                append_jsonl(out_path, rec)
    finally:
        judge.unload()
    print(f"[{judge_key}] done -> {out_path}")
    return out_path


def judge_all(
    predictions_dir: Optional[Path] = None,
    judgments_dir: Optional[Path] = None,
    models=None,
    judges=None,
) -> Path:
    """Score predictions with every judge in JUDGE_REGISTRY (or the subset in `judges`)."""
    predictions_dir = Path(predictions_dir or config.PREDICTIONS_DIR)
    judgments_dir = Path(judgments_dir or config.JUDGMENTS_DIR)
    ensure_dir(judgments_dir)

    judge_keys = judges or list(config.JUDGE_REGISTRY)
    for judge_key in judge_keys:
        _judge_one(judge_key, config.JUDGE_REGISTRY[judge_key],
                   predictions_dir, judgments_dir, models)

    return judgments_dir
