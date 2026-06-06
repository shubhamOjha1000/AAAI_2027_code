"""Run a single VLM over the WearVQA dataset and dump predictions + efficiency metrics."""
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm

from .. import config
from ..dataset import load_wearvqa
from ..utils.io import append_jsonl, ensure_dir
from ..models.base import BaseVLMRunner


def _make_runner(model_name: str) -> BaseVLMRunner:
    entry = config.MODEL_REGISTRY[model_name]
    hf_id = entry["hf_id"]
    dtype = entry["dtype"]
    if model_name == "smolvlm2":
        from ..models.smolvlm2 import SmolVLM2Runner
        return SmolVLM2Runner(hf_id=hf_id, dtype=dtype)
    raise KeyError(f"Unknown model: {model_name}")


def run_model(
    model_name: str,
    max_samples: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    if model_name not in config.MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {model_name}. Choices: {list(config.MODEL_REGISTRY)}")

    max_samples = max_samples if max_samples is not None else config.MAX_SAMPLES
    output_dir = output_dir or config.PREDICTIONS_DIR
    ensure_dir(output_dir)
    out_path = Path(output_dir) / f"{model_name}.jsonl"
    if out_path.exists():
        out_path.unlink()

    print(f"[{model_name}] loading {config.MODEL_REGISTRY[model_name]['hf_id']}")
    runner = _make_runner(model_name)
    t_load_0 = time.perf_counter()
    runner.load()
    t_load = time.perf_counter() - t_load_0
    print(f"[{model_name}] loaded in {t_load:.1f}s")

    samples = load_wearvqa(max_samples=max_samples)
    print(f"[{model_name}] running on {len(samples)} samples (batch_size=1)")

    for s in tqdm(samples, desc=model_name):
        try:
            image = s.load_image()
            m = runner.infer(image=image, question=s.question, max_new_tokens=config.MAX_NEW_TOKENS)
            rec = {
                "model": model_name,
                "sample_id": s.sample_id,
                "idx": s.idx,
                "question": s.question,
                "ground_truth": s.ground_truth,
                "prediction": m.output_text,
                "ttft_s": m.ttft_s,
                "total_generation_s": m.total_generation_s,
                "decode_s": m.decode_s,
                "output_tokens": m.output_tokens,
                "throughput_tok_s": m.throughput_tok_s,
                "peak_vram_gb": m.peak_vram_gb,
                "visual_tokens": m.visual_tokens,
                "domain": s.domain,
                "question_type": s.question_type,
                "quality_flags": s.quality_flags,
                "hand_finger_elements": s.hand_finger_elements,
            }
        except Exception as e:
            rec = {
                "model": model_name,
                "sample_id": s.sample_id,
                "idx": s.idx,
                "question": s.question,
                "ground_truth": s.ground_truth,
                "prediction": "",
                "error": f"{type(e).__name__}: {e}",
                "domain": s.domain,
                "question_type": s.question_type,
                "quality_flags": s.quality_flags,
            }
        append_jsonl(out_path, rec)

    runner.unload()
    print(f"[{model_name}] done -> {out_path}")
    return out_path
