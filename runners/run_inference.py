"""Run a VLM over the WearVQA dataset and dump predictions (+ efficiency metrics).

Two modes:
  - "accuracy"   : batched generation, predictions only (fast; for accuracy checks).
  - "efficiency" : batch_size=1 + token streaming, records TTFT/throughput/VRAM.

CLI:
  python -m wearvqa_bench.runners.run_inference --model smolvlm2 --mode accuracy --batch-size 256
  python -m wearvqa_bench.runners.run_inference --model all      --mode accuracy
"""
import argparse
import time
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm

from .. import config
from ..dataset import load_wearvqa, WearVQASample
from ..utils.io import append_jsonl, ensure_dir
from ..models.base import BaseVLMRunner


def _make_runner(model_name: str) -> BaseVLMRunner:
    entry = config.MODEL_REGISTRY[model_name]
    runner = entry.get("runner", model_name)
    common = dict(
        hf_id=entry["hf_id"],
        dtype=entry["dtype"],
        load_in_4bit=entry.get("load_in_4bit", False),
    )
    if runner == "smolvlm":
        from ..models.smolvlm import SmolVLMRunner
        return SmolVLMRunner(**common)
    if runner == "minicpm_v_4_6":
        from ..models.minicpm_v_4_6 import MiniCPMV46Runner
        return MiniCPMV46Runner(**common)
    if runner == "internvl2_5":
        from ..models.internvl2_5 import InternVL25Runner
        return InternVL25Runner(**common)
    if runner == "llava_next":
        from ..models.llava_next import LlavaNextRunner
        return LlavaNextRunner(**common)
    raise KeyError(f"Unknown runner: {runner} (model {model_name})")


def _base_record(model_name: str, s: WearVQASample) -> dict:
    return {
        "model": model_name,
        "sample_id": s.sample_id,
        "idx": s.idx,
        "question": s.question,
        "ground_truth": s.ground_truth,
        "domain": s.domain,
        "question_type": s.question_type,
        "quality_flags": s.quality_flags,
        "hand_finger_elements": s.hand_finger_elements,
    }


def _run_accuracy(runner, model_name, samples, out_path, batch_size):
    """Batched generation; writes predictions only (no efficiency metrics)."""
    for i in tqdm(range(0, len(samples), batch_size), desc=f"{model_name} (acc)"):
        chunk = samples[i:i + batch_size]
        questions = [s.question for s in chunk]
        try:
            images = [s.load_image() for s in chunk]
            preds = runner.predict_batch(images, questions, config.MAX_NEW_TOKENS)
            err = None
        except Exception as e:  # noqa: BLE001
            preds, err = None, f"{type(e).__name__}: {e}"
        for j, s in enumerate(chunk):
            rec = _base_record(model_name, s)
            rec["prediction"] = "" if preds is None else preds[j]
            if err is not None:
                rec["error"] = err
            append_jsonl(out_path, rec)


def _run_efficiency(runner, model_name, samples, out_path):
    """batch_size=1 streaming; records TTFT / throughput / peak VRAM / visual tokens."""
    for s in tqdm(samples, desc=f"{model_name} (eff)"):
        try:
            m = runner.infer(image=s.load_image(), question=s.question,
                             max_new_tokens=config.MAX_NEW_TOKENS)
            rec = _base_record(model_name, s)
            rec.update({
                "prediction": m.output_text,
                "ttft_s": m.ttft_s,
                "total_generation_s": m.total_generation_s,
                "decode_s": m.decode_s,
                "output_tokens": m.output_tokens,
                "throughput_tok_s": m.throughput_tok_s,
                "peak_vram_gb": m.peak_vram_gb,
                "visual_tokens": m.visual_tokens,
            })
        except Exception as e:  # noqa: BLE001
            rec = _base_record(model_name, s)
            rec.update({"prediction": "", "error": f"{type(e).__name__}: {e}"})
        append_jsonl(out_path, rec)


def run_model(
    model_name: str,
    max_samples: Optional[int] = None,
    output_dir: Optional[Path] = None,
    mode: str = "accuracy",
    batch_size: Optional[int] = None,
) -> Path:
    if model_name not in config.MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {model_name}. Choices: {list(config.MODEL_REGISTRY)}")
    if mode not in {"accuracy", "efficiency"}:
        raise ValueError(f"mode must be 'accuracy' or 'efficiency', got {mode!r}")

    max_samples = max_samples if max_samples is not None else config.MAX_SAMPLES
    batch_size = batch_size if batch_size is not None else config.BATCH_SIZE
    output_dir = output_dir or config.PREDICTIONS_DIR
    ensure_dir(output_dir)
    out_path = Path(output_dir) / f"{model_name}.jsonl"
    if out_path.exists():
        out_path.unlink()

    print(f"[{model_name}] loading {config.MODEL_REGISTRY[model_name]['hf_id']}")
    runner = _make_runner(model_name)
    t0 = time.perf_counter()
    runner.load()
    print(f"[{model_name}] loaded in {time.perf_counter() - t0:.1f}s")

    samples = load_wearvqa(max_samples=max_samples)
    if mode == "accuracy":
        print(f"[{model_name}] {len(samples)} samples | mode=accuracy | batch_size={batch_size}")
        _run_accuracy(runner, model_name, samples, out_path, batch_size)
    else:
        print(f"[{model_name}] {len(samples)} samples | mode=efficiency | batch_size=1")
        _run_efficiency(runner, model_name, samples, out_path)

    runner.unload()
    print(f"[{model_name}] done -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Run WearVQA inference for one or all VLMs.")
    ap.add_argument("--model", required=True,
                    help="registry key (e.g. smolvlm2) or 'all'")
    ap.add_argument("--mode", choices=["accuracy", "efficiency"], default="accuracy",
                    help="accuracy = batched predictions only; efficiency = streamed metrics")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="accuracy mode only; default config.BATCH_SIZE (%d)" % config.BATCH_SIZE)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="cap #samples; -1 = full set; default config.MAX_SAMPLES")
    args = ap.parse_args()

    max_samples = None if (args.max_samples is not None and args.max_samples < 0) else args.max_samples
    models = list(config.MODEL_REGISTRY) if args.model == "all" else [args.model]
    for m in models:
        run_model(m, max_samples=max_samples, mode=args.mode, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
