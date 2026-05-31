"""Efficiency metric helpers: VRAM peak, TTFT / throughput via streamer, visual token counting."""
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class GenerationMetrics:
    ttft_s: float                  # time-to-first-token (seconds)
    total_generation_s: float      # full generate() wall time
    decode_s: float                # total - ttft (time spent generating tokens after the first)
    output_tokens: int             # number of generated tokens (excluding prompt)
    throughput_tok_s: float        # output_tokens / decode_s (post-TTFT decode throughput)
    peak_vram_gb: float            # torch.cuda.max_memory_allocated() during this sample
    visual_tokens: int             # # visual tokens the VLM sent to the LLM decoder
    output_text: str               # decoded model output


def reset_vram_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def run_generation_with_streamer(
    model,
    generate_kwargs: dict,
    streamer,
    tokenizer,
    skip_special_tokens: bool = True,
) -> tuple[str, float, float, int]:
    """
    Run model.generate(**generate_kwargs, streamer=streamer) inside a thread,
    while reading tokens off the streamer to measure TTFT.

    Returns (output_text, ttft_s, total_s, output_tokens).
    """
    generate_kwargs = dict(generate_kwargs)
    generate_kwargs["streamer"] = streamer

    error_box = {}

    def _worker():
        try:
            with torch.no_grad():
                model.generate(**generate_kwargs)
        except Exception as e:
            error_box["err"] = e

    t0 = time.perf_counter()
    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    pieces: list[str] = []
    ttft: Optional[float] = None
    # If generate() raises before the streamer emits its end signal, iterating the
    # streamer would otherwise block forever. A streamer built with a timeout raises
    # queue.Empty instead; we then join the worker and re-raise the real error.
    try:
        for piece in streamer:
            if ttft is None:
                ttft = time.perf_counter() - t0
            pieces.append(piece)
    except queue.Empty:
        pass

    thread.join()
    if "err" in error_box:
        raise error_box["err"]

    total = time.perf_counter() - t0
    text = "".join(pieces)
    if ttft is None:
        ttft = total

    # Best-effort token count via the tokenizer (skip_special_tokens=False is more accurate).
    output_tokens = len(tokenizer(text, add_special_tokens=False).input_ids) if tokenizer is not None else 0
    return text, ttft, total, output_tokens


def make_metrics(
    output_text: str,
    ttft_s: float,
    total_s: float,
    output_tokens: int,
    peak_vram: float,
    visual_tokens: int,
) -> GenerationMetrics:
    decode_s = max(total_s - ttft_s, 1e-6)
    throughput = output_tokens / decode_s if output_tokens > 0 else 0.0
    return GenerationMetrics(
        ttft_s=ttft_s,
        total_generation_s=total_s,
        decode_s=decode_s,
        output_tokens=output_tokens,
        throughput_tok_s=throughput,
        peak_vram_gb=peak_vram,
        visual_tokens=visual_tokens,
        output_text=output_text,
    )
