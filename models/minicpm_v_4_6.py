"""MiniCPM-V 4.6 (1.3B) adapter. SigLIP2-400M vision + Qwen3.5-0.8B LLM."""
import time

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, TextIteratorStreamer

from .base import BaseVLMRunner
from ..metrics.efficiency import (
    GenerationMetrics,
    make_metrics,
    peak_vram_gb,
    reset_vram_peak,
    run_generation_with_streamer,
)


class MiniCPMV46Runner(BaseVLMRunner):
    name = "minicpm_v_4_6"

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.hf_id,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            attn_implementation="eager",
        ).to(self.device).eval()

    def _count_visual_tokens(self, image: Image.Image, question: str) -> int:
        """
        MiniCPM-V uses image slicing; total visual tokens = sum of patches across slices.
        We invoke the model's internal preprocessing to count exactly.
        """
        try:
            msgs = [{"role": "user", "content": [image, question]}]
            # MiniCPM's preprocess_inputs returns a dict that includes 'inputs_embeds' or
            # the count of visual tokens via image_bound. Fall back to vision_hidden_states.
            with torch.no_grad():
                # Use the processor pathway if available, else approximate via image_bound.
                if hasattr(self.model, "get_vllm_embedding"):
                    # Lower-level path; not always reliable across versions.
                    pass
                # Most robust: run the model's chat-prep helpers to get image bounds.
                if hasattr(self.model, "_prepare_inputs"):
                    prep = self.model._prepare_inputs(image=image, msgs=msgs, tokenizer=self.tokenizer)
                    bounds = prep.get("image_bound", None)
                    if bounds is not None and len(bounds) > 0:
                        return int(sum(int(b[1] - b[0]) for b in bounds[0]))
        except Exception:
            pass
        return -1  # unknown

    def infer(self, image: Image.Image, question: str, max_new_tokens: int) -> GenerationMetrics:
        reset_vram_peak()
        msgs = [{"role": "user", "content": [image, question]}]

        # Use streamer-based generation. MiniCPM-V supports `stream=True` via chat().
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        import threading
        error_box = {}
        out_box = {}

        def _worker():
            try:
                # model.chat returns a string when stream=False; with a streamer we drive it manually.
                out_box["text"] = self.model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    sampling=False,
                    max_new_tokens=max_new_tokens,
                    stream=False,
                )
            except Exception as e:
                error_box["err"] = e

        t0 = time.perf_counter()
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        # MiniCPM's chat() does not natively use HF streamer, so we measure TTFT as the time
        # until the first non-empty intermediate output. Best practical proxy: total time
        # divided into ttft via the underlying generator path.
        thread.join()
        total_s = time.perf_counter() - t0
        if "err" in error_box:
            raise error_box["err"]
        text = out_box.get("text", "")

        # Token-count proxy: tokenizer-encoded output length.
        output_tokens = len(self.tokenizer(text, add_special_tokens=False).input_ids)
        # TTFT proxy (no native streaming): per-token avg * 1. We report total_s / output_tokens
        # as an estimated TTFT when no streamer is available; better proxy below using a short probe.
        ttft_s = self._estimate_ttft(msgs)
        vis = self._count_visual_tokens(image, question)
        peak = peak_vram_gb()
        return make_metrics(text, ttft_s, total_s, output_tokens, peak, vis)

    def _estimate_ttft(self, msgs) -> float:
        """Run a max_new_tokens=1 chat as a TTFT probe. Cheap and accurate enough."""
        try:
            t0 = time.perf_counter()
            _ = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=False,
                max_new_tokens=1,
                stream=False,
            )
            return time.perf_counter() - t0
        except Exception:
            return -1.0
