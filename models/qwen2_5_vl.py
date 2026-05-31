"""Qwen2.5-VL-3B-Instruct adapter."""
import time

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TextIteratorStreamer,
)

from .base import BaseVLMRunner
from ..metrics.efficiency import (
    GenerationMetrics,
    make_metrics,
    peak_vram_gb,
    reset_vram_peak,
    run_generation_with_streamer,
)


class Qwen25VLRunner(BaseVLMRunner):
    name = "qwen2_5_vl"

    def load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.hf_id)
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.hf_id,
            torch_dtype=self.dtype,
            attn_implementation="eager",
        ).to(self.device).eval()

    def _prep_inputs(self, image: Image.Image, question: str):
        # Qwen2.5-VL expects qwen_vl_utils.process_vision_info. We install it via pip in setup.
        from qwen_vl_utils import process_vision_info

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        return inputs

    def _count_visual_tokens(self, inputs) -> int:
        """For Qwen2.5-VL: visual tokens to LLM = prod(image_grid_thw) / (merge_size**2)."""
        try:
            grid = inputs["image_grid_thw"]  # shape [num_images, 3]
            merge = int(getattr(self.model.config.vision_config, "spatial_merge_size", 2))
            total = 0
            for row in grid:
                t, h, w = int(row[0]), int(row[1]), int(row[2])
                total += (t * h * w) // (merge * merge)
            return total
        except Exception:
            return -1

    def infer(self, image: Image.Image, question: str, max_new_tokens: int) -> GenerationMetrics:
        reset_vram_peak()
        inputs = self._prep_inputs(image, question)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        text, ttft, total, out_tokens = run_generation_with_streamer(
            self.model, gen_kwargs, streamer, self.tokenizer
        )
        vis = self._count_visual_tokens(inputs)
        peak = peak_vram_gb()
        return make_metrics(text, ttft, total, out_tokens, peak, vis)
