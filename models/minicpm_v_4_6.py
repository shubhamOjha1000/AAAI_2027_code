"""MiniCPM-V-4.6 (1.3B) adapter.

Uses the modern transformers API (AutoModelForImageTextToText + apply_chat_template
with `downsample_mode`). Requires transformers>=5.7.0.
"""
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer

from .base import BaseVLMRunner
from ..metrics.efficiency import (
    GenerationMetrics,
    make_metrics,
    peak_vram_gb,
    reset_vram_peak,
    run_generation_with_streamer,
)

# "16x" merges visual tokens for efficiency (default); "4x" keeps 4x more for detail.
DOWNSAMPLE_MODE = "16x"
# Max slices when splitting a high-res image. Card recommends 36 for images; 9 is the
# default and keeps the visual-token count / VRAM lower on a T4.
MAX_SLICE_NUMS = 9


class MiniCPMV46Runner(BaseVLMRunner):
    name = "minicpm_v_4_6"

    def load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.hf_id)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.hf_id,
            attn_implementation="eager",
            **self._hf_load_kwargs(),
        ).eval()

    def _prep_inputs(self, image: Image.Image, question: str):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            downsample_mode=DOWNSAMPLE_MODE,
            max_slice_nums=MAX_SLICE_NUMS,
        ).to(self.model.device)
        return inputs

    def _count_visual_tokens(self, input_ids: torch.Tensor) -> int:
        img_id = getattr(self.tokenizer, "image_token_id", None)
        if img_id is None:
            return -1
        return int((input_ids == img_id).sum().item())

    def infer(self, image: Image.Image, question: str, max_new_tokens: int) -> GenerationMetrics:
        reset_vram_peak()
        inputs = self._prep_inputs(image, question)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300)
        gen_kwargs = dict(
            **inputs,
            downsample_mode=DOWNSAMPLE_MODE,  # must mirror apply_chat_template
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        text, ttft, total, out_tokens = run_generation_with_streamer(
            self.model, gen_kwargs, streamer, self.tokenizer
        )
        vis = self._count_visual_tokens(inputs["input_ids"])
        peak = peak_vram_gb()
        return make_metrics(text, ttft, total, out_tokens, peak, vis)
