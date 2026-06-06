"""LLaVA-NeXT (v1.6) adapter for llava-hf checkpoints (mistral-7b, vicuna-13b).

The 7B/13B decoders are loaded in 4-bit on a T4 (see config `load_in_4bit`).
"""
import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, TextIteratorStreamer

from .base import BaseVLMRunner
from ..metrics.efficiency import (
    GenerationMetrics,
    make_metrics,
    peak_vram_gb,
    reset_vram_peak,
    run_generation_with_streamer,
)


class LlavaNextRunner(BaseVLMRunner):
    name = "llava_next"

    def load(self) -> None:
        self.processor = LlavaNextProcessor.from_pretrained(self.hf_id)
        self.tokenizer = self.processor.tokenizer
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.hf_id,
            attn_implementation="eager",
            **self._hf_load_kwargs(),
        ).eval()

    def _prep_inputs(self, image: Image.Image, question: str):
        conversation = [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image"},
            ],
        }]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)
        return inputs

    def _count_visual_tokens(self, inputs) -> int:
        img_id = getattr(self.model.config, "image_token_index", None)
        if img_id is None:
            return -1
        return int((inputs["input_ids"] == img_id).sum().item())

    def infer(self, image: Image.Image, question: str, max_new_tokens: int) -> GenerationMetrics:
        reset_vram_peak()
        inputs = self._prep_inputs(image, question)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300)
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
