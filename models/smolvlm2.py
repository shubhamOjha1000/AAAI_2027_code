"""SmolVLM2-2.2B-Instruct adapter."""
import time

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, TextIteratorStreamer

from .base import BaseVLMRunner
from ..metrics.efficiency import (
    GenerationMetrics,
    make_metrics,
    peak_vram_gb,
    reset_vram_peak,
    run_generation_with_streamer,
)


class SmolVLM2Runner(BaseVLMRunner):
    name = "smolvlm2"

    def load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.hf_id)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.hf_id,
            torch_dtype=self.dtype,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            device_map=self.device,  # stream weights straight to GPU; avoids CPU-RAM spike
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
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device, dtype=self.dtype)
        return inputs

    def _count_visual_tokens(self, input_ids: torch.Tensor) -> int:
        img_id = getattr(self.processor.tokenizer, "image_token_id", None)
        if img_id is None:
            tok = self.processor.tokenizer.convert_tokens_to_ids("<image>")
            img_id = tok if tok != self.processor.tokenizer.unk_token_id else None
        if img_id is None:
            return -1
        return int((input_ids == img_id).sum().item())

    def infer(self, image: Image.Image, question: str, max_new_tokens: int) -> GenerationMetrics:
        reset_vram_peak()
        inputs = self._prep_inputs(image, question)
        prompt_len = inputs["input_ids"].shape[1]

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        text, ttft, total, out_tokens = run_generation_with_streamer(
            self.model, gen_kwargs, streamer, self.tokenizer
        )
        vis = self._count_visual_tokens(inputs["input_ids"])
        peak = peak_vram_gb()
        return make_metrics(text, ttft, total, out_tokens, peak, vis)
