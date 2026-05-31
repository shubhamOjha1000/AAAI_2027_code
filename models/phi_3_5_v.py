"""Phi-3.5-Vision-Instruct adapter (microsoft/Phi-3.5-vision-instruct)."""
import time

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    PretrainedConfig,
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


def _force_eager_attention(cfg: PretrainedConfig) -> None:
    """Recursively set eager attention on a config and every nested sub-config."""
    cfg._attn_implementation = "eager"
    cfg._attn_implementation_autoset = True
    for value in vars(cfg).values():
        if isinstance(value, PretrainedConfig):
            _force_eager_attention(value)


class Phi35VRunner(BaseVLMRunner):
    name = "phi_3_5_v"

    def load(self) -> None:
        # num_crops=4 is the recommended single-image setting; 16 for multi-image.
        self.processor = AutoProcessor.from_pretrained(
            self.hf_id, trust_remote_code=True, num_crops=4
        )
        self.tokenizer = self.processor.tokenizer

        # Phi-3.5-V's remote code defaults to FlashAttention-2, which a T4 (Turing, sm75)
        # cannot run at all. The attn_implementation kwarg doesn't propagate through the
        # custom Phi3VConfig, so recursively force eager on the config and EVERY nested
        # sub-config (the vision encoder hides under one of them) before loading.
        cfg = AutoConfig.from_pretrained(self.hf_id, trust_remote_code=True)
        _force_eager_attention(cfg)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id,
            config=cfg,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            device_map=self.device,  # stream weights straight to GPU; avoids CPU-RAM spike
        ).eval()

    def _prep_inputs(self, image: Image.Image, question: str):
        messages = [{"role": "user", "content": f"<|image_1|>\n{question}"}]
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(prompt, [image], return_tensors="pt").to(self.device)
        return inputs

    def _count_visual_tokens(self, inputs) -> int:
        """
        Phi-3.5-V represents visual tokens with negative IDs in input_ids.
        Count occurrences of negative IDs as visual tokens.
        """
        try:
            ids = inputs["input_ids"]
            return int((ids < 0).sum().item())
        except Exception:
            return -1

    def infer(self, image: Image.Image, question: str, max_new_tokens: int) -> GenerationMetrics:
        reset_vram_peak()
        inputs = self._prep_inputs(image, question)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        text, ttft, total, out_tokens = run_generation_with_streamer(
            self.model, gen_kwargs, streamer, self.tokenizer
        )
        vis = self._count_visual_tokens(inputs)
        peak = peak_vram_gb()
        return make_metrics(text, ttft, total, out_tokens, peak, vis)
