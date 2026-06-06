"""InternVL2.5 adapter (1B / 2B).

Follows the official model card: dynamic-tiling image preprocessing + model.chat().
flash-attn is disabled (use_flash_attn=False) because a T4 cannot run it.
"""
import queue
import threading
import time

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from transformers import AutoModel, AutoTokenizer, TextIteratorStreamer

from .base import BaseVLMRunner
from ..metrics.efficiency import (
    GenerationMetrics,
    make_metrics,
    peak_vram_gb,
    reset_vram_peak,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    ratio = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_w, orig_h, image_size)
    target_w = image_size * ratio[0]
    target_h = image_size * ratio[1]
    blocks = ratio[0] * ratio[1]
    resized = image.resize((target_w, target_h))
    cols = target_w // image_size
    tiles = []
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


class InternVL25Runner(BaseVLMRunner):
    name = "internvl2_5"
    INPUT_SIZE = 448
    MAX_NUM = 12

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_id, trust_remote_code=True, use_fast=False
        )
        self.model = AutoModel.from_pretrained(
            self.hf_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=False,        # T4 (Turing) has no FlashAttention
            trust_remote_code=True,
        ).eval().to(self.device)

    def _pixel_values(self, image: Image.Image) -> torch.Tensor:
        transform = _build_transform(self.INPUT_SIZE)
        tiles = _dynamic_preprocess(
            image, image_size=self.INPUT_SIZE, use_thumbnail=True, max_num=self.MAX_NUM
        )
        pixel_values = torch.stack([transform(t) for t in tiles])
        return pixel_values.to(self.dtype).to(self.device)

    def _count_visual_tokens(self, pixel_values: torch.Tensor) -> int:
        n_tok = getattr(self.model, "num_image_token", None)
        if n_tok is None:
            return -1
        return int(pixel_values.shape[0] * n_tok)

    def infer(self, image: Image.Image, question: str, max_new_tokens: int) -> GenerationMetrics:
        reset_vram_peak()
        pixel_values = self._pixel_values(image)
        prompt = "<image>\n" + question

        # InternVL's chat() forwards **generation_config to generate(), so a streamer
        # placed there gives us real TTFT. Run chat in a worker thread and read tokens.
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300
        )
        gen_cfg = dict(max_new_tokens=max_new_tokens, do_sample=False, streamer=streamer)
        error_box = {}

        def _worker():
            try:
                self.model.chat(self.tokenizer, pixel_values, prompt, gen_cfg)
            except Exception as e:  # noqa: BLE001 - surfaced after join
                error_box["err"] = e

        t0 = time.perf_counter()
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        pieces, ttft = [], None
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
        out_tokens = len(self.tokenizer(text, add_special_tokens=False).input_ids)
        vis = self._count_visual_tokens(pixel_values)
        peak = peak_vram_gb()
        return make_metrics(text, ttft, total, out_tokens, peak, vis)
