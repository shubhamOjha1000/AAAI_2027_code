"""Abstract base class all VLM runners implement."""
import gc
from abc import ABC, abstractmethod
from typing import Optional

import torch
from PIL import Image

from ..metrics.efficiency import GenerationMetrics


class BaseVLMRunner(ABC):
    """Each model adapter subclasses this and implements load / infer / unload."""
    name: str = "base"
    hf_id: str = ""

    def __init__(self, hf_id: str, device: str = "cuda", dtype: str = "bfloat16",
                 load_in_4bit: bool = False):
        self.hf_id = hf_id
        self.device = device
        self.dtype = self._resolve_dtype(dtype)
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.processor = None
        self.tokenizer = None

    @staticmethod
    def _resolve_dtype(dtype: str):
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]

    def _hf_load_kwargs(self) -> dict:
        """Shared from_pretrained kwargs. 4-bit (for 7B/13B models on a T4) routes
        weights through bitsandbytes; otherwise stream straight to GPU with device_map
        to keep the CPU-RAM footprint small on Colab."""
        kwargs = {"low_cpu_mem_usage": True}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = self.dtype
            kwargs["device_map"] = self.device
        return kwargs

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def infer(self, image: Image.Image, question: str, max_new_tokens: int) -> GenerationMetrics: ...

    def unload(self) -> None:
        self.model = None
        self.processor = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
