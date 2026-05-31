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

    def __init__(self, hf_id: str, device: str = "cuda", dtype: str = "bfloat16"):
        self.hf_id = hf_id
        self.device = device
        self.dtype = self._resolve_dtype(dtype)
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
