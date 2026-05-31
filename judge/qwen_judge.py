"""LLM-as-judge using Qwen2.5-7B-Instruct in 4-bit. Mirrors the WearVQA paper's 5-criteria rubric."""
import gc
import json
import re
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .. import config


JUDGE_SYSTEM = (
    "You are an impartial evaluator for a visual question-answering benchmark. "
    "Given a question, the ground-truth answer, and a model's response, decide whether the "
    "model's response is CORRECT based on five criteria from the WearVQA benchmark:\n"
    "1. Factual correctness: factually accurate, no hallucinations.\n"
    "2. Relevance: directly addresses the question.\n"
    "3. Completeness: fully answers the question.\n"
    "4. Egocentric phrasing: written from a first-person/wearable assistant point of view.\n"
    "5. Conciseness: brief, no unnecessary information.\n\n"
    "Return a strict JSON object with two keys: "
    "\"correct\" (true or false) and \"reason\" (one short sentence). "
    "Mark CORRECT only if all five criteria are reasonably met."
)


JUDGE_USER_TEMPLATE = (
    "Question: {question}\n"
    "Ground truth: {ground_truth}\n"
    "Model response: {prediction}\n\n"
    "Respond with JSON only."
)


class QwenJudge:
    def __init__(
        self,
        model_id: str = config.JUDGE_MODEL_ID,
        load_in_4bit: bool = config.JUDGE_LOAD_IN_4BIT,
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        kwargs = {"attn_implementation": "eager"}
        if self.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        if not self.load_in_4bit:
            self.model = self.model.to(self.device)
        self.model.eval()

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _format(self, question: str, ground_truth: str, prediction: str) -> str:
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                question=question, ground_truth=ground_truth, prediction=prediction
            )},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @staticmethod
    def _parse(output_text: str) -> dict:
        """Find the first JSON object in the model's output."""
        m = re.search(r"\{.*?\}", output_text, flags=re.DOTALL)
        if not m:
            return {"correct": False, "reason": "judge: no JSON found", "raw": output_text}
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return {"correct": False, "reason": "judge: invalid JSON", "raw": output_text}
        return {
            "correct": bool(obj.get("correct", False)),
            "reason": str(obj.get("reason", "")),
            "raw": output_text,
        }

    @torch.no_grad()
    def judge(self, question: str, ground_truth: str, prediction: str,
              max_new_tokens: int = config.JUDGE_MAX_NEW_TOKENS) -> dict:
        prompt = self._format(question, ground_truth, prediction)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        gen = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        return self._parse(text)
