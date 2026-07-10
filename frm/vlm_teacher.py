"""Frozen SmolVLM2-2.2B wrapper: the encoder (G) + the teacher label generator.

Runs on Colab GPU only. Nothing here is trained.

Responsibilities
----------------
1. encode_global(image)                -> G [K, d]      (question-free, post-connector)
2. build_bundle(image, question, answer)-> token bundle (ids, image/question/answer spans)
3. rollout_importance(bundle)          -> imp_answer, imp_question [K]  (Method A)
4. loo_importance(bundle, G)           -> imp_loo [K]                   (Method B, subset)
5. answer_preservation(bundle, G, keep)-> {logp_full, logp_keep, gen_match}

Key implementation facts for SmolVLM2 (transformers >= 4.46):
  * Load with attn_implementation="eager" so output_attentions works.
  * do_image_splitting=False  -> exactly K=81 global image tokens (one 9x9 tile).
  * image token id = model.config.image_token_id.
  * The model accepts inputs_embeds (no pixel_values) -> lets us inject a modified G
    (mask a token for LOO, or drop non-KEEP tokens for answer-preservation).
"""
import numpy as np
import torch

import config as C


def _dtype(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _resolve_dtype(name):
    """'auto' -> bf16 only where the GPU supports it (A100/L4), else fp16 (T4)."""
    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return _dtype(name)


class VLMTeacher:
    def __init__(self, model_id=C.MODEL_ID, device=C.DEVICE, dtype=C.DTYPE):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.device = device
        self.torch_dtype = _resolve_dtype(dtype)
        self.processor = AutoProcessor.from_pretrained(model_id)
        # one 9x9 global tile -> exactly K image tokens
        try:
            self.processor.image_processor.do_image_splitting = False
        except Exception:
            pass
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, attn_implementation="eager"
        ).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.image_token_id = int(self.model.config.image_token_id)
        self.d = int(self.model.config.text_config.hidden_size)
        self.eos = self.processor.tokenizer.eos_token or ""
        print(f"[VLMTeacher] dtype={self.torch_dtype}, d={self.d}, "
              f"image_token_id={self.image_token_id}")

    # ------------------------------------------------------------------ #
    # low-level helpers
    # ------------------------------------------------------------------ #
    def _prompt_text(self, question):
        messages = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def _proc(self, text, image):
        enc = self.processor(text=text, images=[image], return_tensors="pt").to(self.device)
        # cast floating inputs (pixel_values) to the model dtype so conv/matmul
        # weights and inputs match (fixes "Expected bias Float but got BFloat16");
        # leave int tensors (input_ids, masks) alone.
        for k, v in list(enc.items()):
            if torch.is_tensor(v) and torch.is_floating_point(v):
                enc[k] = v.to(self.torch_dtype)
        return enc

    @staticmethod
    def _as_tensor(x):
        """Normalize a HF return (ModelOutput / tuple / tensor) to a Tensor."""
        if hasattr(x, "last_hidden_state"):
            x = x.last_hidden_state
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x

    def _image_features(self, inputs):
        """Post-connector global tokens -> [K, d] float32 (question-free).

        Robust across transformers versions: try get_image_features (on the inner
        model, then the wrapper), accept it only if it yields exactly K tokens of
        dim d; otherwise run vision_model -> connector manually (guaranteed
        post-connector, K x d).
        """
        pv = inputs["pixel_values"]
        kw = {}
        if "pixel_attention_mask" in inputs:
            kw["pixel_attention_mask"] = inputs["pixel_attention_mask"]
        base = getattr(self.model, "model", self.model)
        feats = None
        with torch.no_grad():
            for obj in (base, self.model):
                if obj is not None and hasattr(obj, "get_image_features"):
                    try:
                        t = self._as_tensor(obj.get_image_features(pixel_values=pv, **kw))
                        t = t.reshape(-1, self.d)
                        if t.shape[0] == C.K:            # right count of post-connector tokens
                            feats = t
                            break
                    except Exception:
                        pass
            if feats is None:                            # manual vision tower + connector
                pv_ = pv.view(-1, *pv.shape[-3:])
                h = self._as_tensor(base.vision_model(pixel_values=pv_))
                feats = self._as_tensor(base.connector(h)).reshape(-1, self.d)
        assert feats.shape[0] == C.K, (
            f"expected {C.K} global tokens, got {feats.shape[0]} — check "
            f"do_image_splitting=False and the SmolVLM2 tile token count")
        return feats.float().cpu().numpy()

    @staticmethod
    def _find_subseq(ids, sub):
        """First start index where `sub` (list) occurs in `ids` (list); else None."""
        n, m = len(ids), len(sub)
        for i in range(n - m + 1):
            if ids[i:i + m] == sub:
                return i
        return None

    # ------------------------------------------------------------------ #
    # public: encode + bundle
    # ------------------------------------------------------------------ #
    def encode_global(self, image):
        inputs = self._proc(self._prompt_text(""), image)
        return self._image_features(inputs)

    def build_bundle(self, image, question, answer):
        """Teacher-forced (image+question+answer) token bundle + G, with spans.

        We build the answer span by CONCATENATING TOKEN IDS (prompt ids + answer
        ids), NOT by re-tokenizing a `prompt + answer` STRING. String re-tokenizing
        could collapse the boundary and yield an empty answer span -> all-zero
        rollout labels. Token concat guarantees a non-empty ans_pos.
        """
        prompt = self._prompt_text(question)
        p_inputs = self._proc(prompt, image)               # image already expanded to K tokens
        p_ids = p_inputs["input_ids"][0]
        L_p = int(p_ids.shape[0])

        ans_ids = self.processor.tokenizer(
            answer + self.eos, add_special_tokens=False)["input_ids"]
        assert len(ans_ids) > 0, f"empty answer tokenization for: {answer!r}"
        ans_t = torch.tensor(ans_ids, dtype=p_ids.dtype, device=p_ids.device)
        full_ids = torch.cat([p_ids, ans_t]).unsqueeze(0)  # [1, L_p + |ans|]

        # reuse the prompt's image tensors; just extend ids + attention mask
        f_inputs = dict(p_inputs)
        f_inputs["input_ids"] = full_ids
        f_inputs["attention_mask"] = torch.ones_like(full_ids)

        ids = full_ids[0].tolist()
        T = len(ids)
        img_pos = [i for i, t in enumerate(ids) if t == self.image_token_id]
        assert len(img_pos) == C.K, f"got {len(img_pos)} image tokens, expected {C.K}"
        ans_pos = list(range(L_p, T))
        assert len(ans_pos) > 0, "empty answer span (ans_pos) — teacher labels would be all zero"

        # question token span: search for the question's token ids inside the prompt
        q_ids = self.processor.tokenizer(question, add_special_tokens=False)["input_ids"]
        start = self._find_subseq(ids[:L_p], q_ids)
        if start is not None:
            q_pos = list(range(start, start + len(q_ids)))
        else:  # fallback: text tokens after the image block, before the answer
            last_img = img_pos[-1]
            q_pos = [i for i in range(last_img + 1, L_p) if ids[i] != self.image_token_id]

        G = self._image_features(f_inputs)
        return {
            "inputs": f_inputs, "ids": ids, "T": T, "L_p": L_p,
            "img_pos": img_pos, "q_pos": q_pos, "ans_pos": ans_pos, "G": G,
            "_answer": answer,
        }

    # ------------------------------------------------------------------ #
    # Method A: attention rollout
    # ------------------------------------------------------------------ #
    def rollout_importance(self, bundle):
        """imp_answer, imp_question [K] from the SAME rollout matrix (Section 2)."""
        with torch.no_grad():
            out = self.model(**bundle["inputs"], output_attentions=True, use_cache=False)
        attns = out.attentions                              # tuple[L] of [1,H,T,T]
        T = bundle["T"]
        R = np.eye(T, dtype=np.float32)
        eye = np.eye(T, dtype=np.float32)
        for A in attns:
            a = A[0].float().mean(0).cpu().numpy()          # [T,T] mean over heads
            a = C.ROLLOUT_RESIDUAL * a + (1 - C.ROLLOUT_RESIDUAL) * eye
            a = a / a.sum(axis=-1, keepdims=True).clip(min=1e-9)
            R = a @ R                                        # rollout over layers
        img_pos = np.array(bundle["img_pos"])
        imp_answer = R[np.ix_(bundle["ans_pos"], img_pos)].sum(0)     # [K]
        imp_question = (R[np.ix_(bundle["q_pos"], img_pos)].sum(0)
                        if bundle["q_pos"] else np.zeros(C.K, np.float32))
        return imp_answer.astype(np.float32), imp_question.astype(np.float32)

    # ------------------------------------------------------------------ #
    # answer log-prob with an injected G (LOO + answer-preservation share this)
    # ------------------------------------------------------------------ #
    def _answer_logprob(self, bundle, G_np):
        """sum log P(answer tokens | image=G_np, question), teacher-forced."""
        inputs = bundle["inputs"]
        input_ids = inputs["input_ids"]
        emb = self.model.get_input_embeddings()(input_ids).clone()    # [1,T,d]
        G = torch.tensor(G_np, dtype=emb.dtype, device=emb.device)
        emb[0, bundle["img_pos"]] = G
        attn_mask = inputs.get("attention_mask")
        with torch.no_grad():
            out = self.model(inputs_embeds=emb, attention_mask=attn_mask, use_cache=False)
        logits = out.logits[0].float()                                # [T,V]
        logp = torch.log_softmax(logits, dim=-1)
        total = 0.0
        for pos in bundle["ans_pos"]:
            if pos == 0:
                continue
            total += float(logp[pos - 1, bundle["ids"][pos]])
        return total

    def loo_importance(self, bundle, G_np, cand_idx=None):
        """imp_loo[k] = relu(logp_full - logp_(mask k)). K fwd passes (Method B)."""
        logp_full = self._answer_logprob(bundle, G_np)
        imp = np.zeros(C.K, np.float32)
        idx = range(C.K) if cand_idx is None else cand_idx
        for k in idx:
            Gm = G_np.copy()
            Gm[k] = 0.0
            imp[k] = max(0.0, logp_full - self._answer_logprob(bundle, Gm))
        return imp

    # ------------------------------------------------------------------ #
    # answer-preservation @ budget (Exp 1 / Exp 5)
    # ------------------------------------------------------------------ #
    def answer_preservation(self, bundle, G_np, keep_idx):
        """Keep only `keep_idx` image tokens (others zeroed); compare to full G.

        Returns retained answer logprob (keep minus full; ~0 = preserved) and a
        greedy-generation string match against the gold answer.
        """
        keep = set(int(i) for i in keep_idx)
        G_keep = G_np.copy()
        drop = [k for k in range(C.K) if k not in keep]
        G_keep[drop] = 0.0

        logp_full = self._answer_logprob(bundle, G_np)
        logp_keep = self._answer_logprob(bundle, G_keep)
        gen_full = self._generate(bundle, G_np)
        gen_keep = self._generate(bundle, G_keep)
        gold = _norm(bundle_answer(bundle))
        return {
            "logp_full": logp_full,
            "logp_keep": logp_keep,
            "retained": logp_keep - logp_full,          # <=0; closer to 0 = better
            "gen_match_full": _match(gen_full, gold),
            "gen_match_keep": _match(gen_keep, gold),
        }

    def _generate(self, bundle, G_np, max_new=40):
        """Greedy generation from the prompt (answer span excluded) with injected G."""
        inputs = bundle["inputs"]
        prompt_ids = inputs["input_ids"][:, :bundle["L_p"]]
        emb = self.model.get_input_embeddings()(prompt_ids).clone()
        G = torch.tensor(G_np, dtype=emb.dtype, device=emb.device)
        # image tokens all fall within the prompt, so positions are valid
        img_pos = [p for p in bundle["img_pos"] if p < bundle["L_p"]]
        emb[0, img_pos] = G[: len(img_pos)]
        with torch.no_grad():
            gen = self.model.generate(inputs_embeds=emb, max_new_tokens=max_new, do_sample=False)
        return self.processor.tokenizer.decode(gen[0], skip_special_tokens=True)


# gold answer is carried on the bundle for convenience
def bundle_answer(bundle):
    return bundle.get("_answer", "")


def _norm(s):
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()


def _match(pred, gold_norm):
    p = _norm(pred)
    if not gold_norm:
        return 0
    # loose containment either way (answers are short phrases inside sentences)
    return int(gold_norm in p or p in gold_norm or _token_overlap(p, gold_norm) >= 0.6)


def _token_overlap(a, b):
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / max(1, len(sb))
